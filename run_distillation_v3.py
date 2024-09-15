import json
import logging
import os
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Union
from functools import partial
from pathlib import Path
from tqdm import tqdm
from shutil import rmtree

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import utils, load_dataset, concatenate_datasets
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from transformers import (
    WhisperConfig, WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizerFast, WhisperFeatureExtractor,
    set_seed, AddedToken, HfArgumentParser, Seq2SeqTrainingArguments,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# https://stackoverflow.com/questions/71692354/facing-ssl-error-with-huggingface-pretrained-models
os.environ['CURL_CA_BUNDLE'] = ''
# disable warning message
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.34.0.dev0")
require_version("datasets>=2.14.6", "To fix: `pip install --upgrade datasets`")
logger = get_logger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Student model."})
    teacher_model_name_or_path: str = field(metadata={"help": "Teacher model."})
    attn_implementation: str = field(default="sdpa", metadata={"help": "Attention implementation."})


@dataclass
class DataTrainingArguments:
    dataset_name_1: str = field(metadata={"help": "The name of the training dataset to use."})
    dataset_split_name_1: str = field(metadata={"help": "The name of the training data split to use."})
    dataset_config_name_1: str = field(metadata={"help": "The configuration name of the training dataset to use."})
    dataset_feature_1: str = field(metadata={"help": "The feature names for the labels."})
    dataset_language_1: str = field(metadata={"help": "Language for multilingual distillation."})
    dataset_task_1: str = field(metadata={"help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."})
    dataset_timestamp_1: str = field(metadata={"help": "Whether or not to predict timestamps."})
    dataset_kl_1: str = field(metadata={"help": "Whether or not to apply KL loss."})
    dataset_name_2: str = field(metadata={"help": "The name of the training dataset to use."})
    dataset_split_name_2: str = field(metadata={"help": "The name of the training data split to use."})
    dataset_config_name_2: str = field(metadata={"help": "The configuration name of the training dataset to use."})
    dataset_feature_2: str = field(metadata={"help": "The feature names for the labels."})
    dataset_language_2: str = field(metadata={"help": "Language for multilingual distillation."})
    dataset_task_2: str = field(metadata={"help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."})
    dataset_timestamp_2: str = field(metadata={"help": "Whether or not to predict timestamps."})
    dataset_kl_2: str = field(metadata={"help": "Whether or not to apply KL loss."})
    max_label_length: int = field(metadata={"help": "Truncate transcriptions that are longer `max_label_length`."})
    num_workers: Optional[int] = field(default=None, metadata={"help": "The number of processes for the preprocessing."})
    wandb_project: str = field(default="distil-whisper", metadata={"help": "The name of the wandb project."})


@dataclass
class DistillationTrainingArguments(Seq2SeqTrainingArguments):
    temperature: float = field(
        default=2.0,
        metadata={"help": "Temperature to anneal the logits when computing the softmax."}
    )
    kl_weight: float = field(
        default=1.0,
        metadata={"help": "Weighting assigned to the MSE loss in the KD formulation."},
    )


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    model_input_name: str
    feature_extractor: WhisperFeatureExtractor
    tokenizer: WhisperTokenizerFast
    decoder_start_token_id: int
    max_target_length: int
    feature: List[str]

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]):
        input_features = {self.model_input_name: [feature[self.model_input_name] for feature in features]}
        batch = self.feature_extractor.pad(input_features, padding="longest", return_tensors="pt")
        for k in self.feature:
            labels_batch = self.tokenizer.pad(
                {"input_ids": [feature[k] for feature in features]},
                max_length=self.max_target_length,
                padding="max_length",
                return_tensors="pt"
            )
            # shift labels to the right to get decoder input ids
            labels = labels_batch["input_ids"]
            batch[f"decoder_input_ids/{k}"] = labels[:, :-1]
            labels = labels[:, 1:]
            labels_mask = labels_batch.attention_mask[:, 1:]
            # replace padding with -100 to ignore correctly when computing the loss
            labels = labels.masked_fill(labels_mask.ne(1), -100)
            # replace initial prompt tokens with -100 to ignore correctly when computing the loss
            bos_index = torch.argmax((labels == self.decoder_start_token_id).long(), dim=1)
            bos_index = torch.where(bos_index > 0, bos_index + 1, bos_index)
            batch[f"labels/{k}"] = torch.where(torch.arange(labels.shape[1]) < bos_index[:, None], -100, labels)
        return batch


def get_parameter_names(model, forbidden_layer_types, forbidden_module):
    """ Returns the names of the model parameters that are not inside a forbidden layer or forbidden module.
    Can be used to get a subset of parameter names for decay masks, or to exclude parameters from an optimiser
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types, forbidden_module)
            if not (
                isinstance(child, tuple(forbidden_layer_types))
                or (child in tuple(forbidden_module) if forbidden_module is not None else False)
            )
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def main():
    # 1. Parse input arguments, keep distinct sets of args, for cleaner separation of model/data/training related args.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, DistillationTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    # 2. Initialize the accelerator and basic logging.
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with=training_args.report_to,
        project_dir=training_args.output_dir,
    )
    accelerator.init_trackers(project_name=data_args.wandb_project)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}."
    )
    if accelerator.is_local_main_process:
        utils.logging.set_verbosity_warning()
    else:
        utils.logging.set_verbosity_error()
    logger.info(f"Training/evaluation parameters {training_args}")

    # 3. Handle the repository creation
    if accelerator.is_main_process:
        repo_name = Path(training_args.output_dir).absolute().name
        repo_id = create_repo(repo_name, exist_ok=True, token=training_args.hub_token).repo_id
        repo = Repository(training_args.output_dir, clone_from=repo_id, token=training_args.hub_token)
        with open(os.path.join(training_args.output_dir, ".gitignore"), "w+") as gitignore:
            if "wandb" not in gitignore:
                gitignore.write("wandb\n")
    accelerator.wait_for_everyone()

    # 4. Load pretrained model, tokenizer, and feature extractor
    config = WhisperConfig.from_pretrained(model_args.model_name_or_path)
    processor = WhisperProcessor.from_pretrained(training_args.output_dir)
    # override timestamp tokens until tokenizer issues are fixed in transformers
    timestamps = [AddedToken("<|%.2f|>" % (i * 0.02), lstrip=False, rstrip=False) for i in range(1500 + 1)]
    processor.tokenizer.add_tokens(timestamps)
    teacher_model = WhisperForConditionalGeneration.from_pretrained(
        model_args.teacher_model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_args.attn_implementation,
    )
    teacher_model.generation_config.update(**{"max_length": data_args.max_label_length})
    assert hasattr(teacher_model.generation_config, "is_multilingual") and teacher_model.generation_config.is_multilingual
    student_model = WhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_args.attn_implementation,
    )
    assert hasattr(student_model.generation_config, "is_multilingual") and student_model.generation_config.is_multilingual
    assert student_model.config.d_model == teacher_model.config.d_model
    assert student_model.config.decoder_start_token_id and teacher_model.config.decoder_start_token_id
    decoder_start_token_id = student_model.config.decoder_start_token_id  # <|startoftranscript|>
    # enable gradient checkpointing if necessary
    if training_args.gradient_checkpointing:
        student_model.gradient_checkpointing_enable()
    # freeze student encoder if necessary
    student_model.freeze_encoder()
    student_model.model.encoder.gradient_checkpointing = False
    student_model.generation_config.update(**{"max_length": data_args.max_label_length})

    # 5. Define optimizer
    decay_parameters = get_parameter_names(
        student_model, [nn.LayerNorm], forbidden_module=[student_model.model.encoder]
    )
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [param for name, param in student_model.named_parameters() if name in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [param for name, param in student_model.named_parameters() if name not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )

    # 6. Prepare everything with accelerate
    student_model, teacher_model, optimizer = accelerator.prepare(student_model, teacher_model, optimizer)
    student_model.train()
    teacher_model.eval()

    # 7. Preprocessing the datasets

    def get_dateset(name, split_name, config_name):
        return concatenate_datasets(
            [load_dataset(name, n, split=split_name, trust_remote_code=True) for n in config_name.split(",")]
        )
        # return concatenate_datasets([dataset])
        # dataset = []
        # for c in config_name.split(","):
        #     dataset += load_dataset(name, c, split=split_name, trust_remote_code=True, num_proc=data_args.num_workers)
        # return concatenate_datasets(dataset)

    def format_dataset_feature(column, language, task, ts, kl):
        column = column.split(",")
        language = language.split(",")
        task = task.split(",")
        ts = [i == "true" for i in ts.split(",")]
        kl = [i == "true" for i in kl.split(",")]
        assert len(column) == len(task) == len(language) == len(ts) == len(kl)
        return {s: {"la": l, "col": c, "ts": t, "kl": k} for c, l, s, t, k in zip(column, language, task, ts, kl)}

    collator = partial(
        DataCollatorSpeechSeq2SeqWithPadding,
        model_input_name=processor.model_input_names[0],  # "input_features"
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        decoder_start_token_id=decoder_start_token_id,
        max_target_length=data_args.max_label_length,
    )
    dataset_1 = get_dateset(data_args.dataset_name_1, data_args.dataset_split_name_1, data_args.dataset_config_name_1)
    feature_1 = format_dataset_feature(
        data_args.dataset_feature_1,
        data_args.dataset_language_1,
        data_args.dataset_task_1,
        data_args.dataset_timestamp_1,
        data_args.dataset_kl_1
    )
    dataset_collator_1 = collator(feature=[i["col"] for i in feature_1.values()])
    dataset_2 = get_dateset(data_args.dataset_name_2, data_args.dataset_split_name_2, data_args.dataset_config_name_2)
    feature_2 = format_dataset_feature(
        data_args.dataset_feature_2,
        data_args.dataset_language_2,
        data_args.dataset_task_2,
        data_args.dataset_timestamp_2,
        data_args.dataset_kl_2
    )
    dataset_collator_2 = collator(feature=[i["col"] for i in feature_2.values()])

    # 8. Model distillation.
    dataset_size = min(len(dataset_1), len(dataset_2)) * 2
    train_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes
    steps_per_epoch = dataset_size // (train_batch_size * training_args.gradient_accumulation_steps)
    total_train_steps = int(steps_per_epoch * training_args.num_train_epochs)
    logger.info("***** Running training *****")
    logger.info(f" Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f" Gradient accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f" Effective batch size (w. parallel & distributed) = {train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f" Total optimization steps = {total_train_steps}")
    p_bar = tqdm(range(total_train_steps), desc="Training", position=0, disable=not accelerator.is_local_main_process)

    def kl_divergence(teacher_logit, student_logit, labels):
        # Rescale distribution by temperature to ensure gradients scale correctly.
        teacher_distribution = nn.functional.softmax(teacher_logit / training_args.temperature, dim=-1)
        # Log softmax of student predictions for numerical stability.
        student_distribution = nn.functional.log_softmax(student_logit / training_args.temperature, dim=-1)
        kl_loss = nn.KLDivLoss(reduction="none")
        divergence = kl_loss(student_distribution, teacher_distribution)
        # ignore padded tokens from divergence, i.e. where labels are not set to -100
        padding_mask = labels >= 0
        padding_mask = padding_mask.unsqueeze(-1)
        divergence = divergence * padding_mask
        # take the average over the mini-batch
        kl_loss = divergence.sum() / padding_mask.sum()
        # KL-divergence loss (scaled by temperature)
        return kl_loss * training_args.temperature ** 2

    def train_step(batch_1, batch_2):
        # Encoder output is shared across transcribe/translation and CE/KL loss.
        print(batch_1.keys())
        input_ids = torch.concat([batch_1["input_ids"], batch_2["input_ids"]])
        hidden_state = student_model(input_ids=input_ids).encoder_last_hidden_state
        # CE loss.
        metrics = defaultdict()
        for feature, batch in zip([feature_1, feature_2], [batch_1, batch_2]):
            for k, v in feature.items():
                gen_config = {"language": v["la"], "task": k, "return_timestamps": v["ts"]}
                student_model.generation_config.update(**gen_config)
                student_outputs = student_model(
                    encoder_outputs=BaseModelOutput(hidden_state[:len(batch["input_ids"])]),
                    labels=batch[f'labels/{v["col"]}'],
                    decoder_input_ids=batch[f'decoder_input_ids/{v["col"]}']
                )
                metrics[f"ce_loss.{k}.{v['la']}.return_timestamps=={v['ts']}"] += student_outputs.loss
                if v["kl"]:
                    # KL loss.
                    with torch.no_grad():
                        teacher_model.generation_config.update(**gen_config)
                        teacher_outputs = teacher_model(
                            encoder_outputs=BaseModelOutput(hidden_state[:len(batch["input_ids"])]),
                            labels=batch[f'labels/{v["col"]}']
                        )
                    metrics[f"kl_loss.{k}.{v['la']}.return_timestamps=={v['ts']}"] += kl_divergence(
                        teacher_outputs.logits,
                        student_outputs.logits,
                        batch[f'labels/{v["col"]}']
                    )
        # Use Distil-Whisper formulation (fix weight of CE loss and tune KL weight, 1 as default).
        ce_loss = sum(v for k, v in metrics.items() if k.startswith("ce_loss."))
        kl_loss = sum(v for k, v in metrics.items() if k.startswith("kl_loss."))
        metrics["loss"] = 0.8 * ce_loss + training_args.kl_weight * kl_loss
        return metrics["loss"], metrics

    cur_step = 0
    for epoch in range(int(training_args.num_train_epochs)):
        # Set up two data loaders for each dataset.
        dataset_1 = dataset_1.shuffle(training_args.seed)
        loader_1 = accelerator.prepare(
            DataLoader(
                dataset_1,
                collate_fn=dataset_collator_1,
                batch_size=int(training_args.per_device_train_batch_size / 2),
                num_workers=training_args.dataloader_num_workers,
                pin_memory=training_args.dataloader_pin_memory,
            )
        )
        # loader_1.dataset.set_epoch(epoch)
        dataset_2 = dataset_2.shuffle(training_args.seed)
        loader_2 = accelerator.prepare(
            DataLoader(
                dataset_2,
                collate_fn=dataset_collator_2,
                batch_size=int(training_args.per_device_train_batch_size / 2),
                num_workers=training_args.dataloader_num_workers,
                pin_memory=training_args.dataloader_pin_memory,
            )
        )
        # loader_2.dataset.set_epoch(epoch)
        # Use the 2nd loader as an iterator
        loader_2_iterator = iter(loader_2)
        for single_batch_1 in loader_1:
            try:
                single_batch_2 = next(loader_2_iterator)
            except StopIteration:
                break

            with accelerator.accumulate(student_model):
                loss, train_metric = train_step(single_batch_1, single_batch_2)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(student_model.parameters(), training_args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                p_bar.update(1)
                cur_step += 1
                if cur_step % training_args.logging_steps == 0:
                    p_bar.write(f"[{cur_step} / {total_train_steps}]\n{json.dumps(train_metric, indent=4)}")
                    accelerator.log(train_metric)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            logger.info(f"save_pretrained to {training_args.output_dir}")
            accelerator.unwrap_model(student_model).save_pretrained(training_args.output_dir)
            logger.info(f"push_to_hub to {repo_name}")
            repo.push_to_hub(
                commit_message=f"Saving train state of step {cur_step} (epoch: {epoch})",
                blocking=False,
            )
    logger.info("close the training job")
    home = os.path.expanduser('~')
    for dataset_name in [data_args.dataset_name_1, data_args.dataset_name_2]:
        for c in data_args.dataset_config_name_1.split(","):
            rmtree(f"{home}/.cache/huggingface/datasets/{dataset_name.replace('/', '___')}/{c}")
    rmtree(f"{home}/.cache/huggingface/datasets/downloads")
    accelerator.end_training()


if __name__ == "__main__":
    main()
