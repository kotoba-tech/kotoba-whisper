"""Training the Whisper model for sequence to sequence speech recognition via teacher-student distillation."""
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import utils, IterableDataset, load_dataset, concatenate_datasets
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AddedToken,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
    get_scheduler,
    set_seed,
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
    dataset_name_2: str = field(metadata={"help": "The name of the training dataset to use."})
    dataset_split_name_2: str = field(metadata={"help": "The name of the training data split to use."})
    dataset_config_name_2: str = field(metadata={"help": "The configuration name of the training dataset to use."})
    dataset_feature_2: str = field(metadata={"help": "The feature names for the labels."})
    dataset_language_2: str = field(metadata={"help": "Language for multilingual distillation."})
    dataset_task_2: str = field(metadata={"help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."})

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_label_length: int = field(
        default=128,
        metadata={"help": "Truncate transcriptions that are longer `max_label_length` tokens."},
    )
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
    """Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`Wav2Vec2Processor`]): The processor used for proccessing the data.
        decoder_start_token_id (:obj: `int`): The start-of-sequence token id of the decoder.
        decoder_prev_token_id (:obj: `int`): The start-of-prompt token id of the decoder
        max_target_length (:obj:`int`, `optional`): Maximum length of the ``labels`` of the returned list.
    """

    processor: Any
    decoder_start_token_id: int
    decoder_prev_token_id: int
    label_features: List[str]
    max_target_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, np.ndarray]:
        # input audio feature
        model_input_name = self.processor.model_input_names[0]  # "input_features"
        input_features = {model_input_name: [feature[model_input_name] for feature in features]}
        batch = self.processor.feature_extractor.pad(input_features, padding="longest", return_tensors="pt")
        # label feature
        # labels = {k: [feature[k] for feature in features] for k in self.label_features}
        # dataloader returns a list of features which we convert to a dict
        # reformat list to dict and set to pytorch format
        for k in self.label_features:
            label_features = {"input_ids": [feature[k] for feature in features]}
            labels_batch = self.processor.tokenizer.pad(
                label_features,
                max_length=self.max_target_length,
                padding="max_length",
                return_tensors="pt",
            )
            # shift labels to the right to get decoder input ids
            labels = labels_batch["input_ids"]
            decoder_input_ids = labels[:, :-1]
            labels = labels[:, 1:]
            labels_mask = labels_batch.attention_mask[:, 1:]
            # replace padding with -100 to ignore correctly when computing the loss
            labels = labels.masked_fill(labels_mask.ne(1), -100)
            # replace initial prompt tokens with -100 to ignore correctly when computing the loss
            bos_index = torch.argmax((labels == self.decoder_start_token_id).long(), dim=1)
            bos_index = torch.where(bos_index > 0, bos_index + 1, bos_index)
            prompt_mask = torch.arange(labels.shape[1]) < bos_index[:, None]
            labels = torch.where(prompt_mask, -100, labels)
            batch[f"labels/{k}"] = labels
            batch[f"decoder_input_ids/{k}"] = decoder_input_ids
            batch[f"tasks/{k}"] =
            batch[f"languages/{k}"] =
        return batch


def get_parameter_names(model, forbidden_layer_types, forbidden_module):
    """Returns the names of the model parameters that are not inside a forbidden layer or forbidden module.
    Can be used to get a subset of parameter names for decay masks, or to exclude parameters from an optimiser
    (e.g. if the module is frozen).
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
        assert training_args.hub_model_id
        # Create repo and retrieve repo_id
        repo_id = create_repo(training_args.hub_model_id, exist_ok=True, token=training_args.hub_token).repo_id
        # Clone repo locally
        repo = Repository(training_args.output_dir, clone_from=repo_id, token=training_args.hub_token)
        with open(os.path.join(training_args.output_dir, ".gitignore"), "w+") as gitignore:
            if "wandb" not in gitignore:
                gitignore.write("wandb\n")
    accelerator.wait_for_everyone()

    # 4. Load pretrained model, tokenizer, and feature extractor
    config = WhisperConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = WhisperTokenizerFast.from_pretrained(model_args.model_name_or_path, use_fast=True)
    # override timestamp tokens until tokenizer issues are fixed in transformers
    timestamps = [AddedToken("<|%.2f|>" % (i * 0.02), lstrip=False, rstrip=False) for i in range(1500 + 1)]
    tokenizer.add_tokens(timestamps)
    teacher_model = WhisperForConditionalGeneration.from_pretrained(
        model_args.teacher_model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_args.attn_implementation,
    )
    assert hasattr(teacher_model.generation_config, "is_multilingual") and teacher_model.generation_config.is_multilingual
    student_model = WhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_args.attn_implementation,
    )
    assert student_model.config.d_model == teacher_model.config.d_model
    assert student_model.config.decoder_start_token_id and teacher_model.config.decoder_start_token_id
    # enable gradient checkpointing if necessary
    if training_args.gradient_checkpointing:
        student_model.gradient_checkpointing_enable()
    # freeze student encoder if necessary
    student_model.freeze_encoder()
    student_model.model.encoder.gradient_checkpointing = False
    # We need to set the language and task ids for previously multilingual checkpoints
    # tokenizer.set_prefix_tokens(language=data_args.language, task=data_args.task, predict_timestamps=False)
    # student_model.generation_config.update(**{"language": data_args.language, "task": data_args.task})
    processor = WhisperProcessor.from_pretrained(training_args.output_dir)

    # 6. Preprocessing the datasets: we need to read the audio files as arrays and tokenize the targets.

    def get_dateset(name, split_name, config_name):
        tmp_dataset = []
        for c in config_name.split(","):
            tmp_dataset += load_dataset(
                name, c, split=split_name, trust_remote_code=True, num_proc=data_args.preprocessing_num_workers
            )
        return concatenate_datasets(tmp_dataset)

    def format_dataset_feature(feature, language, task):
        feature = feature.split(",")
        language = language.split(",")
        task = task.split(",")
        assert len(feature) == len(task) == len(language)
        return {k: [l, t] for k, l, t in zip(feature, language, task)}


    dataset_1 = get_dateset(data_args.dataset_name_1, data_args.dataset_split_1, data_args.dataset_config_name_1)
    feature_1 = format_dataset_feature(data_args.dataset_feature_1, data_args.dataset_language_1, data_args.dataset_task_1)
    dataset_2 = get_dateset(data_args.dataset_name_2, data_args.dataset_split_2, data_args.dataset_config_name_2)
    feature_2 = format_dataset_feature(data_args.dataset_feature_2, data_args.dataset_language_2, data_args.dataset_task_2)
    set_seed(training_args.seed)


    # 9. Define optimizer, LR scheduler, collator
    decay_parameters = get_parameter_names(
        student_model,
        [nn.LayerNorm],
        forbidden_module=[student_model.model.encoder],
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
    # LR scheduler gets stepped by `num_processes` each time -> account for this in warmup / total steps
    train_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes
    steps_per_epoch = len(training_datasets) // (train_batch_size * training_args.gradient_accumulation_steps)
    total_train_steps = steps_per_epoch * training_args.num_train_epochs
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps * accelerator.num_processes,
        num_training_steps=total_train_steps * accelerator.num_processes,
    )

    # 10. Define generation arguments - we need to do this before we wrap the models in DDP
    # so that we can still access the configs
    if training_args.generation_num_beams is not None:
        num_beams = training_args.generation_num_beams
    else:
        num_beams = getattr(student_model.generation_config, "num_beams", 1)
    gen_kwargs = {"max_length": data_args.max_label_length, "num_beams": num_beams, "return_timestamps": True}
    # forcing the language and task tokens helps multilingual models in their generations
    # gen_kwargs.update({"language": data_args.language, "task": data_args.task})

    # 11. Prepare everything with accelerate
    student_model, teacher_model, optimizer, lr_scheduler = accelerator.prepare(
        student_model, teacher_model, optimizer, lr_scheduler
    )
    student_model.train()
    teacher_model.eval()

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

    def train_step(batch):
        # Encoder output is shared across transcribe/translation and CE/KL loss.
        encoder_outputs = BaseModelOutput(student_model(input_ids=batch["input_ids"]).encoder_last_hidden_state)
        # Student model generation (transcription/translation).
        student_outputs_transcription = student_model(encoder_outputs=encoder_outputs, labels=batch[TBA])
        student_outputs_translation = student_model(encoder_outputs=encoder_outputs, labels=batch[TBA])
        # Cross-entropy loss.
        ce_transcription = student_outputs_transcription.loss
        ce_translation = student_outputs_translation.loss
        # Teacher model generation for KL loss.
        with torch.no_grad():
            teacher_outputs = teacher_model(encoder_outputs=encoder_outputs, labels=batch[TBA])
        # KL loss.
        kl_loss = kl_divergence(teacher_outputs.logits, student_outputs_transcription.logits, batch[TBA])
        # Use Distil-Whisper formulation (fix weight of CE loss and tune KL weight, 1 as default).
        loss = 0.8 * (ce_transcription + ce_translation) + training_args.kl_weight * kl_loss
        metrics = {"loss": loss, "loss/kl": kl_loss, "loss/ce_transcription": ce_transcription, "loss/ce_translation": ce_translation}
        return loss, metrics

    logger.info("***** Running training *****")
    logger.info(f" Num examples = {total_train_steps * train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f" Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f" Gradient accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f" Effective batch size (w. parallel & distributed) = {train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f" Total optimization steps = {total_train_steps}")
    train_start = time.time()
    steps_trained_progress_bar = tqdm(
        range(total_train_steps), desc="Train steps ... ", position=0, disable=not accelerator.is_local_main_process
    )
    cur_step = 0
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=student_model.config.decoder_start_token_id,  # <|startoftranscript|>
        decoder_prev_token_id=tokenizer.all_special_ids[-3],  # <|startofprev|>
        max_target_length=data_args.max_label_length,
    )

    for epoch in range(training_args.num_train_epochs):
        training_datasets = training_datasets.shuffle(training_args.seed)
        train_dataloader = DataLoader(
            training_datasets,
            collate_fn=data_collator,
            batch_size=training_args.per_device_train_batch_size,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )
        train_dataloader = accelerator.prepare(train_dataloader)
        if hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(epoch)
        for single_batch in train_dataloader:
            with accelerator.accumulate(student_model):
                loss, train_metric = train_step(single_batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(student_model.parameters(), training_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                steps_trained_progress_bar.update(1)
                cur_step += 1
                if cur_step % training_args.logging_steps == 0:
                    tmp_lr = lr_scheduler.get_last_lr()[0]
                    tmp_time = time.time() - train_start
                    steps_trained_progress_bar.write(
                        f"[Step {cur_step} / {total_train_steps}]: Loss:  {train_metric['loss']}, LR: {tmp_lr}"
                    )
                    train_metric.update({f"time": tmp_time, f"epoch": epoch, f"learning_rate": tmp_lr})
                    accelerator.log(train_metric, step=cur_step)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                logger.info("push_to_hub final model")
                accelerator.unwrap_model(student_model).save_pretrained(training_args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Saving train state of step {cur_step} (epoch: {epoch}): "
                                   f"{data_args.train_dataset_name}-{data_args.train_dataset_config_name}",
                    blocking=False,
                )
    logger.info("close the training job")
    accelerator.end_training()


if __name__ == "__main__":
    main()
