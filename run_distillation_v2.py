"""Training the Whisper model for sequence to sequence speech recognition via teacher-student distillation."""
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import utils, DatasetDict, IterableDataset, load_dataset, concatenate_datasets
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AddedToken,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperFeatureExtractor,
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
    """Arguments pertaining to which model/config/tokenizer we are going to distill from."""
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained Whisper model or model identifier from huggingface.co/models"}
    )
    teacher_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained teacher model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    attn_implementation: str = field(default="sdpa", metadata={"help": "Attention implementation."})


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    train_dataset_name: str = field(
        metadata={"help": "The name of the training dataset to use (via the datasets library)."},
    )
    train_dataset_config_name: str = field(
        metadata={"help": "The configuration name of the training dataset to use (via the datasets library)."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing if using non-streaming mode."},
    )
    max_label_length: int = field(
        default=128,
        metadata={"help": "Truncate transcriptions that are longer `max_label_length` tokens."},
    )
    train_split_name: str = field(
        default="train",
        metadata={"help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"},
    )
    timestamp_probability: float = field(
        default=0.2,
        metadata={"help": "Probability for training on timestamped tokens if the data contains it."}
    )
    return_timestamps: bool = field(
        default=False,
        metadata={"help": "Whether or not to predict timestamps in the generation step."}
    )
    language: str = field(
        default=None,
        metadata={"help": "Language for multilingual distillation."},
    )
    task: str = field(
        default="transcribe",
        metadata={"help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."},
    )
    wandb_project: str = field(
        default="distil-whisper",
        metadata={"help": "The name of the wandb project."},
    )


@dataclass
class DistillationTrainingArguments(Seq2SeqTrainingArguments):
    temperature: Optional[float] = field(
        default=2.0,
        metadata={"help": "Temperature to anneal the logits when computing the softmax."}
    )
    kl_weight: Optional[float] = field(
        default=1.0,
        metadata={
            "help": (
                "Weighting assigned to the MSE loss in the KD formulation. MSE loss is "
                "computed between the teacher-student hidden states and attentions."
            )
        },
    )


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`Wav2Vec2Processor`])
            The processor used for proccessing the data.
        decoder_start_token_id (:obj: `int`)
            The start-of-sequence token id of the decoder.
        decoder_prev_token_id (:obj: `int`)
            The start-of-prompt token id of the decoder
        input_padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned input sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can statistics a batch with sequences of
              different lengths).
        target_padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned target sequences (according to the model's padding side and padding index).
            See above for details.
        max_target_length (:obj:`int`, `optional`):
            Maximum length of the ``labels`` of the returned list and optionally padding length (see above).
    """

    processor: Any
    decoder_start_token_id: int
    decoder_prev_token_id: int
    input_padding: Union[bool, str] = "max_length"
    target_padding: Union[bool, str] = "max_length"
    max_target_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, np.ndarray]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        model_input_name = self.processor.model_input_names[0]  # "input_features"

        # dataloader returns a list of features which we convert to a dict
        input_features = {model_input_name: [feature[model_input_name] for feature in features]}
        label_features = {"input_ids": [feature["labels"] for feature in features]}

        # reformat list to dict and set to pytorch format
        batch = self.processor.feature_extractor.pad(input_features, padding=self.input_padding, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            max_length=self.max_target_length,
            padding=self.target_padding,
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

        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids
        return batch


def get_layers_to_supervise(student_layers: int, teacher_layers: int) -> Dict:
    """Helper function to map the student layer i to the teacher layer j whose statistics we'd like them to emulate. Used
    for MSE loss terms in distillation (hidden-states and activations). Student layers are paired with teacher layers
    in equal increments, e.g. for a 12-layer model distilled to a 3-layer model, student layer 0 emulates teacher layer
    3 (such that it behaves like the first 4 teacher layers), student layer 1 emulates teacher layer 7, and student layer
    2 emulates teacher layer 11. This mapping is summarised by the dictionary: {0: 3, 1: 7, 2: 11}, which is precisely
    the statistics of this function for the arguments (student_layers=3, teacher_layers=12)."""
    layer_intervals = np.linspace(teacher_layers // student_layers - 1, teacher_layers - 1, student_layers, dtype=int)
    layer_intervals[-1] = teacher_layers - 1
    layer_map = {}

    for student_layer, teacher_layer in enumerate(layer_intervals):
        layer_map[student_layer] = teacher_layer

    return layer_map



_RE_CHECKPOINT = re.compile(r"^checkpoint-(\d+)-epoch-(\d+)$")


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
    # 1. Parse input arguments, keep distinct sets of args, for cleaner separation of model/data/training related args
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, DistillationTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Initialize the accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with=training_args.report_to,
        project_dir=training_args.output_dir,
    )
    accelerator.init_trackers(project_name=data_args.wandb_project)

    # 3. Set-up basic logging
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
    logger.info("Training/evaluation parameters %s", training_args)

    last_checkpoint = None

    # 5. Handle the repository creation
    if accelerator.is_main_process:
        if training_args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = training_args.hub_model_id
            if repo_name is None:
                repo_name = Path(training_args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            repo_id = create_repo(repo_name, exist_ok=True, token=training_args.hub_token).repo_id
            # Clone repo locally
            repo = Repository(training_args.output_dir, clone_from=repo_id, token=training_args.hub_token)
            with open(os.path.join(training_args.output_dir, ".gitignore"), "w+") as gitignore:
                if "wandb" not in gitignore:
                    gitignore.write("wandb\n")
        elif training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # 6. Load pretrained model, tokenizer, and feature extractor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_args.model_name_or_path)
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
    if student_model.config.decoder_start_token_id is None or teacher_model.config.decoder_start_token_id is None:
        raise ValueError(
            f"Make sure that `config.decoder_start_token_id` is correctly defined for both the "
            f"student and teacher model. Got {student_model.config.decoder_start_token_id} for the "
            f"student and {teacher_model.config.decoder_start_token_id} for the teacher."
        )
    # enable gradient checkpointing if necessary
    if training_args.gradient_checkpointing:
        student_model.gradient_checkpointing_enable()
    # freeze student encoder if necessary
    student_model.freeze_encoder()
    student_model.model.encoder.gradient_checkpointing = False
    # We need to set the language and task ids for previously multilingual checkpoints
    tokenizer.set_prefix_tokens(language=data_args.language, task=data_args.task, predict_timestamps=False)
    student_model.generation_config.update(**{"language": data_args.language, "task": data_args.task})

    # 7. Create a single speech processor - make sure all processes wait until data is saved
    if accelerator.is_main_process:
        feature_extractor.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)
        student_model.generation_config.save_pretrained(training_args.output_dir)
    accelerator.wait_for_everyone()
    processor = WhisperProcessor.from_pretrained(training_args.output_dir)

    # 8. Preprocessing the datasets: we need to read the audio files as arrays and tokenize the targets.
    set_seed(training_args.seed)
    training_datasets = []
    for single_config in data_args.train_dataset_config_name.split(","):
        training_datasets.append(
            load_dataset(
                data_args.train_dataset_name,
                single_config,
                split=data_args.train_split_name,
                trust_remote_code=True,
                num_proc=data_args.preprocessing_num_workers
            )
        )
    training_datasets = concatenate_datasets(training_datasets)
    return_timestamps = data_args.return_timestamps if data_args.timestamp_probability > 0 else False
    decoder_start_token_id = student_model.config.decoder_start_token_id  # <|startoftranscript|>
    decoder_prev_token_id = tokenizer.all_special_ids[-3]  # <|startofprev|>

    # 9. Define Training Schedule to store some constants
    per_device_train_batch_size = int(training_args.per_device_train_batch_size)
    train_batch_size = per_device_train_batch_size * accelerator.num_processes
    gradient_accumulation_steps = int(training_args.gradient_accumulation_steps)
    steps_per_epoch = len(training_datasets) // (train_batch_size * gradient_accumulation_steps)
    total_train_steps = steps_per_epoch * training_args.num_train_epochs

    # 10. Define optimizer, LR scheduler, collator
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
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps * accelerator.num_processes,
        num_training_steps=total_train_steps * accelerator.num_processes,
    )
    max_label_length = data_args.max_label_length if data_args.max_label_length else student_model.config.max_length
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=decoder_start_token_id,
        decoder_prev_token_id=decoder_prev_token_id,
        input_padding="longest",
        target_padding="max_length",
        max_target_length=max_label_length,
    )

    # 11. Define generation arguments - we need to do this before we wrap the models in DDP
    # so that we can still access the configs
    if training_args.generation_num_beams is not None:
        num_beams = training_args.generation_num_beams
    else:
        num_beams = getattr(student_model.generation_config, "num_beams", 1)
    gen_kwargs = {"max_length": max_label_length, "num_beams": num_beams, "return_timestamps": return_timestamps}
    # forcing the language and task tokens helps multilingual models in their generations
    gen_kwargs.update({"language": data_args.language, "task": data_args.task})

    # 12. Prepare everything with accelerate
    student_model, teacher_model, optimizer, lr_scheduler = accelerator.prepare(
        student_model, teacher_model, optimizer, lr_scheduler
    )

    def kl_divergence(target_distribution, log_predicted_distribution, labels):
        kl_loss = nn.KLDivLoss(reduction="none")
        divergence = kl_loss(log_predicted_distribution, target_distribution)
        # ignore padded tokens from divergence, i.e. where labels are not set to -100
        padding_mask = labels >= 0
        padding_mask = padding_mask.unsqueeze(-1)
        divergence = divergence * padding_mask
        # take the average over the mini-batch
        return divergence.sum() / padding_mask.sum()

    # Define gradient update step fn
    def train_step(batch, temperature=2.0):
        student_model.train()
        teacher_model.eval()
        student_outputs = student_model(**batch)
        with torch.no_grad():
            # if the student and teacher share the same frozen encoder then we don't have to recompute the
            # encoder hidden-states for the teacher model, we can just re-use from the student
            encoder_outputs = BaseModelOutput(student_outputs.encoder_last_hidden_state)
            teacher_outputs = teacher_model(encoder_outputs=encoder_outputs, labels=batch["labels"])
        # CE (data) loss
        ce_loss = student_outputs.loss
        # rescale distribution by temperature to ensure gradients scale correctly
        teacher_distribution = nn.functional.softmax(teacher_outputs.logits / temperature, dim=-1)
        # log softmax of student predictions for numerical stability
        student_distribution = nn.functional.log_softmax(student_outputs.logits / temperature, dim=-1)
        # KL-divergence loss (scaled by temperature)
        kl_loss = kl_divergence(teacher_distribution, student_distribution, batch["labels"]) * temperature**2
        # use Distil-Whisper formulation (fix weight of CE loss and tune KL weight, 1 as default)
        loss = 0.8 * ce_loss + training_args.kl_weight * kl_loss
        metrics = {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss}
        return loss, metrics

    logger.info("***** Running training *****")
    logger.info(f" Num examples = {total_train_steps * train_batch_size * gradient_accumulation_steps}")
    logger.info(f" Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f" Gradient accumulation steps = {gradient_accumulation_steps}")
    logger.info(f" Effective batch size (w. parallel & distributed) = {train_batch_size * gradient_accumulation_steps}")
    logger.info(f" Total optimization steps = {total_train_steps}")

    # ======================== Training ================================
    train_start = time.time()
    steps_trained_progress_bar = tqdm(
        range(total_train_steps), desc="Train steps ... ", position=0, disable=not accelerator.is_local_main_process
    )
    cur_step = 0
    for epoch in range(training_args.num_train_epochs):
        training_datasets = training_datasets.shuffle(training_args.seed)
        train_dataloader = DataLoader(
            training_datasets,
            collate_fn=data_collator,
            batch_size=per_device_train_batch_size,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )
        train_dataloader = accelerator.prepare(train_dataloader)
        if hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(epoch)
        for batch in train_dataloader:
            with accelerator.accumulate(student_model):
                loss, train_metric = train_step(batch, temperature=training_args.temperature)
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
