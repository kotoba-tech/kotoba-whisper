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
from datasets import utils, DatasetDict, IterableDataset, load_dataset
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
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    feature_extractor_name: Optional[str] = field(
        default=None,
        metadata={"help": "feature extractor name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    subfolder: str = field(
        default="",
        metadata={
            "help": "In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can"
            "specify the folder name here."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    train_dataset_name: str = field(
        default=None,
        metadata={
            "help": "The name of the training dataset to use (via the datasets library). Load and combine "
                    "multiple datasets by separating dataset ids by a '+' symbol. For example, to load LibriSpeech "
                    "and Common Voice, set `train_dataset_name='librispeech_asr+common_voice'`."
        },
    )
    train_dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the training dataset to use (via the datasets library). Load and combine "
            "multiple datasets by separating dataset configs by a '+' symbol. Note that the order of the configs should "
            "match the order of the datasets."
        },
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to cache directory for saving and loading datasets"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
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
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    timestamp_probability: float = field(
        default=0.2, metadata={"help": "Probability for training on timestamped tokens if the data contains it."}
    )
    return_timestamps: bool = field(
        default=False, metadata={"help": "Whether or not to predict timestamps in the generation step."}
    )
    language: str = field(
        default=None,
        metadata={
            "help": (
                "Language for multilingual distillation. This argument should be set for multilingual distillation "
                "only. For English speech recognition, it should be left as `None`."
            )
        },
    )
    task: str = field(
        default="transcribe",
        metadata={
            "help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."
            "This argument should be set for multilingual distillation only. For English speech recognition, it should be left as `None`."
        },
    )
    wandb_project: str = field(
        default="distil-whisper",
        metadata={"help": "The name of the wandb project."},
    )


@dataclass
class DistillationTrainingArguments(Seq2SeqTrainingArguments):
    freeze_encoder: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to freeze the entire encoder model. Only recommended when the entire encoder has been "
                "copied from the teacher model."
            )
        },
    )
    temperature: Optional[float] = field(
        default=2.0, metadata={"help": "Temperature to anneal the logits when computing the softmax."}
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
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]

        # dataloader returns a list of features which we convert to a dict
        input_features = {model_input_name: [feature[model_input_name] for feature in features]}
        label_features = {"input_ids": [feature["labels"] for feature in features]}

        # reformat list to dict and set to pytorch format
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.input_padding,
            return_tensors="pt",
        )

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


def log_metric(
    accelerator,
    metrics: Dict,
    train_time: float,
    step: int,
    epoch: int,
    learning_rate: float = None,
    prefix: str = "train",
):
    """Helper function to log all training/evaluation metrics with the correct prefixes and styling."""
    log_metrics = {}
    for k, v in metrics.items():
        log_metrics[f"{prefix}/{k}"] = v
    log_metrics[f"{prefix}/time"] = train_time
    log_metrics[f"{prefix}/epoch"] = epoch
    if learning_rate is not None:
        log_metrics[f"{prefix}/learning_rate"] = learning_rate
    accelerator.log(log_metrics, step=step)


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


def sorted_checkpoints(output_dir=None, checkpoint_prefix="checkpoint") -> List[str]:
    """Helper function to sort saved checkpoints from oldest to newest."""
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]

    for path in glob_checkpoints:
        regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
        if regex_match is not None and regex_match.groups() is not None:
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def rotate_checkpoints(save_total_limit=None, output_dir=None, checkpoint_prefix="checkpoint") -> None:
    """Helper function to delete old checkpoints."""
    if save_total_limit is None or save_total_limit <= 0:
        return
    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted_checkpoints(output_dir=output_dir, checkpoint_prefix=checkpoint_prefix)
    if len(checkpoints_sorted) <= save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
        shutil.rmtree(checkpoint, ignore_errors=True)


_RE_CHECKPOINT = re.compile(r"^checkpoint-(\d+)-epoch-(\d+)$")


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _RE_CHECKPOINT.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_RE_CHECKPOINT.search(x).groups()[0])))


def get_parameter_names(model, forbidden_layer_types, forbidden_module=None):
    """
    Returns the names of the model parameters that are not inside a forbidden layer or forbidden module.
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

    # 4. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

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
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        (model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
    )
    config = WhisperConfig.from_pretrained(
        (model_args.config_name if model_args.config_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
    )
    tokenizer = WhisperTokenizerFast.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
    )
    # override timestamp tokens until tokenizer issues are fixed in transformers
    timestamps = [AddedToken("<|%.2f|>" % (i * 0.02), lstrip=False, rstrip=False) for i in range(1500 + 1)]
    tokenizer.add_tokens(timestamps)
    teacher_model = WhisperForConditionalGeneration.from_pretrained(
        model_args.teacher_model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    student_model = WhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        subfolder=model_args.subfolder,
        token=model_args.token,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if student_model.config.decoder_start_token_id is None or teacher_model.config.decoder_start_token_id is None:
        raise ValueError(
            f"Make sure that `config.decoder_start_token_id` is correctly defined for both the "
            f"student and teacher model. Got {student_model.config.decoder_start_token_id} for the "
            f"student and {teacher_model.config.decoder_start_token_id} for the teacher."
        )
    share_hidden_states = training_args.freeze_encoder and student_model.config.d_model == teacher_model.config.d_model
    # enable gradient checkpointing if necessary
    if training_args.gradient_checkpointing:
        student_model.gradient_checkpointing_enable()
    # freeze student encoder if necessary
    if training_args.freeze_encoder:
        student_model.freeze_encoder()
        student_model.model.encoder.gradient_checkpointing = False
    if hasattr(teacher_model.generation_config, "is_multilingual") and teacher_model.generation_config.is_multilingual:
        # We need to set the language and task ids for previously multilingual checkpoints
        tokenizer.set_prefix_tokens(language=data_args.language, task=data_args.task, predict_timestamps=False)
        student_model.generation_config.update(**{"language": data_args.language, "task": data_args.task})
    elif data_args.language is not None:
        raise ValueError(
            "Setting language token for an English-only checkpoint is not permitted. The language argument should "
            "only be set for multilingual checkpoints."
        )

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
    training_datasets = DatasetDict(
        {
            "train": load_dataset(
                data_args.train_dataset_name,
                data_args.train_dataset_config_name,
                split=data_args.train_split_name,
                trust_remote_code=True,
                cache_dir=data_args.dataset_cache_dir,
                token=model_args.token,
                num_proc=data_args.preprocessing_num_workers
            )
        }
    )
    return_timestamps = data_args.return_timestamps if data_args.timestamp_probability > 0 else False
    decoder_start_token_id = student_model.config.decoder_start_token_id  # <|startoftranscript|>
    decoder_prev_token_id = tokenizer.all_special_ids[-3]  # <|startofprev|>

    # 9. Define Training Schedule to store some constants
    per_device_train_batch_size = int(training_args.per_device_train_batch_size)
    train_batch_size = per_device_train_batch_size * accelerator.num_processes
    gradient_accumulation_steps = int(training_args.gradient_accumulation_steps)
    if training_args.max_steps < 0:
        num_epochs = int(training_args.num_train_epochs)
        steps_per_epoch = len(training_datasets["train"]) // (train_batch_size * gradient_accumulation_steps)
        total_train_steps = steps_per_epoch * num_epochs
    elif training_args.max_steps > 0:
        logger.info("max_steps is given, it will override any value given in num_train_epochs")
        total_train_steps = int(training_args.max_steps)
        # Setting a very large number of epochs so we go as many times as necessary over the iterator.
        num_epochs = sys.maxsize
        steps_per_epoch = total_train_steps
    else:
        raise ValueError("max_steps must be specified when training with a streaming (iterable) dataset")

    # 10. Define optimizer, LR scheduler, collator
    decay_parameters = get_parameter_names(
        student_model,
        [nn.LayerNorm],
        forbidden_module=[student_model.model.encoder] if training_args.freeze_encoder else None,
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
    if data_args.max_label_length is not None:
        max_label_length = data_args.max_label_length
    else:
        max_label_length = student_model.config.max_length
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
    if hasattr(teacher_model.generation_config, "is_multilingual") and teacher_model.generation_config.is_multilingual:
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
            if share_hidden_states:
                # if the student and teacher share the same frozen encoder then we don't have to recompute the
                # encoder hidden-states for the teacher model, we can just re-use from the student
                encoder_outputs = BaseModelOutput(student_outputs.encoder_last_hidden_state)
                teacher_outputs = teacher_model(encoder_outputs=encoder_outputs, labels=batch["labels"])
            else:
                # do the full forward pass for the teacher model (encoder + decoder)
                teacher_outputs = teacher_model(**batch)
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
    logger.info(f"  Num examples = {total_train_steps * train_batch_size * gradient_accumulation_steps}")
    logger.info("  Instantaneous batch size per device =" f" {training_args.per_device_train_batch_size}")
    logger.info("  Gradient accumulation steps =" f" {gradient_accumulation_steps}")
    effective_batch_size = train_batch_size * gradient_accumulation_steps
    logger.info(f"  Total train batch size (w. parallel & distributed) = {effective_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")

    # ======================== Training ================================
    train_time = 0
    train_start = time.time()
    steps_trained_progress_bar = tqdm(
        range(total_train_steps), desc="Train steps ... ", position=0, disable=not accelerator.is_local_main_process
    )
    continue_training = True
    epochs_trained = 0
    cur_step = 0

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    if checkpoint is not None:
        accelerator.load_state(checkpoint)
        # Find num steps and epoch from saved state string pattern
        pattern = r"checkpoint-(\d+)-epoch-(\d+)"
        match = re.search(pattern, checkpoint)
        cur_step = int(match.group(1))
        epochs_trained = int(match.group(2))
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {cur_step}")
        steps_trained_progress_bar.update(cur_step)
        for epoch in range(0, epochs_trained):
            training_datasets["train"] = training_datasets["train"].shuffle(training_args.seed)
        if training_args.max_steps < 0:
            # we know exactly the number of steps per epoch, so can skip through the required number of batches
            resume_step = (cur_step - epochs_trained * steps_per_epoch) * gradient_accumulation_steps
        else:
            # Currently we don't know how many steps we've taken in the current epoch
            # So we just shuffle the dataset one extra time and start from a fresh epoch
            # This is "good enough" for our purposes but not fully correct
            resume_step = None
            training_datasets["train"] = training_datasets["train"].shuffle(training_args.seed)
    else:
        resume_step = None

    for epoch in range(epochs_trained, num_epochs):
        training_datasets["train"] = training_datasets["train"].shuffle(training_args.seed)
        train_dataloader = DataLoader(
            training_datasets["train"],
            collate_fn=data_collator,
            batch_size=per_device_train_batch_size,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )
        train_dataloader = accelerator.prepare(train_dataloader)
        if hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(epoch)

        if resume_step is not None:
            # Skip the first N batches in the dataloader when resuming from a checkpoint
            train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            resume_step = None

        for batch in train_dataloader:
            with accelerator.accumulate(student_model):
                loss, train_metric = train_step(batch, temperature=training_args.temperature)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(student_model.parameters(), training_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Check if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                steps_trained_progress_bar.update(1)
                cur_step += 1

                if cur_step % training_args.logging_steps == 0:
                    steps_trained_progress_bar.write(
                        f"Step... ({cur_step} / {total_train_steps} | Loss:"
                        f" {train_metric['loss']}, Learning Rate:"
                        f" {lr_scheduler.get_last_lr()[0]})"
                    )
                    log_metric(
                        accelerator,
                        metrics=train_metric,
                        learning_rate=lr_scheduler.get_last_lr()[0],
                        train_time=train_time + time.time() - train_start,
                        step=cur_step,
                        epoch=epoch,
                        prefix="train",
                    )

                # save checkpoint and weights after each save_steps and at the end of training
                if (cur_step % training_args.save_steps == 0) or cur_step == total_train_steps:
                    intermediate_dir = os.path.join(training_args.output_dir, f"checkpoint-{cur_step}-epoch-{epoch}")
                    accelerator.save_state(output_dir=intermediate_dir)
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        rotate_checkpoints(training_args.save_total_limit, output_dir=training_args.output_dir)

                        if cur_step == total_train_steps:
                            # un-wrap student model for save
                            student_model = accelerator.unwrap_model(student_model)
                            student_model.save_pretrained(training_args.output_dir)
                            # re-wrap student model for final eval
                            student_model = accelerator.prepare(student_model)

                        if training_args.push_to_hub:
                            repo.push_to_hub(
                                commit_message=f"Saving train state of step {cur_step}",
                                blocking=False,
                            )

                # break condition
                if cur_step == total_train_steps:
                    continue_training = False
                    break

        if not continue_training:
            break

    accelerator.end_training()


if __name__ == "__main__":
    main()
