"""Evaluating a Whisper model on one or more short-form evaluation datasets."""
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from datasets import load_dataset, utils
from huggingface_hub import HfFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
)
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer, BasicTextNormalizer
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
    processor_name: Optional[str] = field(
        default=None,
        metadata={"help": "processor name or path if not the same as model_name"},
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
    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
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
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'."},
    )
    max_label_length: int = field(
        default=256,
        metadata={"help": "Truncate transcriptions that are longer `max_label_length` tokens."},
    )
    dataset_split_name: str = field(
        default="test",
        metadata={"help": "The name of the data set splits to use (via the datasets library)."},
    )
    wandb_project: str = field(
        default="distil-whisper",
        metadata={"help": "The name of the wandb project."},
    )
    max_samples_per_split: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples per split to this value if set."},
    )
    return_timestamps: bool = field(
        default=False,
        metadata={
            "help": "Whether to return the timestamps with the text. This enables the `FlaxWhisperTimestampsLogitsProcessor`."
        },
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


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`Wav2Vec2Processor`])
            The processor used for processing the data.
        decoder_start_token_id (:obj: `int`)
            The start-of-sequence token id of the decoder.
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
        # replace padding with -100 to ignore correctly when computing the loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main():
    # 1. Parse input arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Initialize the accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with=training_args.report_to,
        project_dir=training_args.output_dir,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=7200))],
    )
    accelerator.init_trackers(project_name=data_args.wandb_project)

    # 3. Set-up basic logging to create one log on every process with the configuration for debugging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}"
    )
    if accelerator.is_local_main_process:
        utils.logging.set_verbosity_warning()
    else:
        utils.logging.set_verbosity_error()
    logger.info("Training/evaluation parameters %s", training_args)

    # 4. Load dataset
    token = model_args.token if model_args.token is not None else HfFolder().get_token()
    raw_dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        split=data_args.dataset_split_name,
        cache_dir=data_args.dataset_cache_dir,
        token=token
    )

    # 5. Load pretrained model, tokenizer, and feature extractor
    config = WhisperConfig.from_pretrained(
        (model_args.config_name if model_args.config_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=token,
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        (model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=token,
    )
    tokenizer = WhisperTokenizerFast.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=token,
    )
    processor = WhisperProcessor.from_pretrained(
        (model_args.processor_name if model_args.processor_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=token,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        subfolder=model_args.subfolder,
        token=token,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    return_timestamps = data_args.return_timestamps
    if hasattr(model.generation_config, "is_multilingual") and model.generation_config.is_multilingual:
        # We need to set the language and task ids for multilingual checkpoints
        tokenizer.set_prefix_tokens(
            language=data_args.language, task="transcribe", predict_timestamps=return_timestamps
        )
    elif data_args.language is not None:
        raise ValueError(
            "Setting language token for an English-only checkpoint is not permitted. The language argument should "
            "only be set for multilingual checkpoints."
        )

    # 6. Resample speech dataset: `datasets` takes care of automatically loading and resampling the audio,
    # so we just need to set the correct target sampling rate.
    raw_dataset = raw_dataset.cast_column(
        data_args.audio_column_name,
        datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
    )

    # 7. Preprocessing the datasets.
    if data_args.max_label_length is not None:
        max_label_length = data_args.max_label_length
    else:
        max_label_length = model.config.max_length
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    dataloader_num_workers = training_args.dataloader_num_workers
    model_input_name = feature_extractor.model_input_names[0]
    if data_args.language is not None:
        normalizer = BasicTextNormalizer()
    else:
        normalizer = EnglishTextNormalizer(tokenizer.english_spelling_normalizer)
    if data_args.max_samples_per_split is not None:
        raw_dataset = raw_dataset.select(range(data_args.max_samples_per_split))

    def prepare_dataset(batch):
        # process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]
        # process targets
        batch["labels"] = tokenizer(batch[data_args.text_column_name], max_length=max_label_length, truncation=True).input_ids
        return batch

    vectorized_datasets = raw_dataset.map(
        prepare_dataset,
        remove_columns=[data_args.text_column_name],
        num_proc=num_workers,
        desc="preprocess dataset",
    )

    # 8. Load Metric
    metric = evaluate.load("wer")

    def compute_metrics(preds, labels):
        # replace padded labels by the padding token
        for idx in range(len(labels)):
            labels[idx][labels[idx] == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True, decode_with_timestamps=return_timestamps)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
        wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)
        # normalize everything and re-compute the WER
        norm_pred_str = [normalizer(pred) for pred in pred_str]
        norm_label_str = [normalizer(label) for label in label_str]
        # for logging, we need the pred/labels to match the norm_pred/norm_labels, so discard any filtered samples here
        pred_str = [pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
        label_str = [label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]
        # filtering step to only evaluate the samples that correspond to non-zero normalized references:
        norm_pred_str = [norm_pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
        norm_label_str = [norm_label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]
        wer = 100 * metric.compute(predictions=norm_pred_str, references=norm_label_str)
        return {"wer": wer, "wer_ortho": wer_ortho}, pred_str, label_str, norm_pred_str, norm_label_str

    # 9. Define Training Schedule
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,  # <|startoftranscript|>
        input_padding="longest",
        target_padding="max_length",
        max_target_length=max_label_length,
    )

    # 10. Define generation arguments - we need to do this before we wrap the models in DDP
    # so that we can still access the configs
    num_beams = (
        training_args.generation_num_beams
        if training_args.generation_num_beams is not None
        else getattr(model.generation_config, "num_beams", 1)
    )
    if training_args.generation_max_length is not None:
        max_new_tokens = training_args.generation_max_length
    else:
        max_new_tokens = max_label_length
    gen_kwargs = {"max_new_tokens": max_new_tokens, "num_beams": num_beams, "return_timestamps": return_timestamps}
    if hasattr(model.generation_config, "is_multilingual") and model.generation_config.is_multilingual:
        # forcing the language and task tokens helps multilingual models in their generations
        gen_kwargs.update({"language": data_args.language, "task": "transcribe"})

    # 11. Prepare everything with accelerate
    model = accelerator.prepare(model)
    logger.info("***** Running Evaluation *****")
    logger.info("  Instantaneous batch size per device =" f" {training_args.per_device_eval_batch_size}")
    effective_batch_size = training_args.per_device_eval_batch_size * accelerator.num_processes
    logger.info(f"  Total eval batch size (w. parallel & distributed) = {effective_batch_size}")
    logger.info(f"  Predict labels with timestamps = {return_timestamps}")

    # ======================== Evaluating ==============================
    eval_preds = []
    eval_labels = []
    eval_start = time.time()
    eval_loader = DataLoader(
        vectorized_datasets,
        batch_size=per_device_eval_batch_size,
        collate_fn=data_collator,
        num_workers=dataloader_num_workers,
        pin_memory=True,
    )

    eval_loader = accelerator.prepare(eval_loader)
    batches = tqdm(eval_loader, desc=f"Evaluating {data_args.dataset_split_name}...",
                   disable=not accelerator.is_local_main_process)

    for step, batch in enumerate(batches):
        # Generate predictions and pad to max generated length
        generate_fn = model.module.generate if accelerator.num_processes > 1 else model.generate
        generated_ids = generate_fn(batch["input_features"].to(dtype=torch.bfloat16), **gen_kwargs)
        generated_ids = accelerator.pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id)
        # Gather all predictions and targets
        generated_ids, labels = accelerator.gather_for_metrics((generated_ids, batch["labels"]))
        eval_preds.extend(generated_ids.cpu().numpy())
        eval_labels.extend(labels.cpu().numpy())

    accelerator.wait_for_everyone()
    eval_time = time.time() - eval_start

    # compute WER metric for eval sets
    wer_metric, pred_str, label_str, norm_pred_str, norm_label_str = compute_metrics(eval_preds, eval_labels)
    wer_desc = " ".join([f"Eval {key}: {value} |" for key, value in wer_metric.items()])
    logger.info(wer_desc)

    # Save metrics
    log_metrics = {}
    for k, v in wer_metric.items():
        log_metrics[f"{data_args.dataset_split_name}/{k}"] = v
    log_metrics[f"{data_args.dataset_split_name}/time"] = eval_time
    accelerator.log(log_metrics)
    accelerator.wait_for_everyone()
    accelerator.end_training()

    # Save predictions + metrics
    with open(f"{training_args.output_dir}/metric.{data_args.dataset_split_name}.json", "w") as f:
        json.dump(wer_metric, f)
    str_data = [[label_str[i], pred_str[i], norm_label_str[i], norm_pred_str[i]] for i in range(len(pred_str))]
    pd.DataFrame(str_data, columns=["Target", "Pred", "Norm Target", "Norm Pred"]).to_csv(
        f"{training_args.output_dir}/all_predictions.{data_args.dataset_split_name}.csv"
    )
    str_data = np.asarray(str_data)
    str_data_incorrect = str_data[str_data[:, -2] != str_data[:, -1]]
    pd.DataFrame(str_data_incorrect, columns=["Target", "Pred", "Norm Target", "Norm Pred"]).to_csv(
        f"{training_args.output_dir}/incorrect_predictions.{data_args.dataset_split_name}.csv"
    )


if __name__ == "__main__":
    main()
