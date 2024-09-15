"""Pseudo-labelling audio data using the Whisper model in preparation for distillation.
This script extends run_pseudo_labelling.py to allow
"""
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

from torch import bfloat16
from numpy import ndarray
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from datasets import DatasetDict, load_dataset, utils
from datasets.features import Audio
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


def safe_push(dataset_to_push, repo_name, config_name):
    while True:
        try:
            dataset_to_push.push_to_hub(repo_name, config_name=config_name)
            break
        except Exception as e:
            logger.warning(f"FAILED: push_to_hub on {repo_name} failed. wait 60 sec and retry soon...")
            logger.warning(repr(e))
            time.sleep(60)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to distill from."""
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained Whisper model or model identifier from huggingface.co/models"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    attn_implementation: str = field(default="sdpa", metadata={"help": "Attention implementation."})


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_split: str = field(default="train", metadata={"help": "Split name for the dataset."})
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
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
        default="transcription", metadata={"help": "The name of the dataset column containing the text data."},
    )
    task: str = field(default="transcribe", metadata={"help": "translate or transcribe"})
    language: str = field(default="en", metadata={"help": ("Language for multilingual distillation.")})

    id_column_name: str = field(
        default="name",
        metadata={"help": "The name of the dataset column containing the id data. Defaults to 'id'"},
    )
    max_label_length: int = field(
        default=128,
        metadata={"help": "Truncate transcriptions that are longer `max_label_length` tokens."},
    )
    num_chunks: int = field(default=10, metadata={"help": "Chunk size to split the dataset."},)
    return_timestamps: bool = field(
        default=True,
        metadata={"help": "Whether to return the timestamps with the text."},
    )


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`Wav2Vec2Processor`])
            The processor used for proccessing the data.
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
    text_lang_task: List[List[str]]
    input_padding: Union[bool, str] = "max_length"
    target_padding: Union[bool, str] = "max_length"
    max_target_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], ndarray]]]) -> Dict[str, ndarray]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        model_input_name = self.processor.model_input_names[0]
        # dataloader returns a list of features which we convert to a dict
        input_features = {model_input_name: [feature[model_input_name] for feature in features]}
        # reformat list to dict and set to pytorch format
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.input_padding,
            return_tensors="pt",
        )
        for text, lang, task in self.text_lang_task:
            label_features = {"input_ids": [feature[f"labels/{text}"] for feature in features]}
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
            batch[f"labels/{text}"] = labels
        return batch


def main():
    # 1. Parse input arguments to keep distinct sets of args, for cleaner separation of model/data/training related args
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.text_column_name = data_args.text_column_name.split(",")
    data_args.language = data_args.language.split(",")
    data_args.task = data_args.task.split(",")
    if len(data_args.language) != len(data_args.text_column_name) != len(data_args.task):
        raise ValueError(f"\n{data_args.language}\n{data_args.text_column_name}\n{data_args.task}")
    text_lang_task = list(zip(data_args.text_column_name, data_args.language, data_args.task))
    os.makedirs(training_args.output_dir, exist_ok=True)

    # 2. Initialize the accelerator
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with=None,
        project_dir=training_args.output_dir,
        kwargs_handlers=[kwargs],
    )
    accelerator.init_trackers(project_name="wandb_dummy")
    token = HfFolder().get_token()
    # 3. Set-up basic logging to create one log on every process with the configuration for debugging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    if accelerator.is_local_main_process:
        utils.logging.set_verbosity_warning()
    else:
        utils.logging.set_verbosity_error()
    logger.info(f"Training/evaluation parameters: {training_args}")
    logger.info(f"Tokenizer config: {text_lang_task}")
    # 4. Load pretrained model, tokenizer, and feature extractor
    config = WhisperConfig.from_pretrained(model_args.model_name_or_path, token=token)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_args.model_name_or_path, token=token)
    tokenizers = {}
    for text, lang, task in text_lang_task:
        tokenizer = WhisperTokenizerFast.from_pretrained(
            model_args.model_name_or_path, use_fast=model_args.use_fast_tokenizer, token=token
        )
        tokenizer.set_prefix_tokens(
            language=lang, task=task, predict_timestamps=data_args.return_timestamps
        )
        tokenizers[text] = tokenizer
    processor = WhisperProcessor.from_pretrained(model_args.model_name_or_path, token=token)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        token=token,
        low_cpu_mem_usage=True,
        torch_dtype=bfloat16,
        attn_implementation=model_args.attn_implementation,
    )
    model.eval()
    assert config.decoder_start_token_id is not None, "`config.decoder_start_token_id` is not correctly defined"
    assert hasattr(model.generation_config, "is_multilingual") and model.generation_config.is_multilingual, "model is not multilingual"
    max_label_length = data_args.max_label_length if data_args.max_label_length is not None else config.max_length
    model_input_name = feature_extractor.model_input_names[0]
    # Prepare everything with accelerate
    model = accelerator.prepare(model)
    num_beams = (
        training_args.generation_num_beams
        if training_args.generation_num_beams is not None
        else getattr(model.generation_config, "num_beams", 1)
    )
    gen_kwargs = {
        "max_length": max_label_length,
        "num_beams": num_beams,
        "return_timestamps": data_args.return_timestamps
    }
    effective_batch_size = training_args.per_device_eval_batch_size * accelerator.num_processes
    logger.info("***** Running Labelling *****")
    logger.info("  Instantaneous batch size per device =" f" {training_args.per_device_eval_batch_size}")
    logger.info(f"  Total eval batch size (w. parallel & distributed) = {effective_batch_size}")
    logger.info(f"  Predict labels with timestamps = {data_args.return_timestamps}")

    # 5. Load dataset
    logger.info(f"load dataset {data_args.dataset_name}")
    raw_datasets = DatasetDict({
        data_args.dataset_split: load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.dataset_split,
            trust_remote_code=True,
            token=token,
            num_proc=data_args.preprocessing_num_workers,
        )
    })
    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column_name, Audio(sampling_rate=feature_extractor.sampling_rate)
    )
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=config.decoder_start_token_id,  # <|startoftranscript|>
        input_padding="longest",
        target_padding="max_length",
        max_target_length=max_label_length,
        text_lang_task=text_lang_task
    )

    def prepare_dataset(batch):
        # process audio
        sample = batch[data_args.audio_column_name]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]
        # process targets
        for _text, _, _ in text_lang_task:
            batch[f"labels/{_text}"] = tokenizers[_text](batch[_text], max_length=max_label_length, truncation=True).input_ids
        return batch

    raw_datasets_features = list(next(iter(raw_datasets.values())).features.keys())
    size = len(raw_datasets[data_args.dataset_split])
    chunk_size = int(size/data_args.num_chunks)
    for chunk_id, start in enumerate(range(0, size, chunk_size)):
        end = min(start + chunk_size, size)
        logger.info(f"Processing chunk {chunk_id}: {start}->{end}")
        raw_datasets_tmp = raw_datasets[data_args.dataset_split].select(list(range(start, end)))
        vectorized_datasets = raw_datasets_tmp.map(
            prepare_dataset,
            remove_columns=raw_datasets_features,
            num_proc=data_args.preprocessing_num_workers,
            desc="preprocess dataset"
        )
        # Define Training Schedule
        loader = accelerator.prepare(
            DataLoader(
                vectorized_datasets,
                batch_size=training_args.per_device_eval_batch_size,
                collate_fn=data_collator,
                num_workers=training_args.dataloader_num_workers,
                pin_memory=True
            )
        )
        prediction = {n: [] for n in range(len(text_lang_task))}
        for step, batch in enumerate(tqdm(loader, disable=not accelerator.is_local_main_process)):
            input_features = batch["input_features"]
            # Generate predictions and pad to max generated length
            for n, (text, lang, task) in enumerate(text_lang_task):
                generated_ids = accelerator.unwrap_model(model).generate(
                    input_features.to(dtype=bfloat16),
                    language=lang,
                    task=task,
                    **gen_kwargs
                )
                generated_ids = accelerator.pad_across_processes(generated_ids, dim=1, pad_index=tokenizers[text].pad_token_id)
                generated_ids, _ = accelerator.gather_for_metrics((generated_ids, batch[f"labels/{text}"]))
                prediction[n].extend(generated_ids.cpu().numpy())
        accelerator.wait_for_everyone()
        for n, prediction in prediction.items():
            raw_datasets_tmp = raw_datasets_tmp.add_column(f"whisper_{text_lang_task[n][0]}", prediction)
        if accelerator.is_main_process:
            new_data = DatasetDict({data_args.dataset_split: raw_datasets_tmp})
            safe_push(new_data, training_args.hub_model_id, f"{data_args.dataset_config_name}.{chunk_id}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
