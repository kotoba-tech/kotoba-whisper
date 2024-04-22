"""Filter the dataset based on WER, convert into Mel spectrogram, and upload to huggingface."""
import argparse
import re
import os
import time
from functools import partial
import datasets
import evaluate
import numpy as np
from datasets import DatasetDict, load_dataset
from transformers import WhisperTokenizerFast, WhisperFeatureExtractor, AddedToken, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer

# https://stackoverflow.com/questions/71692354/facing-ssl-error-with-huggingface-pretrained-models
os.environ['CURL_CA_BUNDLE'] = ''

# disable warning message
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def safe_push(dataset_to_push, repo_name, config_name):
    while True:
        try:
            dataset_to_push.push_to_hub(repo_name, config_name=config_name)
            break
        except Exception:
            print(f"FAILED: push_to_hub on {repo_name} failed. wait 60 sec and retry soon...")
            time.sleep(60)


def main():

    parser = argparse.ArgumentParser(description='Preprocessing dataset.')
    parser.add_argument(
        '-d', '--dataset_name', default="asahi417/whisper_transcriptions.reazonspeech.large", type=str,
        help="The name of the training dataset to use (via the datasets library). Load and combine "
             "multiple datasets by separating dataset ids by a '+' symbol. For example, to load LibriSpeech "
             "and Common Voice, set `train_dataset_name='librispeech_asr+common_voice'`."
    )
    parser.add_argument(
        '--dataset_config_name', default="large", type=str,
        help="The configuration name of the training dataset to use (via the datasets library). Load and combine "
             "multiple datasets by separating dataset configs by a '+' symbol. Note that the order of the configs should "
             "match the order of the datasets."
    )
    parser.add_argument(
        '--task', default="transcribe", type=str,
        help="Task, either `transcribe` for speech recognition or `translate` for speech translation."
             "This argument should be set for multilingual distillation only. For English speech recognition, "
             "it should be left as `None`."
    )
    parser.add_argument(
        '--wer_threshold', default=10, type=float,
        help="Filter training data with Whisper transcriptions that have greater than `wer_threshold` "
             "WER with the normalised transcriptions. This only takes effect if training on pseudo-labels targets."
             "If `--use_pseudo_labels=False`, then no WER filtering is performed, since we train directly on the text"
             "transcriptions."
    )
    parser.add_argument(
        '--text_column_name', default="transcription", type=str,
        help="The name of the dataset column containing the text data in the training set."
    )
    parser.add_argument(
        '--audio_column_name', default="audio", type=str,
        help="The name of the dataset column containing the audio data. Defaults to 'audio'"
    )
    parser.add_argument(
        '--split', default="train", type=str,
        help="The name of the training data set split to use (via the datasets library). Defaults to 'train'"
    )
    parser.add_argument(
        '--language', default="ja", type=str,
        help="Language for multilingual distillation. This argument should be set for multilingual distillation "
             "only. For English speech recognition, it should be left as `None`."
    )
    parser.add_argument(
        '--model', default="openai/whisper-large-v3", type=str,
        help="Path to pretrained teacher model or model identifier from huggingface.co/models"
    )
    parser.add_argument(
        '--preprocessing_num_workers', default=128, type=int,
        help="The number of processes to use for the preprocessing if using non-streaming mode."
    )
    parser.add_argument(
        '--preprocessing_batch_size', default=256, type=int,
        help="Number of examples per batch provided to the `prepare_dataset` function."
    )
    parser.add_argument(
        '--condition_on_prev_probability', default=0.2, type=float,
        help="Probability for conditioning on the previous text example."
    )
    parser.add_argument(
        '--max_label_length', default=128, type=int,
        help="Truncate transcriptions that are longer `max_label_length` tokens."
    )
    parser.add_argument(
        '--is_english_model', action="store_true",
        help="Whether the whisper model is multilingual or English-only."
    )
    parser.add_argument(
        '--timestamp_probability', default=0.2, type=float,
        help="Probability for training on timestamped tokens if the data contains it."
    )
    parser.add_argument(
        '--skip_filtering', action="store_true",
        help="Skip WER filtering part."
    )
    parser.add_argument(
        '--skip_logmel', action="store_true",
        help="Skip Logmel conversion part."
    )
    parser.add_argument(
        '--max_duration_in_seconds', default=30.0, type=float,
        help="Filter audio files that are longer than `max_duration_in_seconds` seconds"
    )
    parser.add_argument(
        '--min_duration_in_seconds', default=0.0, type=float,
        help="Filter audio files that are shorter than `min_duration_in_seconds` seconds"
    )

    arg = parser.parse_args()

    dataset = load_dataset(
        arg.dataset_name,
        arg.dataset_config_name,
        split=arg.split,
        num_proc=arg.preprocessing_num_workers,
        trust_remote_code=True,
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(arg.model)
    repo_name = f"{arg.dataset_name}.wer_{arg.wer_threshold}"

    if not arg.skip_filtering:
        #################
        # WER Filtering #
        #################
        metric = evaluate.load("wer")
        tokenizer = WhisperTokenizerFast.from_pretrained(arg.model)
        # override timestamp tokens until tokenizer issues are fixed in transformers
        timestamps = [AddedToken("<|%.2f|>" % (i * 0.02), lstrip=False, rstrip=False) for i in range(1500 + 1)]
        tokenizer.add_tokens(timestamps)
        if arg.language != "en":
            normalizer = BasicTextNormalizer()
            tokenizer.set_prefix_tokens(language=arg.language, task=arg.task, predict_timestamps=False)
        else:
            normalizer = EnglishTextNormalizer(tokenizer.english_spelling_normalizer)

        # We need to set the language and task ids for previously multilingual checkpoints
        teacher_model = WhisperForConditionalGeneration.from_pretrained(arg.model)
        if hasattr(teacher_model.generation_config, "is_multilingual") and teacher_model.generation_config.is_multilingual:
            # We need to set the language and task ids for previously multilingual checkpoints
            tokenizer.set_prefix_tokens(language=arg.language, task=arg.task, predict_timestamps=False)
            timestamp_position = 3
        else:
            timestamp_position = 1

        def is_wer_in_range(ground_truth, whisper_transcript):
            try:
                norm_ground_truth = normalizer(ground_truth)
                if (
                        isinstance(whisper_transcript, str)
                        and whisper_transcript.startswith("[")
                        and whisper_transcript.endswith("]")
                ):
                    whisper_transcript = re.findall(r"\d+", whisper_transcript)
                    whisper_transcript = [int(token) for token in whisper_transcript]
                if isinstance(whisper_transcript, list):
                    whisper_transcript = tokenizer.decode(whisper_transcript, skip_special_tokens=True)
                if len(norm_ground_truth) > 0 and whisper_transcript is not None:
                    norm_whisper_transcript = normalizer(whisper_transcript)
                    wer = 100 * metric.compute(predictions=[norm_whisper_transcript], references=[norm_ground_truth])
                    return wer < arg.wer_threshold
                else:
                    # filter automatically since we can't know the WER
                    return False
            except Exception:
                return False

        raw_datasets = DatasetDict()
        dataset = dataset.cast_column("audio", datasets.features.Audio(feature_extractor.sampling_rate))
        columns_to_keep = {"audio", "text", "whisper_transcript"}
        dataset = dataset.rename_column(arg.text_column_name, "text")
        dataset_features = dataset.features.keys()
        raw_datasets["train"] = dataset.remove_columns(set(dataset_features - columns_to_keep))
        raw_datasets = raw_datasets.cast_column(arg.audio_column_name,
                                                datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate))

        filter_by_wer_threshold = partial(
            raw_datasets["train"].filter, function=is_wer_in_range, input_columns=["text", "whisper_transcript"],
        )
        raw_datasets["train"] = filter_by_wer_threshold(
            num_proc=arg.preprocessing_num_workers, desc="filtering train dataset by wer"
        )

        #################
        # Preprocessing #
        #################
        # Pre-process the raw dataset in a three stage process:
        #   1. Convert the audio arrays to log-mel spectrogram inputs
        #   2. Possibly filter the timestamp tokens from the token ids (depending on the timestamp probability)
        #   3. Possibly add prompt tokens if conditioning on previous text (depending on the conditioning probability)
        timestamp_ids = tokenizer.timestamp_ids()
        timestamp_begin = tokenizer.all_special_ids[-1]
        decoder_prev_token_id = tokenizer.all_special_ids[-3]  # <|startofprev|>
        decoder_eot_token_id = tokenizer.eos_token_id

        def has_timestamp_tokens(input_str):
            """
            Identify whether the input string contains timestamp tokens, of the form <|0.00|>, by searching for
            pairs of left and right-angle brackets.
            """
            return bool(re.search("\<[^\>]*\>", input_str))

        def prepare_train_dataset(batch):
            """
            Pre-process the raw dataset in a three stage process:
                1. Possibly filter the timestamp tokens from the token ids (depending on the timestamp probability)
                2. Possibly add prompt tokens if conditioning on previous text (depending on the conditioning probability)
            """
            # process audio input
            audio = [sample["array"] for sample in batch["audio"]]
            batch["input_length"] = [len(sample) for sample in audio]

            # process text targets - for training these are the Whisper-generated pseudo-labels
            input_str_batched = batch["whisper_transcript"]

            all_token_ids = []
            all_token_ids_unprompted = []
            for input_str in input_str_batched:
                if isinstance(input_str, list):
                    # pseudo-labelled transcriptions have been retained as token ids (`decode_token_ids=False`)
                    token_ids = input_str
                elif input_str[0].startswith("[") and input_str[0].endswith("]"):
                    token_ids = re.findall(r"\d+", input_str)
                    token_ids = [int(token) for token in token_ids]
                else:
                    token_ids = None

                if token_ids is not None:
                    # remove the EOT tokens to get the 'true' token length
                    token_ids = [token for token in token_ids if token != decoder_eot_token_id]
                    token_ids = token_ids + [decoder_eot_token_id]
                    # check whether we have timestamps in the PLs and filter if required
                    has_timestamps = len(set(token_ids) & set(timestamp_ids)) > 0
                    if has_timestamps:
                        # sample from binomial distribution to get probability of training on timestamps
                        predict_timestamps = bool(np.random.binomial(1, arg.timestamp_probability))
                        if not predict_timestamps:
                            # filter timestamps and insert the <|notimestamps|> task token
                            token_ids = [token for token in token_ids if token < timestamp_begin]
                            token_ids.insert(timestamp_position, timestamp_begin)
                else:
                    # pseudo-labelled transcriptions have been decoded to text (`decode_token_ids=True`)
                    has_timestamps = has_timestamp_tokens(input_str)

                    if has_timestamps:
                        predict_timestamps = bool(np.random.binomial(1, arg.timestamp_probability))
                        if not predict_timestamps:
                            # filter timestamp token ids if not part of the prediction task
                            input_str = tokenizer._filter_timestamp_ids(input_str)
                    else:
                        predict_timestamps = False

                    tokenizer.set_prefix_tokens(
                        language=arg.language, task="transcribe", predict_timestamps=predict_timestamps
                    )
                    token_ids = tokenizer(input_str).input_ids

                all_token_ids_unprompted.append(token_ids)
                # check whether to condition on previous text - we do this with probability condition_on_prev_probability
                condition_on_prev = bool(np.random.binomial(1, arg.condition_on_prev_probability))
                if condition_on_prev and len(all_token_ids_unprompted) > 1:
                    # prompt ids are the penultimate token ids in the batch
                    prompt_ids = all_token_ids_unprompted[-2]
                    # strip timestamp tokens from prompt
                    prompt_ids = [token for token in prompt_ids if token < timestamp_begin]
                    if len(prompt_ids) > 0:
                        # remove the standard task tokens and add the special <|startofprev|> token
                        prompt_ids = [decoder_prev_token_id] + prompt_ids[timestamp_position:-1]
                    if len(prompt_ids + token_ids) < arg.max_label_length:
                        token_ids = prompt_ids + token_ids
                all_token_ids.append(token_ids)

            batch["labels"] = all_token_ids
            return batch

        raw_datasets_labeled = DatasetDict()
        map_fn_train = partial(
            raw_datasets["train"].map,
            function=prepare_train_dataset,
            batched=True,
            batch_size=arg.preprocessing_batch_size,
        )
        raw_datasets_labeled["train"] = map_fn_train(
            num_proc=arg.preprocessing_num_workers, desc="preprocess train dataset"
        )

        #############################
        # Filtering by audio length #
        #############################
        # Filter training data with inputs longer than `max_input_length`
        max_input_length = int(arg.max_duration_in_seconds * feature_extractor.sampling_rate)
        min_input_length = int(arg.min_duration_in_seconds * feature_extractor.sampling_rate)

        def is_audio_in_length_range(length):
            return min_input_length < length < max_input_length

        def is_labels_in_length_range(labels):
            return 0 < len(labels) <= arg.max_label_length

        filter_by_audio_fn = partial(
            raw_datasets_labeled.filter, function=is_audio_in_length_range, input_columns=["input_length"]
        )
        raw_datasets_labeled_filtered = filter_by_audio_fn(
            num_proc=arg.preprocessing_num_workers, desc="filtering train dataset by audio length"
        )

        # Filter training data with labels longer than `max_label_length`
        filter_by_labels_fn = partial(
            raw_datasets_labeled_filtered.filter, function=is_labels_in_length_range, input_columns=["labels"]
        )
        raw_datasets_labeled_filtered = filter_by_labels_fn(
            num_proc=arg.preprocessing_num_workers, desc="filtering train dataset"
        )
        safe_push(raw_datasets_labeled_filtered, repo_name, arg.dataset_config_name)
    else:
        raw_datasets_labeled_filtered = dataset

    ######################
    # Log-mel Conversion #
    ######################
    if not arg.skip_logmel:

        def log_mel_transformation(batch):
            """Pre-process the raw dataset: Convert the audio arrays to log-mel spectrogram inputs"""
            audio = [sample["array"] for sample in batch["audio"]]
            inputs = feature_extractor(audio, sampling_rate=feature_extractor.sampling_rate)
            batch["input_features"] = inputs.input_features
            return batch

        map_fn_train = partial(
            raw_datasets_labeled_filtered["train"].map,
            keep_in_memory=True,
            function=log_mel_transformation,
            remove_columns=["audio", "text", "whisper_transcript"],
            batched=True,
            batch_size=arg.preprocessing_batch_size,
        )
        logmel_feature_dataset = DatasetDict({
            "train": map_fn_train(
                num_proc=arg.preprocessing_num_workers,
                desc="obtain log-mel feature from audio"
            )
        })
        safe_push(logmel_feature_dataset, f"{repo_name}.vectorize", arg.dataset_config_name)


if __name__ == "__main__":
    main()
