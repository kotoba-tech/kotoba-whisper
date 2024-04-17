import json
import os

from datasets import load_dataset
from tqdm import tqdm
from transformers import WhisperTokenizer

# https://stackoverflow.com/questions/71692354/facing-ssl-error-with-huggingface-pretrained-models
os.environ['CURL_CA_BUNDLE'] = ''

TOKENIZER = WhisperTokenizer.from_pretrained("openai/whisper-large-v3", language="japanese")
NUM_PROC = 128


def get_cumulative_max_min(
        cumulative_value: float,
        current_value: float,
        current_max: float,
        current_min: float,
):
    if current_max is None or current_value > current_max:
        current_max = current_value
    if current_min is None or current_value < current_min:
        current_min = current_value
    return cumulative_value + current_value, current_max, current_min


def dataset_statistics(data: str = "reazonspeech", data_type: str = "tiny", num_proc: int = 1):
    text_column = "transcription"
    if data == "reazonspeech":
        dataset = load_dataset(
            f"japanese-asr/whisper_transcriptions.reazonspeech.{data_type}.wer_10.0",
            data_type,
            split="train",
            trust_remote_code=True,
            num_proc=num_proc,
        )
        text_column = "text"
    elif data == "ja_asr.reazonspeech_test":
        dataset = load_dataset(
            "japanese-asr/ja_asr.reazonspeech_test",
            split="test",
            trust_remote_code=True
        )
    elif data == "ja_asr.jsut-basic5000":
        dataset = load_dataset(
            "japanese-asr/ja_asr.jsut-basic5000",
            split="test",
            trust_remote_code=True
        )
    elif data == "common_voice_8_0":
        dataset = load_dataset(
            "japanese-asr/ja_asr.common_voice_8_0",
            split="test",
            trust_remote_code=True
        )
    else:
        raise ValueError(f"unknown dataset {data}")
    iterator = iter(dataset)
    duration, duration_max, duration_min = 0, None, None
    amp_max, amp_max_max, amp_max_min = 0, None, None
    amp_min, amp_min_max, amp_min_min = 0, None, None
    amp_mean, amp_mean_max, amp_mean_min = 0, None, None
    text, text_max, text_min = 0, None, None
    token, token_max, token_min = 0, None, None
    data_size = 0
    for value in tqdm(iterator):
        ar = value['audio']['array']
        transcription_char_size = len(value[text_column])
        transcription_token_size = len(TOKENIZER(value[text_column])["input_ids"])
        duration, duration_max, duration_min = get_cumulative_max_min(
            duration,
            len(ar) / value['audio']['sampling_rate'],
            duration_max,
            duration_min
        )
        amp_max, amp_max_max, amp_max_min = get_cumulative_max_min(amp_max, ar.max(), amp_max_max, amp_max_min)
        amp_min, amp_min_max, amp_min_min = get_cumulative_max_min(amp_min, ar.min(), amp_min_max, amp_min_min)
        amp_mean, amp_mean_max, amp_mean_min = get_cumulative_max_min(amp_mean, ar.mean(), amp_mean_max, amp_mean_min)
        text, text_max, text_min = get_cumulative_max_min(text, transcription_char_size, text_max, text_min)
        token, token_max, token_min = get_cumulative_max_min(token, transcription_token_size, token_max, token_min)
        data_size += 1
    return {
        "audio": {
            "duration": {"mean": duration / data_size, "max": duration_max, "min": duration_min},
            "amplitude_max": {"mean": amp_max / data_size, "max": amp_max_max, "min": amp_max_min},
            "amplitude_min": {"mean": amp_min / data_size, "max": amp_min_max, "min": amp_min_min},
            "amplitude_mean": {"mean": amp_mean / data_size, "max": amp_mean_max, "min": amp_mean_min},
        },
        "transcription": {
            "character_length": {"mean": text / data_size, "max": text_max, "min": text_min},
            "token_length": {"mean": token / data_size, "max": token_max, "min": token_min},
        },
        "total_data_size": data_size,
        "total_audio_duration (sec)": duration,
        "total_transcription_size (character)": text
    }


if __name__ == '__main__':
    stats = {}
    if os.path.exists("stats/data_statistics.json"):
        with open("stats/data_statistics.json") as f:
            stats = json.load(f)

    if "ja_asr.reazonspeech_test" not in stats:
        stats["ja_asr.reazonspeech_test"] = dataset_statistics("ja_asr.reazonspeech_test")
    if "ja_asr.jsut_basic5000" not in stats:
        stats["ja_asr.jsut_basic5000"] = dataset_statistics("ja_asr.jsut_basic5000")
    if "common_voice_8_0" not in stats:
        stats["common_voice_8_0"] = dataset_statistics("common_voice_8_0")
    if "reazonspeech.tiny" not in stats:
        stats["reazonspeech.tiny"] = dataset_statistics(data_type="tiny")
    if "reazonspeech.small" not in stats:
        stats["reazonspeech.small"] = dataset_statistics(data_type="small")
    if "reazonspeech.medium" not in stats:
        stats["reazonspeech.medium"] = dataset_statistics(data_type="medium")
    if "reazonspeech.large" not in stats:
        stats["reazonspeech.large"] = dataset_statistics(data_type="large", num_proc=128)
    # if "reazonspeech.all" not in stats:
    #     stats["reazonspeech.all"] = dataset_statistics(data_type="all")

    with open("stats/data_statistics.json", "w") as f:
        json.dump(stats, f)
