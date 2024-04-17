import os
from math import prod
import pandas as pd
from transformers import WhisperForConditionalGeneration

# https://stackoverflow.com/questions/71692354/facing-ssl-error-with-huggingface-pretrained-models
os.environ['CURL_CA_BUNDLE'] = ''

MODELS = [
    "openai/whisper-tiny",
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large-v3",
    "japanese-asr/distil-whisper-large-v3-ja-reazonspeech-tiny",
]


def pretty(num): return "{:,}".format(num)


def show_parameter(target_model_name):
    target_model = WhisperForConditionalGeneration.from_pretrained(target_model_name)
    param_size_embedding = prod(target_model.get_output_embeddings().weight.shape)
    param_size_full = sum(p.numel() for p in target_model.parameters())
    vocab_size = len(target_model.get_output_embeddings().weight)

    print(f"PARAMETER SUMMARY: {target_model_name}")
    print(f"\t*parameter size (full) : {pretty(param_size_full)}")
    print(f"\t*parameter size (vocab): {pretty(param_size_embedding)}")
    print(f"\t*parameter size (rest) : {pretty(param_size_full - param_size_embedding)}")
    print(f"\t*ratio of vocab param  : {round(param_size_embedding / param_size_full * 100, 1)}%")
    print(f"\t*vocab size            : {pretty(vocab_size)}")
    return {
        "model": target_model_name,
        "vocab_size": vocab_size,
        "param_size_full": param_size_full,
        "param_size_embedding": param_size_embedding
    }


if __name__ == '__main__':
    parameter_report = []
    for m in MODELS:
        parameter_report.append(show_parameter(m))
    pd.DataFrame(parameter_report).to_csv("stats/model_statistics.csv")
