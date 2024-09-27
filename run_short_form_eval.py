"""Compute CER/WER for Japanese ASR models.
- Requirement for Nemo model
git clone https://github.com/reazon-research/ReazonSpeech
pip install --upgrade pip setuptools wheel
pip install --upgrade cython
pip install cython_bbox
pip install ReazonSpeech/pkg/nemo-asr
pip install numpy==1.26.4
sudo apt install ffmpeg
"""
import json
import os
import argparse
from pprint import pprint

import torch
import pandas as pd
from transformers import pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
from transformers import WhisperTokenizer
from datasets import load_dataset
from evaluate import load

pretty_dataset_names = {
    "japanese-asr/ja_asr.jsut_basic5000": "JSUT Basic 5000",
    "japanese-asr/ja_asr.common_voice_8_0": "CommonVoice 8 (Japanese test set)",
    "japanese-asr/ja_asr.reazonspeech_test": "ReazonSpeech (held out test set)",
    "japanese-asr/en_asr.esb_eval": "ESB",
    "japanese-asr/ja2en.s2t_translation": "Translation (Ja->En)",
    "japanese-asr/en2ja.s2t_translation": "Translation (En->Ja)"
}

parser = argparse.ArgumentParser(description='Compute CER/WER for Japanese ASR model.')
parser.add_argument('-m', '--model', default="kotoba-tech/kotoba-whisper-v1.1", type=str)
parser.add_argument('-d', '--dataset', default="japanese-asr/ja_asr.jsut_basic5000", type=str)
parser.add_argument('--dataset-split', default="test", type=str)
parser.add_argument('--dataset-config', default=None, type=str)
parser.add_argument('-a', '--attn', default="sdpa", type=str)
parser.add_argument('-l', '--language', default="ja", type=str)
parser.add_argument('-t', '--task', default="transcribe", type=str)
parser.add_argument('--column-audio', default="audio", type=str)
parser.add_argument('--column-text', default="transcription", type=str)
parser.add_argument('-b', '--batch', default=16, type=int)
parser.add_argument('-c', '--chunk-length', default=15, type=int)
parser.add_argument('-o', '--output-dir', default="eval_pipeline", type=str)
parser.add_argument('-p', '--punctuator', action="store_true")
parser.add_argument('-s', '--stable-ts', action="store_true")
parser.add_argument('--translation-model', default="facebook/nllb-200-3.3B", type=str)
parser.add_argument('--pretty-table', action="store_true")
arg = parser.parse_args()

os.makedirs(arg.output_dir, exist_ok=True)
output_metric_file = f"{arg.output_dir}/metric.{arg.language}.{arg.task}.jsonl"

# display mode
if arg.pretty_table:

    def pretty(m, p, s):
        if p and s:
            return f"[{m}](https://huggingface.co/{m}) (punctuator + stable-ts)"
        if s:
            return f"[{m}](https://huggingface.co/{m}) (stable-ts)"
        if p:
            return f"[{m}](https://huggingface.co/{m}) (punctuator)"
        return f"[{m}](https://huggingface.co/{m})"


    with open(output_metric_file) as f:
        metrics = [json.loads(s) for s in f.read().split("\n") if len(s) > 0]
    df_metric = pd.DataFrame(metrics).sort_values(["dataset", "model", "chunk_length_s", "punctuator", "stable_ts"])
    df_metric = df_metric[df_metric.language == arg.language]
    df_metric = df_metric[df_metric.task == arg.task]
    df_metric = df_metric.drop_duplicates(["dataset", "dataset_config", "dataset_split", "model", "chunk_length_s", "punctuator", "stable_ts"])
    metrics = [i.to_dict() for _, i in df_metric.iterrows()]
    with open(output_metric_file, "w") as f:
        f.write("\n".join([json.dumps(i) for i in metrics]))

    df_metric = df_metric.round(1)
    df_metric["dataset"] = [f"[{pretty_dataset_names[m] if m in pretty_dataset_names else m}](https://huggingface.co/datasets/{m})" + (f" ({c})" if c is not None else "") for m, c in zip(df_metric["dataset"], df_metric["dataset_config"])]
    df_metric["model"] = [pretty(m, p, s) for m, p, s in zip(df_metric["model"], df_metric["punctuator"], df_metric["stable_ts"])]

    print("\nNORM (WER)")
    print(df_metric.pivot_table(values="wer_norm", columns="dataset", index="model", aggfunc='first').to_markdown(), "\n")
    print("\nRAW (WER)")
    print(df_metric.pivot_table(values="wer_raw", columns="dataset", index="model", aggfunc='first').to_markdown(), "\n")
    if arg.language == "ja":
        print("\nNORM (CER)")
        print(df_metric.pivot_table(values="cer_norm", columns="dataset", index="model", aggfunc='first').to_markdown(), "\n")
        print("\nRAW (CER)")
        print(df_metric.pivot_table(values="cer_raw", columns="dataset", index="model", aggfunc='first').to_markdown(), "\n")

    exit()

# model config
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_kwargs = {"attn_implementation": arg.attn} if torch.cuda.is_available() and arg.attn else {}
generate_kwargs = {"language": arg.language, "task": arg.task}
pipeline_config = dict(
    model=arg.model,
    torch_dtype=torch_dtype,
    device=device,
    model_kwargs=model_kwargs,
    chunk_length_s=arg.chunk_length,
    batch_size=arg.batch
)

# instantiate pipeline
metric = {
    "model": arg.model,
    "dataset": arg.dataset,
    "dataset_config": arg.dataset_config,
    "dataset_split": arg.dataset_split,
    "chunk_length_s": arg.chunk_length,
    "language": arg.language,
    "task": arg.task
}
stable_ts, punctuator = None, None
if arg.model in ["japanese-asr/ja-cascaded-s2t-translation", "japanese-asr/en-cascaded-s2t-translation"]:
    prediction_path = (f"{arg.output_dir}/model-{os.path.basename(arg.model)}.dataset-{os.path.basename(arg.dataset)}."
                       f"dataset_config-{arg.dataset_config}.dataset_split-{arg.dataset_split}."
                       f"language-{arg.language}.task-{arg.task}.stable-ts-{stable_ts}."
                       f"punctuator-{punctuator}.chunk_length-{arg.chunk_length}."
                       f"translation.{os.path.basename(arg.translation_model)}.csv")
    metric["translation_model"] = arg.translation_model
else:
    prediction_path = (f"{arg.output_dir}/model-{os.path.basename(arg.model)}.dataset-{os.path.basename(arg.dataset)}."
                       f"dataset_config-{arg.dataset_config}.dataset_split-{arg.dataset_split}."
                       f"language-{arg.language}.task-{arg.task}.stable-ts-{stable_ts}."
                       f"punctuator-{punctuator}.chunk_length-{arg.chunk_length}.csv")

if os.path.exists(prediction_path):
    df = pd.read_csv(prediction_path)
    prediction_norm = df["prediction_norm"].values.tolist()
    reference_norm = df["reference_norm"].values.tolist()
    prediction_raw = df["prediction_raw"].values.tolist()
    reference_raw = df["reference_raw"].values.tolist()
    audio_id = df["id"].values.tolist()
    if arg.model in ["kotoba-tech/kotoba-whisper-v1.1", "kotoba-tech/kotoba-whisper-v2.1"]:
        stable_ts, punctuator = arg.stable_ts, arg.punctuator
else:
    if arg.model in ["kotoba-tech/kotoba-whisper-v1.1", "kotoba-tech/kotoba-whisper-v2.1"]:
        pipe = pipeline(trust_remote_code=True, punctuator=arg.punctuator, stable_ts=arg.stable_ts, **pipeline_config)
        stable_ts, punctuator = arg.stable_ts, arg.punctuator
    elif arg.model in ["japanese-asr/ja-cascaded-s2t-translation", "japanese-asr/en-cascaded-s2t-translation"]:
        assert arg.task == "translate"
        language_code = {"ja": "jpn_Jpan", "en": "eng_Latn"}
        pipe = pipeline(
            model=arg.model,
            torch_dtype=torch_dtype,
            device=device,
            model_translation=arg.translation_model,
            tgt_lang=language_code[arg.language],
            model_kwargs=model_kwargs,
            chunk_length_s=arg.chunk_length,
            batch_size=arg.batch,
            trust_remote_code=True,
        )
        generate_kwargs = {}
    elif arg.model in ["reazon-research/reazonspeech-nemo-v2"]:
        assert arg.task == "transcribe" and arg.language == "ja"

        from reazonspeech.nemo.asr import load_model, transcribe, interface
        model = load_model()

        def pipe(audio_input, generate_kwargs):
            texts = []
            for i in audio_input:
                texts += [transcribe(model, interface.AudioData(waveform=i["array"], samplerate=i["sampling_rate"])).text]
            return [{"text": i} for i in texts]

    else:
        pipe = pipeline("automatic-speech-recognition", **pipeline_config)

    # load the dataset and get prediction
    if arg.dataset_config:
        dataset = load_dataset(arg.dataset, arg.dataset_config, split=arg.dataset_split, trust_remote_code=True)
    else:
        dataset = load_dataset(arg.dataset, split=arg.dataset_split, trust_remote_code=True)
    output = pipe(dataset[arg.column_audio], generate_kwargs=generate_kwargs)
    prediction_raw = [i["text"] for i in output]
    reference_raw = dataset[arg.column_text]
    audio_id = [i["path"] for i in dataset[arg.column_audio]]

    if arg.language == "en":
        tokenizer = WhisperTokenizer.from_pretrained(arg.model)
        normalizer = EnglishTextNormalizer(tokenizer.english_spelling_normalizer)
        normalize = lambda x: normalizer(x)
    elif arg.language == "ja":
        normalizer = BasicTextNormalizer()
        normalize = lambda x: normalizer(x).replace(" ", "").replace("。.", "。")
    else:
        normalizer = BasicTextNormalizer()
        normalize = lambda x: normalizer(x)

    prediction_norm = [normalize(i) for i in prediction_raw]
    reference_norm = [normalize(i) for i in reference_raw]
    # remove empty reference
    flag = [len(i) != 0 for i in reference_norm]
    reference_norm = list(list(zip(*filter(lambda x: x[1], zip(reference_norm, flag))))[0])
    reference_raw = list(list(zip(*filter(lambda x: x[1], zip(reference_raw, flag))))[0])
    prediction_norm = list(list(zip(*filter(lambda x: x[1], zip(prediction_norm, flag))))[0])
    prediction_raw = list(list(zip(*filter(lambda x: x[1], zip(prediction_raw, flag))))[0])
    audio_id = list(list(zip(*filter(lambda x: x[1], zip(audio_id, flag))))[0])

# compute metrics
metric.update({"punctuator": punctuator, "stable_ts": stable_ts})
cer_metric = load("cer")
cer_norm = 100 * cer_metric.compute(predictions=prediction_norm, references=reference_norm)
cer_raw = 100 * cer_metric.compute(predictions=prediction_raw, references=reference_raw)
wer_metric = load("wer")
wer_norm = 100 * wer_metric.compute(predictions=prediction_norm, references=reference_norm)
wer_raw = 100 * wer_metric.compute(predictions=prediction_raw, references=reference_raw)
metric.update({"cer_raw": cer_raw, "wer_raw": wer_raw, "cer_norm": cer_norm, "wer_norm": wer_norm})

# save the results
metrics = []
if os.path.exists(output_metric_file):
    with open(output_metric_file) as f:
        metrics += [json.loads(s) for s in f.read().split("\n") if len(s) > 0]
metrics.append(metric)
pprint(metrics)
with open(output_metric_file, "w") as f:
    f.write("\n".join([json.dumps(s) for s in metrics]))

# save prediction
df = pd.DataFrame(
    [audio_id, reference_norm, prediction_norm, reference_raw, prediction_raw],
    index=["id", "reference_norm", "prediction_norm", "reference_raw", "prediction_raw"]
).T
df.to_csv(prediction_path, index=False)
