"""Compute CER/WER for Japanese ASR models."""
import json
import os
import argparse
from pprint import pprint

import torch
import pandas as pd
from transformers import pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from datasets import load_dataset
from evaluate import load

parser = argparse.ArgumentParser(description='Compute CER/WER for Japanese ASR model.')
parser.add_argument('-m', '--model', default="kotoba-tech/kotoba-whisper-v1.1", type=str)
parser.add_argument('-d', '--dataset', default="japanese-asr/ja_asr.jsut_basic5000", type=str)
parser.add_argument('-a', '--attn', default="sdpa", type=str)
parser.add_argument('-b', '--batch', default=16, type=int)
parser.add_argument('-c', '--chunk-length', default=15, type=int)
parser.add_argument('-o', '--output-dir', default="eval_pipeline", type=str)
parser.add_argument('-p', '--punctuator', action="store_true")
parser.add_argument('-s', '--stable-ts', action="store_true")
parser.add_argument('--pretty-table', action="store_true")
arg = parser.parse_args()

os.makedirs(arg.output_dir, exist_ok=True)
output_metric_file = f"{arg.output_dir}/metric.jsonl"

# display mode
if arg.pretty_table:
    with open(output_metric_file) as f:
        metrics = [json.loads(s) for s in f.read().split("\n") if len(s) > 0]
    df_metric = pd.DataFrame(metrics).round(1).sort_values(["dataset", "model"])
    df_metric["cer/wer (norm)"] = [f"{c}/{w}" for c, w in zip(df_metric["cer_norm"], df_metric["wer_norm"])]
    df_metric["cer/wer (raw)"] = [f"{c}/{w}" for c, w in zip(df_metric["cer_raw"], df_metric["wer_raw"])]

    def pretty(m, p, s):
        if p and s:
            return f"{m} (punctuator + stable-ts)"
        if s:
            return f"{m} (stable-ts)"
        if p:
            return f"{m} (punctuator)"
        return m

    df_metric["model"] = [pretty(m, p, s) for m, p, s in zip(df_metric["model"], df_metric["punctuator"], df_metric["stable_ts"])]
    df_metric = df_metric[["model", "dataset", "punctuator", "stable_ts", "cer/wer (raw)", "cer/wer (norm)"]]
    print(df_metric)
    df_metric = df_metric.drop_duplicates()
    print("\nNORM")
    print(df_metric.pivot(values="cer/wer (norm)", columns="dataset", index="model").to_markdown())
    print("\nRAW")
    print(df_metric.pivot(values="cer/wer (raw)", columns="dataset", index="model").to_markdown())
    exit()

# model config
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_kwargs = {"attn_implementation": arg.attn} if torch.cuda.is_available() and arg.attn else {}
generate_kwargs = {"language": "japanese", "task": "transcribe"}
pipeline_config = dict(
    model=arg.model,
    torch_dtype=torch_dtype,
    device=device,
    model_kwargs=model_kwargs,
    chunk_length_s=arg.chunk_length,
    batch_size=arg.batch
)

# instantiate pipeline
metric = {"model": arg.model, "dataset": arg.dataset, "chunk_length_s": arg.chunk_length}
if arg.model in ["kotoba-tech/kotoba-whisper-v1.1"]:
    pipe = pipeline(trust_remote_code=True, punctuator=arg.punctuator, stable_ts=arg.stable_ts, **pipeline_config)
    stable_ts, punctuator = arg.stable_ts, arg.punctuator
else:
    pipe = pipeline("automatic-speech-recognition", **pipeline_config)
    stable_ts, punctuator = None, None
metric.update({"punctuator": punctuator, "stable_ts": stable_ts})

# load the dataset and get prediction
dataset = load_dataset(arg.dataset, split="test")
output = pipe(dataset['audio'], generate_kwargs=generate_kwargs)
normalizer = BasicTextNormalizer()
prediction_norm = [normalizer(i['text']).replace(" ", "") for i in output]
references_norm = [normalizer(i).replace(" ", "") for i in dataset['transcription']]
prediction_raw = [i['text'].replace(" ", "") for i in output]
references_raw = [i.replace(" ", "") for i in dataset['transcription']]

# compute metrics
cer_metric = load("cer")
cer_norm = 100 * cer_metric.compute(predictions=prediction_norm, references=references_norm)
cer_raw = 100 * cer_metric.compute(predictions=prediction_raw, references=references_raw)
wer_metric = load("wer")
wer_norm = 100 * wer_metric.compute(predictions=prediction_norm, references=references_norm)
wer_raw = 100 * wer_metric.compute(predictions=prediction_raw, references=references_raw)
metric.update({"cer_raw": cer_raw, "wer_raw": wer_raw, "cer_norm": cer_norm, "wer_norm": wer_norm})

# save the results
metrics = []
if os.path.exists(output_metric_file):
    with open(output_metric_file) as f:
        metrics += [json.loads(s) for s in f.read().split("\n") if len(s) > 0]
output_prediction_file = f"{arg.output_dir}/prediction.csv"
dfs = None
if os.path.exists(output_prediction_file):
    dfs = pd.read_csv(output_prediction_file, index_col=0)
metrics.append(metric)
pprint(metrics)
with open(output_metric_file, "w") as f:
    f.write("\n".join([json.dumps(s) for s in metrics]))

# save prediction
audio_id = [i["path"] for i in dataset['audio']]
df = pd.DataFrame(
    [audio_id, references_norm, prediction_norm, references_raw, prediction_raw],
    index=["id", "reference_norm", "prediction_norm", "reference_raw", "prediction_raw"]
).T
df["model"] = arg.model
df["dataset"] = arg.dataset
df["stable_ts"] = stable_ts
df["punctuator"] = punctuator
df["chunk_length_s"] = arg.chunk_length
dfs = df if dfs is None else pd.concat([dfs, df])
dfs.to_csv(output_prediction_file, index=False)

