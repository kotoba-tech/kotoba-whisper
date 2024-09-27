import argparse
import json
import os
from statistics import mean, stdev
from time import time

import torch
from transformers import pipeline
from datasets import load_dataset


output_file = "misc/runtime_pipeline.jsonl"
n_trial = 5

parser = argparse.ArgumentParser(description='Runtime check.')
parser.add_argument('-m', '--model', default="kotoba-tech/kotoba-whisper-v1.0", type=str)
parser.add_argument('-a', '--attn', default=None, type=str)
parser.add_argument('-t', '--task', default="transcribe", type=str)
parser.add_argument('-l', '--language', default="ja", type=str)
parser.add_argument('-c', '--chunk-length', default=15, type=int)
parser.add_argument('-b', '--batch', default=16, type=int)
arg = parser.parse_args()

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
if arg.model in ["japanese-asr/ja-cascaded-s2t-translation", "japanese-asr/en-cascaded-s2t-translation"]:
    language_code = {"ja": "jpn_Jpan", "en": "eng_Latn"}
    pipe = pipeline(
        model=arg.model,
        torch_dtype=torch_dtype,
        device=device,
        model_translation="facebook/nllb-200-3.3B",
        tgt_lang=language_code[arg.language],
        model_kwargs=model_kwargs,
        chunk_length_s=arg.chunk_length,
        batch_size=arg.batch,
        trust_remote_code=True,
    )
    generate_kwargs = {}
else:
    pipe = pipeline("automatic-speech-recognition", **pipeline_config)
    stable_ts, punctuator = None, None
metric = {
    "model": arg.model, "chunk_length_s": arg.chunk_length, "stable_ts": stable_ts, "punctuator": punctuator,
    "attention": arg.attn, "device": device
}

runtime = []
dataset = load_dataset("kotoba-tech/kotoba-whisper-eval", split="train")
x = dataset['audio'][0]  # long interview audio
elapsed = []
for _ in range(n_trial):
    start = time()
    transcription = pipe(x.copy(), generate_kwargs=generate_kwargs, return_timestamps=arg.return_timestamps)
    elapsed.append(time() - start)
runtime.append(
    {
        "model": arg.model, "chunk_length_s": arg.chunk_length, "stable_ts": stable_ts, "punctuator": punctuator,
        "attention": arg.attn, "device": device, "file": x['path'], "return_timestamps": arg.return_timestamps,
        "batch": arg.batch, "time (mean)": mean(elapsed), "time (std)": stdev(elapsed),
        "transcription": transcription['text'], "time (all)": elapsed
    }
)

if os.path.exists(output_file):
    with open(output_file) as f:
        runtime += [json.loads(s) for s in f.read().split("\n") if len(s) > 0]

with open(output_file, "w") as f:
    f.write("\n".join([json.dumps(i) for i in runtime]))