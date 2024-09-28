import argparse
import json
import os
from statistics import mean, stdev
from time import time
from pprint import pprint

import torch
import pandas as pd
import numpy as np
from transformers import pipeline


def generate_dummy_audio(duration: int = 5, sampling_rate: int = 16000):
    array = np.random.rand(duration * sampling_rate)
    array = (array - 0.5) * 2 * 0.007
    return {"array": array, "sampling_rate": sampling_rate, "path": "tmp"}


parser = argparse.ArgumentParser(description='Runtime check.')
parser.add_argument('-m', '--model', default="kotoba-tech/kotoba-whisper-v1.0", type=str)
parser.add_argument('-a', '--attn', default=None, type=str)
parser.add_argument('-t', '--task', default="transcribe", type=str)
parser.add_argument('-o', '--output', default="eval_pipeline/runtime_pipeline.jsonl", type=str)
parser.add_argument('-l', '--language', default="en", type=str)
parser.add_argument('-n', '--n-trial', default=15, type=int)
parser.add_argument('-d', '--duration', default=10, type=int)
parser.add_argument('-s', '--sampling-rate', default=16000, type=int)
parser.add_argument('--translation-model', default="facebook/nllb-200-3.3B", type=str)
parser.add_argument('--pretty-table', action="store_true")
arg = parser.parse_args()

if arg.pretty_table:

    def pretty(m, t):
        if str(t) != "nan":
            return f"[{m}](https://huggingface.co/{m}) ([{t}](https://huggingface.co/{t}))"
        return f"[{m}](https://huggingface.co/{m})"


    with open(arg.output) as f:
        runtime = [json.loads(s) for s in f.read().split("\n") if len(s) > 0]
    df = pd.DataFrame(runtime)
    df["model"] = [pretty(m, t) for m, t in zip(df["model"], df["translation_model"])]
    print(df.pivot_table(columns="duration", index="model", values="time (mean)").round(2).to_markdown())
    exit()
# model config
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_kwargs = {"attn_implementation": arg.attn} if torch.cuda.is_available() and arg.attn else {}
pipeline_config = dict(
    model=arg.model, torch_dtype=torch_dtype, device=device, model_kwargs=model_kwargs, chunk_length_s=15,
    trust_remote_code=True,
)
# instantiate pipeline
metric = {"model": arg.model, "attention": arg.attn, "device": device, "duration": arg.duration}
if arg.model in ["japanese-asr/ja-cascaded-s2t-translation", "japanese-asr/en-cascaded-s2t-translation"]:
    language_code = {"ja": "jpn_Jpan", "en": "eng_Latn"}
    pipe = pipeline(model_translation=arg.translation_model, tgt_lang=language_code[arg.language], **pipeline_config)
    generate_kwargs = {}
    metric["translation_model"] = arg.translation_model
else:
    pipe = pipeline("automatic-speech-recognition", **pipeline_config)
    generate_kwargs = {"language": arg.language, "task": arg.task}
# generate dummy audio
audio = generate_dummy_audio(arg.duration, arg.sampling_rate)
# run test
elapsed = []
for _ in range(arg.n_trial + 1):
    start = time()
    transcription = pipe(audio.copy(), generate_kwargs=generate_kwargs)
    elapsed.append(time() - start)
elapsed = elapsed[1:]
metric.update({"time (mean)": mean(elapsed), "time (std)": stdev(elapsed), "time (all)": elapsed})
pprint(metric)
runtime = [metric]
if os.path.exists(arg.output):
    with open(arg.output) as f:
        runtime += [json.loads(s) for s in f.read().split("\n") if len(s) > 0]
else:
    os.makedirs(os.path.dirname(arg.output), exist_ok=True)
with open(arg.output, "w") as f:
    f.write("\n".join([json.dumps(i) for i in runtime]))
