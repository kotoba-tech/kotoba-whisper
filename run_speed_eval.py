import argparse
import json
import os
from statistics import mean, stdev
from time import time

import torch
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
arg = parser.parse_args()

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
for _ in range(arg.n_trial):
    start = time()
    transcription = pipe(audio.copy(), generate_kwargs=generate_kwargs)
    elapsed.append(time() - start)
metric.update({"time (mean)": mean(elapsed), "time (std)": stdev(elapsed), "time (all)": elapsed})
runtime = [metric]
if os.path.exists(arg.output):
    with open(arg.output) as f:
        runtime += [json.loads(s) for s in f.read().split("\n") if len(s) > 0]
else:
    os.makedirs(os.path.dirname(arg.output), exist_ok=True)
with open(arg.output, "w") as f:
    f.write("\n".join([json.dumps(i) for i in runtime]))
