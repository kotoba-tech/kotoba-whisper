"""
python misc/check_runtime_pipeline.py -m "kotoba-tech/kotoba-whisper-v1.0"
python misc/check_runtime_pipeline.py -m "openai/whisper-tiny"
python misc/check_runtime_pipeline.py -m "openai/whisper-small"
python misc/check_runtime_pipeline.py -m "openai/whisper-medium"
python misc/check_runtime_pipeline.py -m "openai/whisper-large-v3"
"""
import argparse
import json
import os
from statistics import mean, stdev
from time import time

import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


parser = argparse.ArgumentParser(description='Check inference speed by processing long audio. ')
parser.add_argument('-m', '--model', default=None, type=str)
parser.add_argument('-o', '--output-file', default="misc/runtime_pipeline.json", type=str)
arg = parser.parse_args()

if arg.model is not None:
    # config
    model_id = arg.model
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # load model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs={'attn_implementation': "sdpa"}
    )

    # load sample audio (concatenate instances to create a long audio)
    dataset = load_dataset("japanese-asr/ja_asr.reazonspeech_test", split="test")
    sampling_rate = dataset[0]['audio']['sampling_rate']
    assert sampling_rate == pipe.feature_extractor.sampling_rate
    sample_short = {
        "array": dataset[0]['audio']['array'],
        "sampling_rate": sampling_rate
    }

    long_array = np.concatenate([i["array"] for i in dataset["audio"]])[:3600 * sampling_rate]  # create one hour audio
    sample_long = {
        "array": long_array,
        "sampling_rate": sampling_rate
    }

    # run inference
    pipe(sample_short)
    start = time()
    result = pipe(sample_long)
    elapsed = time() - start
    print(result["text"], elapsed)

    # log
    report = {}
    if os.path.exists(arg.output_file):
        with open(arg.output_file) as f:
            report = json.load(f)
    if arg.model in report:
        report[arg.model] += [elapsed]
    else:
        report[arg.model] = [elapsed]

    report[f"avg.{arg.model}"] = mean(report[arg.model])
    if len(report[arg.model]) > 2:
        report[f"std.{arg.model}"] = stdev(report[arg.model])
    print(json.dumps(report, indent=4))

    with open(arg.output_file, "w") as f:
        json.dump(report, f)
if os.path.exists(arg.output_file):
    with open(arg.output_file) as f:
        report = json.load(f)
    df = pd.DataFrame([(k.replace("avg.", ""), v) for k, v in report.items() if k.startswith("avg")])
    df = df.round(2)
    df.columns = ["Model", "Inference Time (sec)"]
    print(df.to_markdown(index=False))
