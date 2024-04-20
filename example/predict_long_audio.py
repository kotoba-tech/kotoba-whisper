import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset, Audio

# config
model_id = "kotoba-tech/kotoba-whisper-v1.0"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# load model
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

# load sample audio & downsample to 16kHz
dataset = load_dataset("japanese-asr/ja_asr.reazonspeech_test", split="test")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
input_features = processor(dataset[10]["audio"]["array"], return_tensors="pt").input_features

# --- Without prompt ---
output_without_prompt = model.generate(input_features)
print(processor.decode(output_without_prompt[0]))
# <|startoftranscript|><|ko|><|transcribe|><|notimestamps|>81歳、力強い走りに変わってきます。<|endoftext|>

# --- With prompt ---: Let's change `81` to `91`.
prompt_ids = processor.get_prompt_ids("91歳", return_tensors="pt")
output_with_prompt = model.generate(input_features, prompt_ids=prompt_ids)
print(processor.decode(output_with_prompt[0]))
# <|startofprev|> 91歳<|startoftranscript|><|ko|><|transcribe|><|notimestamps|> あっぶったでもスルガさん、91歳、力強い走りに変わってきます。<|endoftext|>
