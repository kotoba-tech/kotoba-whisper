import os
from shutil import rmtree
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets, DatasetDict

home_dir = os.path.expanduser("~")
chunk = 10

# whisper_transcriptions.reazonspeech.all.wer_10.0.vectorize (fix the name suffix from vectorize -> vectorized)
# missing: 54
tmp_dataset = []
ind = 8
for n in tqdm(range(81, 83)):
    dataset = f"japanese-asr/whisper_transcriptions.reazonspeech.all_{n}.wer_10.0.vectorize"
    data = load_dataset(dataset, split="train")
    tmp_dataset.append(data)
    if len(tmp_dataset) == chunk:
        new_data = DatasetDict({"train": concatenate_datasets(tmp_dataset)})
        while True:
            try:
                new_data.push_to_hub("japanese-asr/whisper_transcriptions.reazonspeech.all.wer_10.0.vectorized", config_name=f"split_{ind}")
                break
            except Exception:
                pass
        ind += 1
        tmp_dataset = []
    # clear cache
    rmtree(f"{home_dir}/.cache/huggingface/datasets/japanese-asr___whisper_transcriptions.reazonspeech.all_{n}.wer_10.0.vectorize")
if len(tmp_dataset) != 0:
    new_data = DatasetDict({"train": concatenate_datasets(tmp_dataset)})
    new_data.push_to_hub("japanese-asr/whisper_transcriptions.reazonspeech.all.wer_10.0.vectorized", config_name=f"split_{ind}")


# whisper_transcriptions.reazonspeech.all
tmp_dataset = []
ind = 0
for n in tqdm(range(1, 83)):
    dataset = f"japanese-asr/whisper_transcriptions.reazonspeech.all_{n}"
    data = load_dataset(dataset, split="train")
    tmp_dataset.append(data)
    if len(tmp_dataset) == chunk:
        new_data = DatasetDict({"train": concatenate_datasets(tmp_dataset)})
        while True:
            try:
                new_data.push_to_hub("japanese-asr/whisper_transcriptions.reazonspeech.all", config_name=f"split_{ind}")
                break
            except Exception:
                pass
        ind += 1
        tmp_dataset = []
    # clear cache
    rmtree(f"{home_dir}/.cache/huggingface/datasets/japanese-asr___whisper_transcriptions.reazonspeech.all_{n}")
if len(tmp_dataset) != 0:
    new_data = DatasetDict({"train": concatenate_datasets(tmp_dataset)})
    new_data.push_to_hub("japanese-asr/whisper_transcriptions.reazonspeech.all", config_name=f"split_{ind}")


# whisper_transcriptions.reazonspeech.all.wer_10.0
tmp_dataset = []
ind = 0
for n in tqdm(range(1, 83)):
    dataset = f"japanese-asr/whisper_transcriptions.reazonspeech.all_{n}.wer_10.0"
    data = load_dataset(dataset, split="train")
    tmp_dataset.append(data)
    if len(tmp_dataset) == chunk:
        new_data = DatasetDict({"train": concatenate_datasets(tmp_dataset)})
        while True:
            try:
                new_data.push_to_hub("japanese-asr/whisper_transcriptions.reazonspeech.all.wer_10.0", config_name=f"split_{ind}")
                break
            except Exception:
                pass
        ind += 1
        tmp_dataset = []
    # clear cache
    rmtree(f"{home_dir}/.cache/huggingface/datasets/japanese-asr___whisper_transcriptions.reazonspeech.all_{n}.wer_10.0")
if len(tmp_dataset) != 0:
    new_data = DatasetDict({"train": concatenate_datasets(tmp_dataset)})
    new_data.push_to_hub("japanese-asr/whisper_transcriptions.reazonspeech.all.wer_10.0", config_name=f"split_{ind}")


