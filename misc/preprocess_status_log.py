from pprint import pprint
from huggingface_hub import DatasetFilter, HfApi

api = HfApi()
filt = DatasetFilter(author='japanese-asr')
dataset_list = api.list_datasets(filter=filt)
dataset_list = [i.id for i in dataset_list]
dataset_list = [i for i in dataset_list if "japanese-asr/whisper_transcriptions.reazonspeech.all" in i and i.endswith("wer_10.0.vectorize")]
ids = [int(i.split(".all_")[1].split(".wer_")[0]) for i in dataset_list]
ids_left = [i for i in range(1, 83) if i not in ids]
pprint(sorted(ids_left))
