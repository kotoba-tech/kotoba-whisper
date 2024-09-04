from pprint import pprint
from huggingface_hub import HfApi
from huggingface_hub import DatasetFilter

api = HfApi()
filt = DatasetFilter(author='japanese-asr')
dataset_list = api.list_datasets(filter=filt)
dataset_list = [i.id for i in dataset_list]
dataset_list = [i for i in dataset_list if "japanese-asr/whisper_transcriptions.reazonspeech.all_" in i and i.endswith("wer_10.0")]
for i in dataset_list:
    api.delete_repo(repo_id=i, repo_type="dataset")


api = HfApi()
filt = DatasetFilter(author='japanese-asr')
dataset_list = api.list_datasets(filter=filt)
dataset_list = [i.id for i in dataset_list]
dataset_list = [i for i in dataset_list if "japanese-asr/whisper_transcriptions.reazonspeech.all_" in i and "vectorize" not in i]
for i in dataset_list:
    api.delete_repo(repo_id=i, repo_type="dataset")


api = HfApi()
filt = DatasetFilter(author='japanese-asr')
dataset_list = api.list_datasets(filter=filt)
dataset_list = [i.id for i in dataset_list]
dataset_list = [i for i in dataset_list if "japanese-asr/whisper_transcriptions.reazonspeech.all_" in i and "vectorize" in i]
for i in dataset_list:
    api.delete_repo(repo_id=i, repo_type="dataset")
