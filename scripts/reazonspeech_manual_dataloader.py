"""custom HF data loader to load a large audio dataset from local
- run `reazonspeech_manual_downloader.py` to download the desired data type "tiny/small/medium/large/all" first
- credit: https://huggingface.co/datasets/reazon-research/reazonspeech/blob/main/reazonspeech.py

Example:
```
import os
from datasets import load_dataset

dataset = load_dataset(
    f"{os.getcwd()}/distillation_scripts/reazonspeech_manual_dataloader.py",
    "all",
    dataset_dir_suffix="0_400",
    split="train",
    trust_remote_code=True
)
```
"""
import os
from glob import glob
import tarfile
import datasets
from datasets.tasks import AutomaticSpeechRecognition

# https://stackoverflow.com/questions/71692354/facing-ssl-error-with-huggingface-pretrained-models
os.environ['CURL_CA_BUNDLE'] = ''

# disable warning message
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
DATA_SIZE = ["tiny", "small", "medium", "large", "all"]
DATA_DIR_SUFFIX = os.environ.get("DATA_DIR_SUFFIX")


class ReazonSpeechConfig(datasets.BuilderConfig):

    def __init__(self, *args, **kwargs):
        self.dataset_dir_suffix = kwargs.pop("dataset_dir_suffix", None)

        super().__init__(*args, **kwargs)


class ReazonSpeech(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [ReazonSpeechConfig(name=name) for name in DATA_SIZE]
    DEFAULT_CONFIG_NAME = "tiny"
    DEFAULT_WRITER_BATCHDATA_SIZE = 256

    def _info(self):
        return datasets.DatasetInfo(
            task_templates=[AutomaticSpeechRecognition()],
            features=datasets.Features(
                {
                    "name": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16000),
                    "transcription": datasets.Value("string"),
                }
            )
        )

    def _split_generators(self, dl_manager):
        data_dir = f"{os.path.expanduser('~')}/.cache/reazon_manual_download/{self.config.name}"
        if self.config.dataset_dir_suffix is not None:
            data_dir = f"{data_dir}_{self.config.dataset_dir_suffix}"
        audio_files = glob(f"{data_dir}/*.tar")
        audio = [dl_manager.iter_archive(path) for path in audio_files]
        transcript_file = f"{data_dir}/{self.config.name}.{self.config.name}.tsv"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"audio_files": audio_files, "transcript_file": transcript_file, "audio": audio},
            ),
        ]

    def _generate_examples(self, audio_files, transcript_file, audio):

        # hashTable of a file and the associated transcript
        meta = {}
        with open(transcript_file, "r", encoding="utf-8") as fp:
            for line in fp:
                filename, transcription = line.rstrip("\n").split("\t")
                meta[filename] = transcription

        # iterator over audio
        for i, audio_single_dump in enumerate(audio):
            for filename, file in audio_single_dump:
                filename = filename.lstrip("./")
                if filename not in meta:  # skip audio without transcription
                    continue
                try:
                    data = file.read()
                except tarfile.ReadError:
                    print(f"skip {filename}")
                    continue
                yield filename, {
                    "name": filename,
                    "audio": {"path": os.path.join(audio_files[i], filename), "bytes": data},
                    "transcription": meta[filename],
                }
