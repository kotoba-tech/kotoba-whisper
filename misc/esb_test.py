# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""ESB datasets."""

import csv
from collections import defaultdict
import os
import json
import urllib
import re
import logging

import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
import requests
from io import BytesIO
from pathlib import Path
from huggingface_hub import HfApi, HfFolder
import datasets


_DESCRIPTIONS = {
    "ami": """
    The AMI Meeting Corpus is a multi-modal data set consisting of 100 hours of meeting recordings. 
    The AMI Meeting Corpus consists of 100 hours of meeting recordings. The recordings use a range of signals 
    synchronized to a common timeline. These include close-talking and far-field microphones, individual and 
    room-view video cameras, and output from a slide projector and an electronic whiteboard. 
    """,
    "spgispeech": """
    The SPGISpeech corpus is derived from company earnings calls manually transcribed by S&P Global, Inc. 
    according to a professional style guide detailing conventions for capitalization, punctuation, denormalization 
    of non-standard words and tran- scription of disfluencies in spontaneous speech. The basic unit of SPGISpeech is a 
    pair consisting of a 5 to 15 second long 16 bit, 16kHz mono wav audio file and its transcription.
    """,
    "voxpopuli": """
    A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation. 
    The raw data is collected from 2009-2020 European Parliament event recordings. 
    """,
    "tedlium": """
    The TED-LIUM corpus is English-language TED talks, with transcriptions, sampled at 16kHz. 
    All talks and text are property of TED Conferences LLC.
    """,
    "gigaspeech": """
    GigaSpeech is an evolving, multi-domain English speech recognition corpus with 10,000 hours of high quality
    labeled audio suitable for supervised training, and 40,000 hours of total audio suitable for semi-supervised
    and unsupervised training. Around 40,000 hours of transcribed audio is first collected from audiobooks, podcasts
    and YouTube, covering both read and spontaneous speaking styles, and a variety of topics, such as arts, science,
    sports, etc. A new forced alignment and segmentation pipeline is proposed to create sentence segments suitable
    for speech recognition training, and to filter out segments with low-quality transcription. For system training,
    GigaSpeech provides five subsets of different sizes, 10h, 250h, 1000h, 2500h, and 10000h.
    """,
    "librispeech": """
    LibriSpeech is a corpus of approximately 1000 hours of read English speech with sampling rate of 16 kHz,
    prepared by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read
    audiobooks from the LibriVox project, and has been carefully segmented and aligned.
    """,
    "common_voice": """
    Common Voice is Mozilla's initiative to help teach machines how real people speak. 
    The Common Voice dataset consists of a unique MP3 and corresponding text file. 
    """,
    "earnings22": """
    The Earnings 22 dataset ( also referred to as earnings22 ) is a 119-hour corpus of English-language earnings calls 
    collected from global companies. The primary purpose is to serve as a benchmark for industrial and academic 
    automatic speech recognition (ASR) models on real-world accented speech.
    """
}

_CITATIONS = {
    "ami": """
    @inproceedings{10.1007/11677482_3,
    author = {Carletta, Jean and Ashby, Simone and Bourban, Sebastien and Flynn, Mike and Guillemot, Mael and Hain, Thomas 
    and Kadlec, Jaroslav and Karaiskos, Vasilis and Kraaij, Wessel and Kronenthal, Melissa and Lathoud, Guillaume 
    and Lincoln, Mike and Lisowska, Agnes and McCowan, Iain and Post, Wilfried and Reidsma, Dennis and Wellner, Pierre},
    title = {The AMI Meeting Corpus: A Pre-Announcement},
    year = {2005},
    isbn = {3540325492},
    publisher = {Springer-Verlag},
    address = {Berlin, Heidelberg},
    url = {https://doi.org/10.1007/11677482_3},
    doi = {10.1007/11677482_3},
    booktitle = {Proceedings of the Second International Conference on Machine Learning for Multimodal Interaction},
    pages = {28–39},
    numpages = {12},
    location = {Edinburgh, UK},
    series = {MLMI'05}
    }
    """,
    "spgispeech": """
    @article{2021arXiv210402014O,
    author = {{O'Neill}, Patrick K. and {Lavrukhin}, Vitaly and {Majumdar}, Somshubra and {Noroozi}, Vahid and {Zhang}, Yuekai 
    and {Kuchaiev}, Oleksii and {Balam}, Jagadeesh and {Dovzhenko}, Yuliya and {Freyberg}, Keenan and {Shulman}, Michael D. 
    and {Ginsburg}, Boris and {Watanabe}, Shinji and {Kucsko}, Georg},
    title = "{SPGISpeech: 5,000 hours of transcribed financial audio for fully formatted end-to-end speech recognition}",
    journal = {arXiv e-prints},
    keywords = {Computer Science - Computation and Language, Electrical Engineering and Systems Science - Audio and Speech Processing},
    year = 2021,
    month = apr,
    eid = {arXiv:2104.02014},
    pages = {arXiv:2104.02014},
    eprint = {2104.02014},
    primaryClass = {cs.CL},
    adsurl = {https://ui.adsabs.harvard.edu/abs/2021arXiv210402014O},
    adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    """,
    "voxpopuli": """
    @inproceedings{wang-etal-2021-voxpopuli,
    title = "{V}ox{P}opuli: A Large-Scale Multilingual Speech Corpus for Representation Learning, 
    Semi-Supervised Learning and Interpretation",
    author = "Wang, Changhan  and Riviere, Morgane  and Lee, Ann  and Wu, Anne  and Talnikar, Chaitanya  and Haziza, 
    Daniel  and Williamson, Mary  and Pino, Juan  and Dupoux, Emmanuel",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th 
    International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.80",
    doi = "10.18653/v1/2021.acl-long.80",
    pages = "993--1003",
    }
    """,
    "tedlium": """
    @inproceedings{hernandez2018tedlium3,
    title={TED-LIUM 3: twice as much data and corpus repartition for experiments on speaker adaptation},
    author={Hernandez, Fran{\\c{c}}ois and Nguyen, Vincent and Ghannay, Sahar and Tomashenko, Natalia and Est{\\`e}ve, Yannick},
    booktitle={International Conference on Speech and Computer},
    pages={198--208},
    year={2018},
    organization={Springer}
    }
    """,
    "gigaspeech": """
    @article{DBLP:journals/corr/abs-2106-06909,
    author    = {Guoguo Chen and Shuzhou Chai and Guanbo Wang and Jiayu Du and Wei{-}Qiang Zhang and Chao Weng and Dan Su 
    and Daniel Povey and Jan Trmal and Junbo Zhang and Mingjie Jin and Sanjeev Khudanpur and Shinji Watanabe and 
    Shuaijiang Zhao and Wei Zou and Xiangang Li and Xuchen Yao and Yongqing Wang and Yujun Wang and Zhao You and Zhiyong Yan},
    title     = {GigaSpeech: An Evolving, Multi-domain {ASR} Corpus with 10, 000 Hours
               of Transcribed Audio},
    journal   = {CoRR},
    volume    = {abs/2106.06909},
    year      = {2021},
    url       = {https://arxiv.org/abs/2106.06909},
    eprinttype = {arXiv},
    eprint    = {2106.06909},
    timestamp = {Wed, 29 Dec 2021 14:29:26 +0100},
    biburl    = {https://dblp.org/rec/journals/corr/abs-2106-06909.bib},
    bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    """,
    "librispeech": """
    @inproceedings{panayotov2015librispeech,
    title={Librispeech: an ASR corpus based on public domain audio books},
    author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
    booktitle={Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on},
    pages={5206--5210},
    year={2015},
    organization={IEEE}
    }
    """,
    "common_voice": """
    @inproceedings{commonvoice:2020,
    author = {Ardila, R. and Branson, M. and Davis, K. and Henretty, M. and Kohler, M. and Meyer, J. and Morais, R. and Saunders, L. and Tyers, F. M. and Weber, G.},
    title = {Common Voice: A Massively-Multilingual Speech Corpus},
    booktitle = {Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020)},
    pages = {4211--4215},
    year = 2020
    }
    """,
    "earnings22": """
    @misc{https://doi.org/10.48550/arxiv.2203.15591,
    doi = {10.48550/ARXIV.2203.15591},
    url = {https://arxiv.org/abs/2203.15591},
    author = {Del Rio, Miguel and Ha, Peter and McNamara, Quinten and Miller, Corey and Chandra, Shipra},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Earnings-22: A Practical Benchmark for Accents in the Wild},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution Share Alike 4.0 International}
    }
    """,
}

_HOMEPAGE_URLS = {
    "ami": "https://groups.inf.ed.ac.uk/ami/corpus/",
    "spgispeech": "https://datasets.kensho.com/datasets/spgispeech",
    "voxpopuli": "https://github.com/facebookresearch/voxpopuli",
    "tedlium": "https://www.openslr.org/51/",
    "gigaspeech": "https://github.com/SpeechColab/GigaSpeech",
    "librispeech": "http://www.openslr.org/12",
    "common_voice": "https://commonvoice.mozilla.org/en/datasets",
    "earnings22": "https://github.com/revdotcom/speech-datasets/tree/main/earnings22",
}

_LICENSES = {
    "ami": "CC BY 4.0",
    "spgispeech": "Custom license (academic use only)",
    "voxpopuli": "CC0, also see https://www.europarl.europa.eu/legal-notice/en/",
    "tedlium": "Creative Commons BY-NC-ND 3.0 (http://creativecommons.org/licenses/by-nc-nd/3.0/deed.en)",
    "gigaspeech": "Apache License 2.0",
    "librispeech": "CC BY 4.0",
    "common_voice": "Mozilla Public License 2.0 (https://github.com/common-voice/common-voice/blob/main/LICENSE)",
    "earnings22": "CC BY-SA 4.0",
}

_DATASET_TO_CONFIGS = {
    "spgispeech": ["l", "s", "m"],
    "gigaspeech": ["l", "xs", "s", "m", "xl"],
    "librispeech": ["default", "clean.100", "clean.360", "other.500"],
}

_ALL_CONFIGS = list(_DATASET_TO_CONFIGS) + ["earnings22", "ami", "tedlium", "voxpopuli", "common_voice"]


class ESBConfig(datasets.BuilderConfig):
    """BuilderConfig for the ESB datasets. """

    def __init__(self, name, subconfig, description, citation, homepage, license, **kwargs):
        """
        Args:
          name: `string`, name of a dataset to be downloaded (for example, "gigaspeech")
          subconfig: `string`, specific configuration of a dataset, relevant for "spgispeech", "gigaspeech", and "librispeech"
          description: `string`: dataset decsription
          citation: `string`: dataset citation
          homepage: `string`: dataset homepage
          license: `string`: dataset license
          **kwargs: keyword arguments forwarded to super.
        """
        if name in _DATASET_TO_CONFIGS:
            # first config is the default one
            self.subconfig = _DATASET_TO_CONFIGS[name][0] if subconfig == "default" else subconfig
        else:
            self.subconfig = None

        super(ESBConfig, self).__init__(
            name=name,
            version=datasets.Version("1.0.0", ""),
            **kwargs
        )
        self.description = description
        self.citation = citation
        self.homepage = homepage
        self.license = license


def _build_config(name, subconfig):
    return ESBConfig(
        name=name,
        subconfig=subconfig,
        description=_DESCRIPTIONS[name],
        citation=_CITATIONS[name],
        homepage=_HOMEPAGE_URLS[name],
        license=_LICENSES[name],
    )


class ESBDatasets(datasets.GeneratorBasedBuilder):
    """ESB benchmark dataset dataset."""

    DEFAULT_WRITER_BATCH_SIZE = 256
    BUILDER_CONFIGS = [
        _build_config(name, subconfig="default") for name in _ALL_CONFIGS
    ]

    def _info(self):
        features = datasets.Features(
                    {
                        "audio": datasets.Audio(sampling_rate=16_000),
                        "dataset": datasets.Value("string"),
                        "text": datasets.Value("string"),
                        "id": datasets.Value("string"),
                    }
                )
        return datasets.DatasetInfo(  # TODO: add benchmark's own license and description
                features=features,
                description=self.config.description,
                homepage=self.config.homepage,
                license=self.config.license,
                citation=self.config.citation,
            )

    def _split_generators(self, dl_manager):
        if self.config.name == "ami":
            return self._ami_split_generators(dl_manager)
        elif self.config.name == "spgispeech":
            return self._spgispeech_split_generators(dl_manager)
        elif self.config.name == "voxpopuli":
            return self._voxpopuli_split_generators(dl_manager)
        elif self.config.name == "tedlium":
            return self._tedlium_split_generators(dl_manager)
        elif self.config.name == "gigaspeech":
            return self._gigaspeech_split_generators(dl_manager)
        elif self.config.name == "librispeech":
            return self._librispeech_split_generators(dl_manager)
        elif self.config.name == "common_voice":
            return self._common_voice_split_generators(dl_manager)
        elif self.config.name == "earnings22":
            return self._earnings_split_generators(dl_manager)

    def _generate_examples(self, *args, **kwargs):
        if self.config.name == "ami":
            yield from self._ami_generate_examples(*args, **kwargs)
        elif self.config.name == "spgispeech":
            yield from self._spgispeech_generate_examples(*args, **kwargs)
        elif self.config.name == "voxpopuli":
            yield from self._voxpopuli_generate_examples(*args, **kwargs)
        elif self.config.name == "tedlium":
            yield from self._tedlium_generate_examples(*args, **kwargs)
        elif self.config.name == "gigaspeech":
            yield from self._gigaspeech_generate_examples(*args, **kwargs)
        elif self.config.name == "librispeech":
            yield from self._librispeech_generate_examples(*args, **kwargs)
        elif self.config.name == "common_voice":
            yield from self._common_voice_generate_examples(*args, **kwargs)
        elif self.config.name == "earnings22":
            yield from self._earnings_generate_examples(*args, **kwargs)

    def _ami_split_generators(self, dl_manager):
        splits = ["dev", "eval"]
        audio_archives_urls = {}
        for split in splits:
            audio_archives_urls[split] = [
                _AMI_AUDIO_ARCHIVE_URL.format(split=split, _id=m) for m in _AMI_SAMPLE_IDS[split]
            ]

        audio_archives = dl_manager.download(audio_archives_urls)
        local_extracted_archives_paths = dl_manager.extract(audio_archives) if not dl_manager.is_streaming else {
            split: [None] * len(audio_archives[split]) for split in splits
        }

        annotations_urls = {split: _AMI_ANNOTATIONS_ARCHIVE_URL.format(split=split) for split in splits}
        annotations = dl_manager.download(annotations_urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "audio_archives": [dl_manager.iter_archive(archive) for archive in audio_archives["dev"]],
                    "local_extracted_archives_paths": local_extracted_archives_paths["dev"],
                    "annotation": annotations["dev"],
                    "split": "dev"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "audio_archives": [dl_manager.iter_archive(archive) for archive in audio_archives["eval"]],
                    "local_extracted_archives_paths": local_extracted_archives_paths["eval"],
                    "annotation": annotations["eval"],
                    "split": "eval"
                },
            ),
        ]

    def _ami_generate_examples(self, audio_archives, local_extracted_archives_paths, annotation, split):
        assert len(audio_archives) == len(local_extracted_archives_paths)

        with open(annotation, "r", encoding="utf-8") as f:
            transcriptions = {}
            for line in f.readlines():
                line_items = line.strip().split()
                _id = line_items[0]
                text = " ".join(line_items[1:])
                _, meeting_id, microphone_id, speaker_id, begin_time, end_time = _id.split("_")
                audio_filename = "_".join([split, _id.lower()]) + ".wav"

                transcriptions[audio_filename] = {
                    "id": _id,
                    "text": text if split != "eval" else "",
                }

        features = ["id", "text"]
        for archive, local_archive_path in zip(audio_archives, local_extracted_archives_paths):
            for audio_path, audio_file in archive:
                # audio_path is like 'EN2001a/train_ami_en2001a_h00_mee068_0414915_0415078.wav'
                audio_meta = transcriptions[audio_path.split("/")[-1]]

                yield audio_path, {
                    "audio": {
                        "path": os.path.join(local_archive_path, audio_path) if local_archive_path else audio_path,
                        "bytes": audio_file.read(),
                    },
                    "dataset": "ami",
                    **{feature: audio_meta[feature] for feature in features}
                }

    def _spgispeech_split_generators(self, dl_manager):
        subconfig = self.config.subconfig
        subsets = [subconfig] + ["dev", "test"]

        meta_path = dl_manager.download_and_extract(
            {subset: os.path.join(_SPGISPEECH_META_BASE_URL, _SPGISPEECH_META_FILENAMES[subset]) for subset in subsets}
        )

        archive_urls = defaultdict(list)
        for subset in subsets:
            for subset_dir in _SPGISPEECH_SUBSET_TO_DIR[subset]:
                for archive_name in _SPGISPEECH_AUDIO_ARCHIVES_NAMES[subset_dir]:
                    archive_urls[subset].append(os.path.join(_SPGISPEECH_AUDIO_BASE_URL, subset_dir, archive_name))

        archive_paths = dl_manager.download(archive_urls)

        local_extracted_archive_paths = (
            dl_manager.extract(archive_paths)
            if not dl_manager.is_streaming
            else {subset: [None] * len(archive_paths[subset]) for subset in subsets}
        )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "local_extracted_archive_paths": local_extracted_archive_paths[subconfig],
                    "archives": [dl_manager.iter_archive(path) for path in archive_paths[subconfig]],
                    "meta_path": meta_path[subconfig],
                    "is_test": False,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "local_extracted_archive_paths": local_extracted_archive_paths["dev"],
                    "archives": [dl_manager.iter_archive(path) for path in archive_paths["dev"]],
                    "meta_path": meta_path["dev"],
                    "is_test": False,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "local_extracted_archive_paths": local_extracted_archive_paths["test"],
                    "archives": [dl_manager.iter_archive(path) for path in archive_paths["test"]],
                    "meta_path": meta_path["test"],
                    "is_test": True,
                },
            ),
        ]

    def _spgispeech_generate_examples(self, local_extracted_archive_paths, archives, meta_path, is_test):
        # define the expected metadata dict keys,
        # some files have metadata with erroneous entries that we have to filter out
        dict_keys = {"id": "wav_filename", "text": "transcript"}

        logging.info("Reading spgispeech metadata")
        with open(meta_path, encoding="utf-8") as f:
            csvreader = csv.DictReader(f, delimiter="|")
            metadata = {x["wav_filename"]: dict((k, x[v]) for k, v in dict_keys.items()) for x in tqdm(csvreader, leave=False)}

        for local_extracted_archive_path, archive in zip(local_extracted_archive_paths, archives):
            # Here we iterate over all the files within the TAR archive:
            for audio_filename, audio_file in archive:
                audio_filename = audio_filename.lstrip("./")
                # if an audio file exists locally (i.e. in default, non-streaming mode) set the full path to it
                # joining path to directory that the archive was extracted to and audio filename.
                path = (
                    os.path.join(local_extracted_archive_path, audio_filename)
                    if local_extracted_archive_path
                    else audio_filename
                )
                # get the .wav filename by removing the directory path from the audio filename
                wav_filename = "/".join(audio_filename.split("/")[-2:])
                example = dict(metadata[wav_filename])
                if is_test: example["text"] = ""
                example["audio"] = {"path": path, "bytes": audio_file.read()}
                example["dataset"] = "spgispeech"
                yield audio_filename, example

    def _voxpopuli_split_generators(self, dl_manager):
        n_shards_path = dl_manager.download_and_extract(_VOXPOPULI_N_SHARDS_FILE)
        with open(n_shards_path) as f:
            n_shards = json.load(f)["en"]  # we use only English language in this benchmark
        splits = ["dev", "test"]

        audio_urls = {}
        for split in splits:
            audio_urls[split] = [
                _VOXPOPULI_AUDIO_ARCHIVE_PATH.format(split=split, n_shard=i) for i in range(n_shards[split])
            ]

        meta_urls = {
            split: _VOXPOPULI_METADATA_PATH.format(split=split) for split in splits
        }

        meta_paths = dl_manager.download_and_extract(meta_urls)
        audio_paths = dl_manager.download(audio_urls)

        local_extracted_audio_paths = (
            dl_manager.extract(audio_paths) if not dl_manager.is_streaming else
            {
                split: [None] * len(audio_paths[split]) for split in splits
            }
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "audio_archives": [dl_manager.iter_archive(archive) for archive in audio_paths["dev"]],
                    "local_extracted_archives_paths": local_extracted_audio_paths["dev"],
                    "meta_path": meta_paths["dev"],
                    "is_test": False,
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "audio_archives": [dl_manager.iter_archive(archive) for archive in audio_paths["test"]],
                    "local_extracted_archives_paths": local_extracted_audio_paths["test"],
                    "meta_path": meta_paths["test"],
                    "is_test": True,
                }
            ),
        ]

    def _voxpopuli_generate_examples(self, audio_archives, local_extracted_archives_paths, meta_path, is_test):
        assert len(audio_archives) == len(local_extracted_archives_paths)

        logging.info("Reading voxpopuli metadata.")
        with open(meta_path) as f:
            metadata = {x["id"]: x for x in tqdm(csv.DictReader(f, delimiter="\t"), leave=False)}

        for audio_archive, local_extracted_archive_path in zip(audio_archives, local_extracted_archives_paths):
            for audio_filename, audio_file in audio_archive:
                audio_id = audio_filename.split(os.sep)[-1].split(".wav")[0]
                path = os.path.join(local_extracted_archive_path, audio_filename) if local_extracted_archive_path else audio_filename

                yield audio_id, {
                    "id": audio_id,
                    "text": metadata[audio_id]["normalized_text"].lower() if not is_test else "",
                    "audio": {"path": path, "bytes": audio_file.read()},
                    "dataset": "voxpopuli",
                }

    def _librispeech_split_generators(self, dl_manager):
        dev_splits, test_splits = ["dev.clean", "dev.other"], ["test.clean", "test.other"]
        dl_urls = {config_name: _LIBRISPEECH_DL_URLS[config_name] for config_name in dev_splits + test_splits}
        archive_paths = dl_manager.download(dl_urls)
        # (Optional) In non-streaming mode, we can extract the archive locally to have actual local audio files:
        local_extracted_archives = dl_manager.extract(archive_paths) if not dl_manager.is_streaming else {}
        dev_splits = [
            datasets.SplitGenerator(
                name="validation.clean",
                gen_kwargs={
                    "local_extracted_archives": [local_extracted_archives.get("dev.clean")],
                    "archives": [dl_manager.iter_archive(archive_paths["dev.clean"])],
                    "is_test": False,
                },
            ),
            datasets.SplitGenerator(
                name="validation.other",
                gen_kwargs={
                    "local_extracted_archives": [local_extracted_archives.get("dev.other")],
                    "archives": [dl_manager.iter_archive(archive_paths["dev.other"])],
                    "is_test": False,
                },
            ),
        ]
        test_splits = [
            datasets.SplitGenerator(
                name="test.clean",
                gen_kwargs={
                    "local_extracted_archives": [local_extracted_archives.get("test.clean")],
                    "archives": [dl_manager.iter_archive(archive_paths["test.clean"])],
                    "is_test": True,
                },
            ),
            datasets.SplitGenerator(
                name="test.other",
                gen_kwargs={
                    "local_extracted_archives": [local_extracted_archives.get("test.other")],
                    "archives": [dl_manager.iter_archive(archive_paths["test.other"])],
                    "is_test": True,
                },
            ),
        ]
        return dev_splits + test_splits

    def _librispeech_generate_examples(self, archives, local_extracted_archives, is_test):
        key = 0
        audio_data = {}
        transcripts = []
        for archive, local_extracted_archive in zip(archives, local_extracted_archives):
            for path, f in archive:
                if path.endswith(".flac"):
                    id_ = path.split("/")[-1][: -len(".flac")]
                    audio_data[id_] = f.read()
                elif path.endswith(".trans.txt"):
                    for line in f:
                        if line:
                            line = line.decode("utf-8").strip()
                            id_, transcript = line.split(" ", 1)

                            # Error correction
                            transcript = transcript.lower()

                            audio_file = f"{id_}.flac"
                            audio_file = (
                                os.path.join(local_extracted_archive, audio_file)
                                if local_extracted_archive
                                else audio_file
                            )
                            transcripts.append(
                                {
                                    "id": id_,
                                    "file": audio_file,
                                    "text": transcript if not is_test else "",
                                }
                            )
                if audio_data and len(audio_data) == len(transcripts):
                    for transcript in transcripts:
                        audio = {"path": transcript["file"], "bytes": audio_data[transcript["id"]]}
                        del transcript["file"]
                        yield key, {"audio": audio, "dataset": "librispeech", **transcript}
                        key += 1
                    audio_data = {}
                    transcripts = []

    def _common_voice_get_bundle_url(self, locale, url_template):
        # path = encodeURIComponent(path)
        path = url_template.replace("{locale}", locale)
        path = urllib.parse.quote(path.encode("utf-8"), safe="~()*!.'")
        # use_cdn = self.config.size_bytes < 20 * 1024 * 1024 * 1024
        # response = requests.get(f"{_API_URL}/bucket/dataset/{path}/{use_cdn}", timeout=10.0).json()
        response = requests.get(f"{_COMMON_VOICE_API_URL}/bucket/dataset/{path}", timeout=10.0).json()
        return response["url"]

    def _common_voice_log_download(self, locale, bundle_version, auth_token):
        if isinstance(auth_token, bool):
            auth_token = HfFolder().get_token()
        whoami = HfApi().whoami(auth_token)
        email = whoami["email"] if "email" in whoami else ""
        payload = {"email": email, "locale": locale, "dataset": bundle_version}
        requests.post(f"{_COMMON_VOICE_API_URL}/{locale}/downloaders", json=payload).json()

    def _common_voice_split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        hf_auth_token = dl_manager.download_config.use_auth_token
        if hf_auth_token is None:
            raise ConnectionError(
                "Please set use_auth_token=True or use_auth_token='<TOKEN>' to download this dataset"
            )

        bundle_version = _COMMON_VOICE_BUNDLE_URL_TEMPLATE.split("/")[0]
        dl_manager.download_config.ignore_url_params = True

        self._common_voice_log_download("en", bundle_version, hf_auth_token)
        archive_path = dl_manager.download(self._common_voice_get_bundle_url("en", _COMMON_VOICE_BUNDLE_URL_TEMPLATE))
        local_extracted_archive = dl_manager.extract(archive_path) if not dl_manager.is_streaming else None

        path_to_data = "/".join([bundle_version, "en"])
        path_to_clips = "/".join([path_to_data, "clips"]) if path_to_data else "clips"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "archive_iterator": dl_manager.iter_archive(archive_path),
                    "metadata_filepath": "/".join([path_to_data, "dev.tsv"]) if path_to_data else "dev.tsv",
                    "path_to_clips": path_to_clips,
                    "is_test": False,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "archive_iterator": dl_manager.iter_archive(archive_path),
                    "metadata_filepath": "/".join([path_to_data, "test.tsv"]) if path_to_data else "test.tsv",
                    "path_to_clips": path_to_clips,
                    "is_test": True,
                },
            ),
        ]

    def _common_voice_generate_examples(
        self,
        local_extracted_archive,
        archive_iterator,
        metadata_filepath,
        path_to_clips,
        is_test,
    ):
        """Yields examples."""
        data_fields = list(self._info().features.keys())
        metadata = {}
        metadata_found = False
        for path, f in archive_iterator:
            if path == metadata_filepath:
                metadata_found = True
                lines = (line.decode("utf-8") for line in f)
                reader = csv.DictReader(lines, delimiter="\t", quoting=csv.QUOTE_NONE)
                for row in reader:
                    # set absolute path for mp3 audio file
                    if not row["path"].endswith(".mp3"):
                        row["path"] += ".mp3"
                    row["path"] = os.path.join(path_to_clips, row["path"])
                    # accent -> accents in CV 8.0
                    if "accents" in row:
                        row["accent"] = row["accents"]
                        del row["accents"]
                    # if data is incomplete, fill with empty values
                    for field in data_fields:
                        if field not in row:
                            row[field] = ""
                    metadata[row["path"]] = row
            elif path.startswith(path_to_clips):
                assert metadata_found, "Found audio clips before the metadata TSV file."
                if not metadata:
                    break
                if path in metadata:
                    dict_result = dict(metadata[path])
                    # set the audio feature and the path to the extracted file
                    path = os.path.join(local_extracted_archive, path) if local_extracted_archive else path
                    result = {"id": dict_result["client_id"], "dataset": "common_voice",
                              "audio": {"path": path, "bytes": f.read()}}

                    # Error correction
                    text = dict_result["sentence"]
                    if text.startswith('"') and text.endswith('"'):
                        # we can remove trailing quotation marks as they do not affect the transcription
                        text = text[1:-1]
                    if len(text) == 0:
                        continue
                    # replace double quotation marks with single
                    text = text.replace('""', '"')
                    result["text"] = text if not is_test else ""

                    yield path, result

    def _tedlium_split_generators(self, dl_manager):
        archive_path = dl_manager.download(_TEDLIUM_URLS)
        # (Optional) In non-streaming mode, we can extract the archive locally to have actual local audio files:
        local_extracted_archive = dl_manager.extract(archive_path) if not dl_manager.is_streaming else {}
        split_paths = [
                          (datasets.Split.VALIDATION, "dev"),
                          (datasets.Split.TEST, "test"),
                      ]
        splits = []
        for split, split_name in split_paths:
            kwargs = {
                "filepath": [dl_manager.iter_archive(sharded_path) for sharded_path in archive_path[split_name]],
                "local_extracted_archive": local_extracted_archive.get(split_name),
                "split_path": split_name,
            }
            splits.append(datasets.SplitGenerator(name=split, gen_kwargs=kwargs))
        return splits

    def _tedlium_generate_examples(self, filepath, local_extracted_archive, split_path):
        """Generate examples from a TED-LIUM stm file."""
        if local_extracted_archive:
            for local_archive in local_extracted_archive:
                # The stm directory houses the speaker and transcription information in .stm format
                split_dir = os.path.join(local_archive, split_path)
                stm_files = [os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith(".stm")]
                for file in stm_files:
                    # the .sph speaker file almost always has the same file name as the .stm file
                    speaker_file = Path(file).stem
                    audio_file = os.path.join(split_dir, speaker_file + ".sph")
                    segment, sampling_rate = sf.read(audio_file, dtype=np.int16)
                    with open(file) as f:
                        for line in f:
                            line = line.strip()
                            fn, channel, speaker, start, end, label, transcript = line.split(" ", 6)
                            transcript = _maybe_trim_suffix(transcript)

                            # Error correction
                            transcript = transcript.lower()
                            if transcript in ignore_segments:
                                continue
                            # delete the <unk> token from the text
                            transcript = transcript.replace("<unk>", "")
                            # replace spaced apostrophes with un-spaced (it 's -> it's)
                            for contraction in tedlium_contractions:
                                transcript = transcript.replace(contraction, contraction[1:])
                            # JIWER compliance (for WER/CER calc.)
                            # remove multiple spaces
                            transcript = re.sub(r"\s\s+", " ", transcript)
                            # strip trailing spaces
                            transcript = transcript.strip()
                            if len(transcript) == 0:
                                continue

                            if speaker_file != fn:
                                # handle the case where the stm file does not have the same file name as the transcript
                                speaker_file = fn
                                audio_file = os.path.join(split_dir, speaker_file + ".sph")
                                segment, sampling_rate = sf.read(audio_file, dtype=np.int16)
                            samples = _extract_audio_segment(segment, sampling_rate, float(start), float(end))
                            key = "-".join([speaker, start, end, label])
                            example = {
                                "audio": {"path": audio_file, "array": samples, "sampling_rate": sampling_rate},
                                "text": transcript if split_path != "test" else "",
                                "id": key,
                                "dataset": "tedlium",
                            }
                            yield key, example

        else:
            audio_data = {}
            transcripts = defaultdict(list)
            for file in filepath:
                for path, f in file:
                    if path.endswith(".sph"):
                        # get the speaker id
                        fn = path.split("/")[-1].strip(".sph")
                        # read the audio data from raw byte form and add key-value pair to dict
                        audio_data[fn] = sf.read(BytesIO(f.read()), dtype=np.int16)
                    elif path.endswith(".stm"):
                        for line in f:
                            if line:
                                line = line.decode("utf-8").strip()
                                fn, channel, speaker, start, end, label, transcript = line.split(" ", 6)
                                transcript = _maybe_trim_suffix(transcript)

                                # Error correction
                                transcript = transcript.lower()
                                if transcript in ignore_segments:
                                    continue
                                # delete the <unk> token from the text
                                transcript = transcript.replace("<unk>", "")
                                # replace spaced apostrophes with un-spaced (it 's -> it's)
                                for contraction in tedlium_contractions:
                                    transcript = transcript.replace(contraction, contraction[1:])
                                # JIWER compliance (for WER/CER calc.)
                                # remove multiple spaces
                                transcript = re.sub(r"\s\s+", " ", transcript)
                                # strip trailing spaces
                                transcript = transcript.strip()
                                if len(transcript) == 0:
                                    continue

                                audio_file = path.replace("stm", "sph")
                                key = "-".join([speaker, start, end, label])
                                # append metadata information to the dict of transcripts for the associated speaker
                                transcripts[fn].append(
                                    {
                                        "text": transcript,
                                        "file": audio_file,
                                        "id": key,
                                        "start": start,
                                        "end": end,
                                        "channel": channel,
                                        "fn": fn,
                                    }
                                )

                    if audio_data and audio_data.keys() == transcripts.keys():
                        for fn, speaker in transcripts.items():
                            for transcript in speaker:
                                segment, sampling_rate = audio_data[transcript["fn"]]
                                samples = _extract_audio_segment(
                                    segment,
                                    sampling_rate,
                                    float(transcript["start"]),
                                    float(transcript["end"]),
                                )
                                audio = {"path": transcript["file"], "array": samples,
                                         "sampling_rate": sampling_rate}
                                key = transcript["id"]
                                yield key, {
                                    "audio": audio,
                                    "text": transcript["text"] if split_path != "test" else "",
                                    "dataset": "tedlium",
                                    "id": transcript["id"],
                                }
                        audio_data = {}
                        transcripts = defaultdict(list)

    def _gigaspeech_split_generators(self, dl_manager):
        splits_to_configs = {
            "dev": ["dev"],
            "test": ["test"],
        }

        # 1. prepare sharded archives with audio files
        audio_archives_urls = defaultdict(list)
        for split, subsets in splits_to_configs.items():
            for subset in subsets:
                audio_archives_urls[split].extend(
                    [
                        _GIGASPEECH_AUDIO_ARCHIVE_URL.format(subset=subset, is_additional=_is_additional(subset),
                                                             archive_id=i)
                        for i in range(_GIGASPEECH_N_ARCHIVES[subset])
                    ]
                )
        audio_archives_paths = dl_manager.download(audio_archives_urls)
        local_audio_archives_paths = dl_manager.extract(audio_archives_paths) if not dl_manager.is_streaming \
            else {}

        # 2. prepare sharded metadata csv files
        meta_urls = defaultdict(list)
        for split, subsets in splits_to_configs.items():
            for subset in subsets:
                meta_urls[split].extend(
                    [
                        _GIGASPEECH_META_URL.format(subset=subset, is_additional=_is_additional(subset), archive_id=i)
                        for i in range(_GIGASPEECH_N_ARCHIVES[subset])
                    ]
                )
        meta_paths = dl_manager.download_and_extract(meta_urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "audio_archives_iterators": [
                        dl_manager.iter_archive(archive_path) for archive_path in audio_archives_paths["dev"]
                    ],
                    "local_audio_archives_paths": local_audio_archives_paths.get("dev"),
                    "meta_paths": meta_paths["dev"],
                    "is_test": False,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "audio_archives_iterators": [
                        dl_manager.iter_archive(archive_path) for archive_path in audio_archives_paths["test"]
                    ],
                    "local_audio_archives_paths": local_audio_archives_paths.get("test"),
                    "meta_paths": meta_paths["test"],
                    "is_test": True,
                },
            ),
        ]

    def _gigaspeech_generate_examples(self, audio_archives_iterators, local_audio_archives_paths, meta_paths, is_test):
        assert len(audio_archives_iterators) == len(meta_paths)
        if local_audio_archives_paths:
            assert len(audio_archives_iterators) == len(local_audio_archives_paths)

        for i, (meta_path, audio_archive_iterator) in enumerate(zip(meta_paths, audio_archives_iterators)):
            meta_dict = dict()
            with open(meta_path) as csvfile:
                meta_csv = csv.DictReader(csvfile)
                for line in meta_csv:
                    meta_dict[line["sid"]] = line

            for audio_path_in_archive, audio_file in audio_archive_iterator:
                # `audio_path_in_archive` is like "dev_chunks_0000/YOU1000000029_S0000095.wav"
                audio_filename = os.path.split(audio_path_in_archive)[1]
                audio_id = audio_filename.split(".wav")[0]
                audio_meta = meta_dict[audio_id]
                audio_meta["id"] = audio_meta.pop("sid")
                text = audio_meta.pop("text_tn")

                # Error correction
                text = text.lower()
                if text in ignore_segments:
                    continue
                for junk_token in gigaspeech_junk_tokens:
                    text = text.replace(junk_token, "")
                # convert spelled out punctuation to symbolic form
                for punctuation, replacement in gigaspeech_punctuation.items():
                    text = text.replace(punctuation, replacement)
                # JIWER compliance (for WER/CER calc.)
                # remove multiple spaces
                text = re.sub(r"\s\s+", " ", text)
                # strip trailing spaces
                text = text.strip()
                if len(text) == 0:
                    continue

                audio_meta["text"] = text if not is_test else ""

                path = os.path.join(local_audio_archives_paths[i], audio_path_in_archive) if local_audio_archives_paths \
                    else audio_path_in_archive

                yield audio_id, {
                    "audio": {"path": path, "bytes": audio_file.read()},
                    "dataset": "gigaspeech",
                    **{feature: value for feature, value in audio_meta.items() if feature in self.info.features}
                }

    def _earnings_split_generators(self, dl_manager):
        meta_url = _EARNINGS_BASE_URL + "metadata.csv"
        meta_path = dl_manager.download_and_extract(meta_url)

        with open(meta_path, encoding="utf-8") as f:
            csvreader = csv.DictReader(f, delimiter=",")
            metadata, all_ids = {}, set()
            for row in csvreader:
                all_ids.update([row["source_id"]])
                metadata[row["file"]] = row["sentence"]  # we need only text in this benchmark

        split_to_ids = {"dev": _EARNINGS_DEV_IDS, "test": _EARNINGS_TEST_IDS}

        dl_urls = {}
        for split, split_ids in split_to_ids.items():
            dl_urls[split] = [_EARNINGS_BASE_URL + f"data/chunked/{source_id}.tar.gz" for source_id in split_ids]
        archive_paths = dl_manager.download(dl_urls)

        local_extracted_archive_paths = (
            dl_manager.extract(archive_paths)
            if not dl_manager.is_streaming
            else {split: [None] * len(archive_paths[split]) for split in ["dev", "test"]}
        )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "local_extracted_archive_paths": local_extracted_archive_paths["dev"],
                    "archives": [dl_manager.iter_archive(path) for path in archive_paths["dev"]],
                    "metadata": metadata,
                    "is_test": False,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "local_extracted_archive_paths": local_extracted_archive_paths["test"],
                    "archives": [dl_manager.iter_archive(path) for path in archive_paths["test"]],
                    "metadata": metadata,
                    "is_test": True,
                },
            ),
        ]

    def _earnings_generate_examples(self, local_extracted_archive_paths, archives, metadata, is_test):
        for local_extracted_archive_path, archive in zip(local_extracted_archive_paths, archives):
            # Here we iterate over all the files within the TAR archive:
            for audio_filename, audio_file in archive:
                audio_filename = audio_filename.lstrip("./")
                # if an audio file exists locally (i.e. in default, non-streaming mode) set the full path to it
                # joining path to directory that the archive was extracted to and audio filename.
                path = (
                    os.path.join(local_extracted_archive_path, audio_filename)
                    if local_extracted_archive_path
                    else audio_filename
                )

                # Error correction
                text = metadata[audio_filename]
                if text.lower() in ignore_segments:
                    continue
                # Remove  junk tokens
                for junk_token in earnings_junk_tokens:
                    text = text.replace(junk_token, "")
                # JIWER compliance (for WER/CER calc.)
                # remove multiple spaces
                text = re.sub(r"\s\s+", " ", text)
                # strip trailing spaces
                text = text.strip()
                if len(text) == 0:
                    continue

                yield audio_filename, {
                    "id": audio_filename,
                    "text": text if not is_test else "",
                    "dataset": "earnings22",
                    "audio": {"path": path, "bytes": audio_file.read()}
                }


def _maybe_trim_suffix(transcript):
    # stm files for the TEDLIUM release 1 train split contain a key (enclosed in
    # parens) at the end.
    splits = transcript.rsplit(" ", 1)
    transcript = splits[0]
    if len(splits) > 1:
        suffix = splits[-1]
        if not suffix.startswith("("):
            transcript += " " + suffix
    return transcript


def _extract_audio_segment(segment, sampling_rate, start_sec, end_sec):
    """Extracts segment of audio samples (as an ndarray) from the given segment."""
    # The dataset only contains mono audio.
    start_sample = int(start_sec * sampling_rate)
    end_sample = min(int(end_sec * sampling_rate), segment.shape[0])
    samples = segment[start_sample:end_sample]
    return samples


def _parse_gender(label_str):
    """Parse gender string from STM "<label>" field."""
    gender = re.split(",|_", label_str)[-1][:-1]
    # Fix inconsistencies in the data.
    if not gender:
        gender = -1  # Missing label.
    elif gender == "<NA":  # In TEDLIUM release 3 training data.
        gender = -1  # Missing label.
    elif gender == "F":
        gender = "female"
    elif gender == "M":
        gender = "male"
    return gender


def _is_additional(name):
    if name in {"s", "m", "l", "xl"}:
        return "_additional"
    return ""


_AMI_TRAIN_SAMPLE_IDS = [
    "EN2001a",
    "EN2001b",
    "EN2001d",
    "EN2001e",
    "EN2003a",
    "EN2004a",
    "EN2005a",
    "EN2006a",
    "EN2006b",
    "EN2009b",
    "EN2009c",
    "EN2009d",
    "ES2002a",
    "ES2002b",
    "ES2002c",
    "ES2002d",
    "ES2003a",
    "ES2003b",
    "ES2003c",
    "ES2003d",
    "ES2005a",
    "ES2005b",
    "ES2005c",
    "ES2005d",
    "ES2006a",
    "ES2006b",
    "ES2006c",
    "ES2006d",
    "ES2007a",
    "ES2007b",
    "ES2007c",
    "ES2007d",
    "ES2008a",
    "ES2008b",
    "ES2008c",
    "ES2008d",
    "ES2009a",
    "ES2009b",
    "ES2009c",
    "ES2009d",
    "ES2010a",
    "ES2010b",
    "ES2010c",
    "ES2010d",
    "ES2012a",
    "ES2012b",
    "ES2012c",
    "ES2012d",
    "ES2013a",
    "ES2013b",
    "ES2013c",
    "ES2013d",
    "ES2014a",
    "ES2014b",
    "ES2014c",
    "ES2014d",
    "ES2015a",
    "ES2015b",
    "ES2015c",
    "ES2015d",
    "ES2016a",
    "ES2016b",
    "ES2016c",
    "ES2016d",
    "IB4005",
    "IN1001",
    "IN1002",
    "IN1005",
    "IN1007",
    "IN1008",
    "IN1009",
    "IN1012",
    "IN1013",
    "IN1014",
    "IN1016",
    "IS1000a",
    "IS1000b",
    "IS1000c",
    "IS1000d",
    "IS1001a",
    "IS1001b",
    "IS1001c",
    "IS1001d",
    "IS1002b",
    "IS1002c",
    "IS1002d",
    "IS1003a",
    "IS1003b",
    "IS1003c",
    "IS1003d",
    "IS1004a",
    "IS1004b",
    "IS1004c",
    "IS1004d",
    "IS1005a",
    "IS1005b",
    "IS1005c",
    "IS1006a",
    "IS1006b",
    "IS1006c",
    "IS1006d",
    "IS1007a",
    "IS1007b",
    "IS1007c",
    "IS1007d",
    "TS3005a",
    "TS3005b",
    "TS3005c",
    "TS3005d",
    "TS3006a",
    "TS3006b",
    "TS3006c",
    "TS3006d",
    "TS3007a",
    "TS3007b",
    "TS3007c",
    "TS3007d",
    "TS3008a",
    "TS3008b",
    "TS3008c",
    "TS3008d",
    "TS3009a",
    "TS3009b",
    "TS3009c",
    "TS3009d",
    "TS3010a",
    "TS3010b",
    "TS3010c",
    "TS3010d",
    "TS3011a",
    "TS3011b",
    "TS3011c",
    "TS3011d",
    "TS3012a",
    "TS3012b",
    "TS3012c",
    "TS3012d",
]

_AMI_VALIDATION_SAMPLE_IDS = [
    "ES2011a",
    "ES2011c",
    "IB4001",
    "IB4003",
    "IB4010",
    "IS1008a",
    "IS1008c",
    "TS3004a",
    "TS3004c",
    "ES2011b",
    "ES2011d",
    "IB4002",
    "IB4004",
    "IB4011",
    "IS1008b",
    "IS1008d",
    "TS3004b",
    "TS3004d",
]

_AMI_EVAL_SAMPLE_IDS = [
    "EN2002a",
    "EN2002b",
    "EN2002c",
    "EN2002d",
    "ES2004a",
    "ES2004b",
    "ES2004c",
    "ES2004d",
    "IS1009a",
    "IS1009b",
    "IS1009c",
    "IS1009d",
    "TS3003a",
    "TS3003b",
    "TS3003c",
    "TS3003d",
]

_AMI_SAMPLE_IDS = {
    "train": _AMI_TRAIN_SAMPLE_IDS,
    "dev": _AMI_VALIDATION_SAMPLE_IDS,
    "eval": _AMI_EVAL_SAMPLE_IDS,
}

_AMI_BASE_DATA_URL = "https://huggingface.co/datasets/speech-seq2seq/ami/resolve/main/"

_AMI_AUDIO_ARCHIVE_URL = _AMI_BASE_DATA_URL + "audio/ihm/{split}/{_id}.tar.gz"

_AMI_ANNOTATIONS_ARCHIVE_URL = _AMI_BASE_DATA_URL + "annotations/{split}/text"

_SPGISPEECH_BASE_URL = "https://huggingface.co/datasets/kensho/spgispeech/resolve/main/data/"

_SPGISPEECH_AUDIO_BASE_URL = _SPGISPEECH_BASE_URL + "audio"

_SPGISPEECH_SUBSET_TO_DIR = {
    "s": ["s"],
    "m": ["s", "m_additional"],
    "l": ["s", "m_additional", "l_additional"],
    "dev": ["dev"],
    "test": ["test"],
}

# the second number in range is the number of archives (shards) in a subset
_SPGISPEECH_AUDIO_ARCHIVES_NAMES = {
    "s": [f"s_part_{i}.tar.gz" for i in range(0, 6)],
    "m_additional": [f"m_part_{i}.tar.gz" for i in range(0, 21)],
    "l_additional": [f"l_part_{i}.tar.gz" for i in range(0, 103)],
    "dev": [f"dev_part_{i}.tar.gz" for i in range(0, 3)],
    "test": [f"test_part_{i}.tar.gz" for i in range(0, 3)],
}

_SPGISPEECH_META_BASE_URL = _SPGISPEECH_BASE_URL + "meta"

_SPGISPEECH_META_FILENAMES = {
    "s": "train_small.csv",
    "m": "train_medium.csv",
    "l": "train.csv",
    "dev": "dev.csv",
    "test": "test.csv",
}

_VOXPOPULI_BASE_DATA_DIR = "https://huggingface.co/datasets/polinaeterna/voxpopuli/resolve/main/data/"

_VOXPOPULI_N_SHARDS_FILE = _VOXPOPULI_BASE_DATA_DIR + "n_files.json"

_VOXPOPULI_AUDIO_ARCHIVE_PATH = _VOXPOPULI_BASE_DATA_DIR + "en/{split}/{split}_part_{n_shard}.tar.gz"

_VOXPOPULI_METADATA_PATH = _VOXPOPULI_BASE_DATA_DIR + "en/asr_{split}.tsv"

_LIBRISPEECH_DL_URL = "http://www.openslr.org/resources/12/"

_LIBRISPEECH_DL_URLS = {
    "dev.clean": _LIBRISPEECH_DL_URL + "dev-clean.tar.gz",
    "dev.other": _LIBRISPEECH_DL_URL + "dev-other.tar.gz",
    "test.clean": _LIBRISPEECH_DL_URL + "test-clean.tar.gz",
    "test.other": _LIBRISPEECH_DL_URL + "test-other.tar.gz",
}

_COMMON_VOICE_API_URL = "https://commonvoice.mozilla.org/api/v1"

_COMMON_VOICE_BUNDLE_URL_TEMPLATE = 'cv-corpus-9.0-2022-04-27/cv-corpus-9.0-2022-04-27-{locale}.tar.gz'

_TEDLIUM_BASE_URL = "https://huggingface.co/datasets/LIUM/tedlium/resolve/main/TEDLIUM_release3/legacy/"

_TEDLIUM_URLS = {
    "train": [_TEDLIUM_BASE_URL + "train_1.tar.gz", _TEDLIUM_BASE_URL + "train_2.tar.gz"],
    "dev": [_TEDLIUM_BASE_URL + "dev.tar.gz"],
    "test": [_TEDLIUM_BASE_URL + "test.tar.gz"],
}

_GIGASPEECH_BASE_DATA_URL = "https://huggingface.co/datasets/speechcolab/gigaspeech/resolve/main/data/"

_GIGASPEECH_AUDIO_ARCHIVE_URL = _GIGASPEECH_BASE_DATA_URL + "audio/{subset}_files{is_additional}/{subset}_chunks_{archive_id:04}.tar.gz"

_GIGASPEECH_META_URL = _GIGASPEECH_BASE_DATA_URL + "metadata/{subset}_metadata{is_additional}/{subset}_chunks_{archive_id:04}_metadata.csv"

_GIGASPEECH_CONFIGS_TO_ALL_CONFIGS = {
    "xs": ["xs"],
    "s": ["xs", "s"],
    "m": ["xs", "s", "m"],
    "l": ["xs", "s", "m", "l"],
    "xl": ["xs", "s", "m", "l", "xl"],
}

_GIGASPEECH_N_ARCHIVES = {
    "xs": 1,
    "s": 23,
    "m": 69,
    "l": 136,
    "xl": 602,
    "dev": 1,
    "test": 3,
}

_EARNINGS_BASE_URL = "https://huggingface.co/datasets/anton-l/earnings22_baseline_5_gram/resolve/main/"

_EARNINGS_DEV_IDS = {
    "4420696",
    "4448760",
    "4461799",
    "4469836",
    "4473238",
    "4482110",
}
_EARNINGS_TEST_IDS = {
    "4432298",
    "4450488",
    "4470290",
    "4479741",
    "4483338",
    "4485244",
}


tedlium_contractions = [" 's", " 't", " 're", " 've", " 'm", " 'll", " 'd", " 'clock", " 'all"]
gigaspeech_punctuation = {" <comma>": ",", " <period>": ".", " <questionmark>": "?", " <exclamationpoint>": "!"}
gigaspeech_junk_tokens = ["<other>", "<sil>"]
swb_junk_tokens = ["[noise]", "[laughter]", "[silence]", "[vocalized-noise]", "<a_aside>", "<b_aside>", "<e_aside>",
                    "[laughter-", "_1", "[laugh]", "[sigh]", "[cough]", "[mn]", "[breath]", "[lipsmack]",
                    "[sneeze]", "[skip]", "[pause]", "(%hesitation)", "(%HESITATION)"]
swb_punctuations = ["{", "}", "[", "]-", "]", "((", "))", "(", ")", "."]
swb_fillers = r"\b(uh|uhm|um|hmm|mm|mhm|mmm)\b"
earnings_junk_tokens = ["<noise>", "<crosstalk>", "<affirmative>", "<inaudible>", "inaudible", "<laugh>", "<silence>"]
ignore_segments = ["ignore_time_segment_in_scoring", "<noise>", "<music>", "[noise]", "[laughter]", "[silence]",
                   "[vocalized-noise]", "<crosstalk>", "<affirmative>", "<inaudible>", "<laugh>", ""]
ignore_segments = ignore_segments + gigaspeech_junk_tokens + swb_junk_tokens + earnings_junk_tokens