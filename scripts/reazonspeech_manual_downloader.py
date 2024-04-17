"""Manual script to download reazonspeech dataset locally
- files are saved at your cache folder (`~/.cache/reazon_manual_download/{target_size}`)
- dataset https://huggingface.co/datasets/reazon-research/reazonspeech
"""
import argparse
import os
import shutil
import urllib.request
from time import sleep
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm
import tarfile

# https://stackoverflow.com/questions/71692354/facing-ssl-error-with-huggingface-pretrained-models
os.environ['CURL_CA_BUNDLE'] = ''

# disable warning message
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

BASE_URL = "https://reazonspeech.s3.abci.ai/"
DATASET = {
    "tiny": {"tsv": 'v2-tsv/tiny.tsv', "audio": "v2/{:03x}.tar", "nfiles": 1},
    "small": {"tsv": 'v2-tsv/small.tsv', "audio": "v2/{:03x}.tar", "nfiles": 12},
    "medium": {"tsv": 'v2-tsv/medium.tsv', "audio": "v2/{:03x}.tar", "nfiles": 116},
    "large": {"tsv": 'v2-tsv/large.tsv', "audio": "v2/{:03x}.tar", "nfiles": 579},
    "all": {"tsv": 'v2-tsv/all.tsv', "audio": "v2/{:03x}.tar", "nfiles": 4095}  # test set is in the 4096th file.
}


def dl(url, target_file):
    try:
        urllib.request.urlretrieve(url, target_file)
        return True
    except Exception as e:
        print(f"network error during downloading {target_file}")
        if os.path.exists(target_file):
            os.remove(target_file)
        return False


def get_broken_files(target_files):
    broken_files = []
    for i in tqdm(target_files):
        if not i.endswith(".tar"):
            continue
        if not os.path.exists(i):
            print(f"file not exist: {i}")
            broken_files.append(i)
            continue
        try:
            with tarfile.open(i) as t:
                t.extractall(path="tmp_data_dir")
        except tarfile.ReadError:
            print(f"broken file found: {i}")
            broken_files.append(i)
    print(f"{len(broken_files)} broken files found.")
    if os.path.exists("tmp_data_dir"):
        shutil.rmtree("tmp_data_dir", ignore_errors=True)
    return broken_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download ReazonSpeech locally.')
    parser.add_argument('-t', '--target', default="tiny", help="tiny/small/medium/large/all", type=str)
    parser.add_argument('-p', '--pool', default=10, type=int)
    parser.add_argument('-s', '--start_que', default=None, type=int)
    parser.add_argument('-e', '--end_que', default=None, type=int)
    parser.add_argument('--health_check', action="store_true")
    arg = parser.parse_args()
    target_dir = f"{os.path.expanduser('~')}/.cache/reazon_manual_download/{arg.target}"
    if arg.health_check:
        print("health check mode")
        target_broken_files = get_broken_files(glob(f"{target_dir}/*.tar"))
        print(f"{len(target_broken_files)} missing/broken tar file")
        target_broken_files = [i for i in target_broken_files if os.path.exists(i)]
        print(f"remove {len(target_broken_files)} broken tar files")
        for i in target_broken_files:
            os.remove(i)
        exit()

    # all urls to download
    files = list(range(DATASET[arg.target]["nfiles"]))
    if arg.start_que is not None:
        assert arg.end_que is not None
        files = files[arg.start_que:arg.end_que]
        target_dir = f"{target_dir}_{arg.start_que}_{arg.end_que}"

    os.makedirs(target_dir, exist_ok=True)
    urls = [BASE_URL + DATASET[arg.target]["audio"].format(idx) for idx in files]
    urls.append(BASE_URL + DATASET[arg.target]["tsv"])
    urls = [i for i in urls if not os.path.exists(f"{target_dir}/{arg.target}.{os.path.basename(i)}")]
    filenames = [f"{target_dir}/{arg.target}.{os.path.basename(i)}" for i in urls]
    print(f"Total files to download: {len(filenames)}")

    counter = 0
    while True:
        if counter == 10:
            print("failed too many times. sleep for a while.")
            counter = 0
            sleep(120)
        # start downloader
        print(f"Worker: {arg.pool}")
        if arg.pool == 1:
            for _url, _file in tqdm(zip(urls, filenames), total=len(filenames)):
                dl(_url, _file)
        else:
            with Pool(arg.pool) as pool:
                pool.starmap(dl, tqdm(zip(urls, filenames), total=len(filenames)))
        print("download complete")

        # check the tar files
        print("tar file health check")
        filenames = get_broken_files(filenames)
        if len(filenames) == 0:
            break
        for i in filenames:
            if os.path.exists(i):
                os.remove(i)
        print(f"retry downloading {len(filenames)} files")
        counter += 1
