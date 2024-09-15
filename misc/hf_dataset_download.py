import argparse
from datasets import load_dataset

parser = argparse.ArgumentParser(description='Preprocessing dataset.')
parser.add_argument('-d', '--dataset', type=str)
parser.add_argument('-c', '--config', type=str)
parser.add_argument('-n', '--num', type=int, default=1)
arg = parser.parse_args()
[
    load_dataset(arg.dataset, n, trust_remote_code=True, num_proc=arg.num)
    for n in arg.config.split(",")
]




