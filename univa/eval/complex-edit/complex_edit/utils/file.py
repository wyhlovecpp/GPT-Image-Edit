from tqdm import tqdm
import json


def read_jsonl(path):
    lst = []
    with open(path, mode="r") as f:
        for line in tqdm(f.readlines()):
            lst.append(json.loads(line))

    return lst


def dump_jsonl(lst, path):
    with open(path, mode="w") as f:
        for sample in lst:
            f.write(json.dumps(sample) + "\n")
