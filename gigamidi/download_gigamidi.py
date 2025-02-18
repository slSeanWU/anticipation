import json
import os

import tqdm
import datasets
import symusic
import numpy as np

CACHE_DIR = "/home/slseanwu/.hf_cache"
OUT_DIR = "/scratch/shared/datasets/gigamidi/raw_huggingface"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float16):
            return float(obj)
        return super().default(obj)


def check_valid(midi_bytes):
    try:
        symusic.Score.from_midi(midi_bytes)
        return True
    except:
        return False


def save_midi_and_metadata(example, split=None, use_prefix=True):
    if use_prefix:
        prefix = example["md5"][:2]
    else:
        prefix = ""

    if split is None:
        example_dir = os.path.join(OUT_DIR, prefix, example["md5"])
    else:
        example_dir = os.path.join(OUT_DIR, split, prefix, example["md5"])

    os.makedirs(example_dir, exist_ok=True)

    midi_bytes = example["music"]
    with open(os.path.join(example_dir, "music.mid"), "wb") as f:
        f.write(midi_bytes)

    with open(os.path.join(example_dir, "metadata.json"), "w") as f:
        # save all fields other than `music`
        metadata = {k: v for k, v in example.items() if k != "music"}

        # format with 4 indent spaces
        f.write(json.dumps(metadata, indent=4, cls=NumpyEncoder))


if __name__ == "__main__":
    ds = datasets.load_dataset("Metacreation/GigaMIDI", cache_dir=CACHE_DIR)

    ds_train, ds_valid, ds_test = ds["train"], ds["validation"], ds["test"]
    print(len(ds_train), len(ds_valid), len(ds_test))

    all_valid = 0

    # iterate over training set
    for i, example in tqdm.tqdm(
        enumerate(ds_train), total=len(ds_train), desc="Processing training set"
    ):
        midi_bytes = example["music"]
        valid = check_valid(midi_bytes)

        all_valid += valid

        if valid:
            save_midi_and_metadata(example, split="train")

    print("[train] got", all_valid, "valid examples out of", len(ds_train))

    all_valid = 0

    # iterate over valid set
    for i, example in tqdm.tqdm(
        enumerate(ds_valid), total=len(ds_valid), desc="Processing validation set"
    ):
        midi_bytes = example["music"]
        valid = check_valid(midi_bytes)

        all_valid += valid

        if valid:
            save_midi_and_metadata(example, split="valid")

    print("[valid] got", all_valid, "valid examples out of", len(ds_valid))

    all_valid = 0

    # iterate over test set
    for i, example in tqdm.tqdm(
        enumerate(ds_test), total=len(ds_test), desc="Processing test set"
    ):
        midi_bytes = example["music"]
        valid = check_valid(midi_bytes)

        all_valid += valid

        if valid:
            save_midi_and_metadata(example, split="test")

    print("[test] got", all_valid, "valid examples out of", len(ds_test))
