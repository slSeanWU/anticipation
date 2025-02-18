"""To make sure we know which GigaMIDI files are already in LMD dataset, we need to check for md5 collisions."""

import glob
import json
import os

import tqdm
import hashlib

GIGAMIDI_RAW_DATA_DIR = "/scratch/shared/datasets/gigamidi/raw_huggingface/"
LMD_RAW_DATA_DIR = "/scratch/shared/datasets/lmd-full/"


def compute_md5_from_midi(midi_path):
    return hashlib.md5(open(midi_path, "rb").read()).hexdigest()


def check_gigamidi_md5_consistency(metadata_path, midi_path):
    """ensure we're computing md5 of midi in correct way

    `metadata_path` contains metadata copied from Huggingface `Metacreation/GigaMIDI` dataset
    """
    md5_meta = json.load(open(metadata_path))["md5"]

    # compute md5 of midi file
    md5_midi = compute_md5_from_midi(midi_path)

    return md5_meta == md5_midi


def make_lmd_md5_lookup(lmd_root):
    lmd_md5_dict = {}

    for midi_path in tqdm.tqdm(
        glob.glob(f"{lmd_root}/**/*.mid", recursive=True), desc="build LMD md5 lookup"
    ):
        md5 = compute_md5_from_midi(midi_path)
        lmd_md5_dict[md5] = os.path.basename(midi_path)

    return lmd_md5_dict


def check_md5_collision(lmd_lookup: dict[str, str], midis_to_check: list[str]):
    collisions = []

    colliding_md5s = set(lmd_lookup.keys())

    for midi_path in tqdm.tqdm(midis_to_check, desc="check md5 collision"):
        md5 = compute_md5_from_midi(midi_path)

        if md5 in colliding_md5s:
            collisions.append(
                {
                    "lmd_path": lmd_lookup[md5],
                    "gigamidi_path": midi_path.split(GIGAMIDI_RAW_DATA_DIR)[-1],
                    "md5": md5,
                }
            )
            print(collisions[-1])

    return collisions


if __name__ == "__main__":
    midi_files = glob.glob(f"{GIGAMIDI_RAW_DATA_DIR}/**/*.mid", recursive=True)
    print("found", len(midi_files), "midi files in GigaMIDI dataset")

    lmd_midi_files = glob.glob(f"{LMD_RAW_DATA_DIR}/**/*.mid", recursive=True)
    print("found", len(lmd_midi_files), "midi files in LMD dataset")

    lmd_md5_lookup = make_lmd_md5_lookup(LMD_RAW_DATA_DIR)
    print("built LMD md5 lookup with", len(lmd_md5_lookup), "entries")

    collisions = check_md5_collision(lmd_md5_lookup, midi_files)

    print(
        "found",
        len(collisions),
        "collisions out of total",
        len(midi_files),
        "midi files",
    )

    collision_output_path = (
        os.path.dirname(GIGAMIDI_RAW_DATA_DIR) + "/gigamidi_lmd_md5_collisions.json"
    )

    with open(collision_output_path, "w") as f:
        json.dump(collisions, f, indent=4)
