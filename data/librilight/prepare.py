import argparse
import itertools
import json
import os
import sys
from dataclasses import dataclass
from math import ceil
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import torch
import tqdm

SAMPLE_RATE = 16000
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parents[1]
# add to pythonpath because we're not a proper Python package
sys.path.append(str(REPO_ROOT))

MAX_LENGTH_SEC = 60

import time

from tokenizer import CodecTokenizer, MuLawTokenizer, Tokenizer


@dataclass
class DatasetFile:
    path: Path
    duration: float | None = None
    sample_rate: int | None = None
    size_bytes: int | None = None


class AudioTooShort(Exception):
    """The audio is too short to process."""


def load_audio_file_raw(dataset_file: DatasetFile) -> np.ndarray:
    # print(f"Loading {Path(dataset_file.path).name}")
    audio, sr = librosa.load(dataset_file.path, mono=True)

    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def get_audio_chunks(dataset_files: list[DatasetFile]) -> Iterable[np.ndarray]:
    n_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", f"{os.cpu_count()}"))

    with Pool(processes=n_cpus) as pool:
        for loaded_file in pool.imap(load_audio_file_raw, dataset_files, chunksize=1):
            for i in range(0, len(loaded_file), SAMPLE_RATE * MAX_LENGTH_SEC):
                yield loaded_file[i : i + SAMPLE_RATE * MAX_LENGTH_SEC]


def main(egs_file: Path, tokenizer: Tokenizer[np.ndarray], batch_size: int):
    out_dir = SCRIPT_DIR / f"{egs_file.stem}_{tokenizer}"

    dataset_files = []
    with open(egs_file, "r", encoding="utf-8") as f:
        for line in f:
            dataset_files.append(DatasetFile(**json.loads(line)))

    dataset_files = dataset_files[::-1]

    train_fraction = 0.99
    splits = {
        "train": dataset_files[: int(len(dataset_files) * train_fraction)],
        "val": dataset_files[int(len(dataset_files) * train_fraction) :],
    }

    for split, file_list in splits.items():
        print(f"Processing {split} split with {len(file_list)} files.")
        t_split_start = time.time()

        audio_data = []

        # Account for the fact that long files get split up
        n_chunks = sum(ceil(file.duration / MAX_LENGTH_SEC) for file in file_list)

        t_last = None
        for audio_batch in tqdm.tqdm(
            itertools.batched(get_audio_chunks(file_list), batch_size),
            total=ceil(n_chunks / batch_size),
            desc=f"Creating {split} split",
        ):
            t2 = time.time()
            if t_last is not None:
                time_waited = t2 - t_last
                if time_waited > 0.1:
                    print(
                        f"Warning: waited {time_waited:.2f} seconds for next batch, "
                        "I/O might be the bottleneck"
                    )

            with torch.no_grad():
                encoded_list = tokenizer.encode_list(list(audio_batch))

            audio_data += [x.cpu().numpy() for x in encoded_list]
            t_last = time.time()

        audio_data = np.concatenate(audio_data, axis=0)

        out_dir.mkdir(exist_ok=True)
        output_file = out_dir / Path(f"{split}.bin")
        audio_data.tofile(output_file)
        print(f"Saved {split} data to {output_file}")
        print(f"Creating {split} took {time.time() - t_split_start:.2f}s")

    create_meta(tokenizer, out_dir)


def create_meta(tokenizer: Tokenizer, out_dir: Path):
    meta = {
        "sample_rate": SAMPLE_RATE,
        "modality": "audio",
        "tokenizer": str(tokenizer),
        "vocab_size": tokenizer.vocab_size(),
        "dtype": str(tokenizer.dtype()).replace("torch.", ""),
    }

    meta_file = out_dir / "meta.json"
    meta_file.parent.mkdir(exist_ok=True)

    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "egs_file",
        help='The "egs" file is a .jsonl with one dataset file ("egs-ample") per line',
        type=Path,
    )
    parser.add_argument("--tokenizer", default="mu-law-256")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    assert args.tokenizer == "mu-law-256" or args.tokenizer.startswith("codec")

    if args.tokenizer.startswith("codec"):
        tokenizer = CodecTokenizer(args.tokenizer)
    else:
        tokenizer = MuLawTokenizer()

    main(args.egs_file, tokenizer, batch_size=args.batch_size)
