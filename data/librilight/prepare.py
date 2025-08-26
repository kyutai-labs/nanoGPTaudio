import argparse
import itertools
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import torch
import tqdm

import wandb

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parents[1]
# add to pythonpath because we're not a proper Python package
sys.path.append(str(REPO_ROOT))

MAX_LENGTH_SEC = 60

import time

from tokenizer import Tokenizer, audio_tokenizer_from_name


@dataclass
class DatasetFile:
    path: Path
    duration: float | None = None
    sample_rate: int | None = None
    size_bytes: int | None = None


class AudioTooShort(Exception):
    """The audio is too short to process."""


def load_audio_file_raw(args: tuple[DatasetFile, int]) -> np.ndarray:
    dataset_file, sample_rate = args
    # print(f"Loading {Path(dataset_file.path).name}")
    audio, sr = librosa.load(dataset_file.path, mono=True)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    return audio


def get_audio_chunks(
    dataset_files: list[DatasetFile], sample_rate: int
) -> Iterable[np.ndarray]:
    n_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", f"{os.cpu_count()}"))

    with Pool(processes=n_cpus) as pool:
        args_iter = ((df, sample_rate) for df in dataset_files)
        for loaded_file in pool.imap(load_audio_file_raw, args_iter, chunksize=16):
            for i in range(0, len(loaded_file), sample_rate * MAX_LENGTH_SEC):
                yield loaded_file[i : i + sample_rate * MAX_LENGTH_SEC]


def ask_if_overwriting(output_file: Path, min_size_gb: int):
    if not output_file.exists():
        return

    file_size_gb = output_file.stat().st_size
    if file_size_gb < min_size_gb:
        return

    if (
        input(
            f"Output file {output_file} already exists and has "
            f"{file_size_gb:.2f}GB, overwrite?"
        ).lower()
        != "y"
    ):
        print("Exiting.")
        exit(1)


def main(
    egs_file: Path, tokenizer: Tokenizer[np.ndarray], batch_size: int, wandb_log: bool
):
    out_dir = SCRIPT_DIR / f"{egs_file.stem}_{tokenizer}"
    train_fraction = 0.99

    wandb_run_name = out_dir.name + "_" + datetime.now().strftime("%m%d_%H%M%S")
    wandb.init(
        project="vaclav-nanogpt-audio-dataset",
        name=wandb_run_name,
        config={
            "egs_file": str(egs_file),
            "tokenizer": str(tokenizer),
            "out_dir": str(out_dir),
            "batch_size": batch_size,
            "train_fraction": train_fraction,
        },
        mode=None if wandb_log else "disabled",  # None is the default, will use wandb
    )

    dataset_files = []
    with open(egs_file, "r", encoding="utf-8") as f:
        for line in f:
            dataset_files.append(DatasetFile(**json.loads(line)))

    dataset_files = dataset_files[::-1]

    splits = {
        "train": dataset_files[: int(len(dataset_files) * train_fraction)],
        "val": dataset_files[int(len(dataset_files) * train_fraction) :],
    }

    for split, file_list in splits.items():
        print(f"Processing {split} split with {len(file_list)} files.")
        t_split_start = time.time()

        # Account for the fact that long files get split up
        n_chunks = sum(ceil(file.duration / MAX_LENGTH_SEC) for file in file_list)

        out_dir.mkdir(exist_ok=True)
        output_file = out_dir / Path(f"{split}.bin")

        if split == "train":
            ask_if_overwriting(output_file, min_size_gb=1)

        # Stream the outputs to the file as we go to avoid keeping everything in memory
        with output_file.open("wb") as f:
            t_last_step_end = None
            sample_rate = tokenizer.sample_rate()
            total_duration_h = 0
            total_tokens = 0

            for audio_batch in tqdm.tqdm(
                itertools.batched(get_audio_chunks(file_list, sample_rate), batch_size),
                total=ceil(n_chunks / batch_size),
                desc=f"Creating {split} split",
            ):
                t_step_start = time.time()
                if t_last_step_end is not None:
                    time_waited = t_step_start - t_last_step_end
                    if time_waited > 0.1:
                        print(
                            f"Warning: waited {time_waited:.2f} seconds for next batch, "
                            "I/O might be the bottleneck"
                        )

                t_batch_start = time.time()
                with torch.no_grad():
                    encoded_list = tokenizer.encode_list(list(audio_batch))
                t_tokenization_end = time.time()

                for x in encoded_list:
                    f.write(x.cpu().numpy().tobytes())
                t_write_end = time.time()

                total_duration_h += (
                    sum(len(a) for a in audio_batch) / tokenizer.sample_rate() / 3600
                )
                total_tokens += sum(len(tokens) for tokens in encoded_list)

                wandb.log(
                    {
                        f"{split}/output_file_size_mb": output_file.stat().st_size
                        / (1024**2),
                        f"{split}/total_duration_h": total_duration_h,
                        f"{split}/total_tokens": total_tokens,
                        f"{split}/tokenization_time_s": t_tokenization_end
                        - t_batch_start,
                        f"{split}/write_time_s": t_write_end - t_tokenization_end,
                        f"{split}/batch_load_time_s": t_step_start
                        - (t_last_step_end or t_step_start),
                        f"{split}/total_step_time_s": t_write_end - t_step_start,
                    }
                )

                t_last_step_end = time.time()

        print(f"Saved {split} data to {output_file}")
        print(f"Creating {split} took {time.time() - t_split_start:.2f}s")

    create_meta(tokenizer, out_dir)


def create_meta(tokenizer: Tokenizer, out_dir: Path):
    meta = {
        "sample_rate": tokenizer.sample_rate(),
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
    parser.add_argument("--wandb-log", action="store_true")
    args = parser.parse_args()

    tokenizer = audio_tokenizer_from_name(args.tokenizer)

    main(args.egs_file, tokenizer, batch_size=args.batch_size, wandb_log=args.wandb_log)
