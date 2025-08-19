import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import torch
import tqdm
from einops import rearrange

SAMPLE_RATE = 16000
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parents[1]

# add to pythonpath because we're not a proper Python package
sys.path.append(str(REPO_ROOT))

from codec import Codec


def get_codec_checkpoint(codec_name: str):
    return SCRIPT_DIR.parents[1] / "models" / codec_name / "codec_ckpt.pt"


def get_file_list() -> list[Path]:
    base_dir = Path(
        "/lustre/scwpod02/client/kyutai/vaclav/datasets/expresso/audio_48khz/conversational/"
    )

    # Be sure to shuffle reproducibly
    files = sorted(list(base_dir.glob("**/*.wav")))
    rng = np.random.default_rng(37)
    rng.shuffle(files)
    return files


class AudioTooShort(Exception):
    """The audio is too short to process."""


def main(encoding: str):
    if encoding.startswith("codec"):
        codec = Codec.from_checkpoint(get_codec_checkpoint(encoding))
        vocab_size = codec.config.codebook_size
    else:
        assert encoding == "mu-law-256"
        vocab_size = 256

    out_dir = SCRIPT_DIR / encoding

    def load_audio_file(file: Path) -> np.ndarray:
        audio, sr = librosa.load(file, mono=True)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        if encoding == "mu-law-256":
            audio_mu_law = (librosa.mu_compress(audio, mu=255) + 128).astype(np.uint8)
            return audio_mu_law
        else:
            assert encoding.startswith("codec")
            audio = torch.Tensor(audio)
            factor = codec.encoder.downscaling_factor()

            if len(audio) < factor:
                raise AudioTooShort

            audio = audio[None, None, : len(audio) // factor * factor]

            codes, _reconstructed, _losses = codec(audio)

            # To avoid having to model multiple streams, flatten the levels of the RVQ
            # into one
            flat_codes = rearrange(codes, "1 n_codes t -> (t n_codes)")
            # We might fit into int16 but it could lead to annoying bugs
            assert flat_codes.dtype == torch.int32

            return flat_codes

    files = get_file_list()
    print(f"Found {len(files)} audio files.")

    train_fraction = 0.9
    splits = {
        "train": files[: int(len(files) * train_fraction)],
        "val": files[int(len(files) * train_fraction) :],
    }

    for split, file_list in splits.items():
        print(f"Processing {split} split with {len(file_list)} files.")
        # TODO: get multiprocessing working with the model, or use batching
        # with Pool(processes=os.cpu_count()) as pool:
        #     audio_data = list(
        #         tqdm.tqdm(
        #             pool.imap(load_audio_file, file_list),
        #             total=len(file_list),
        #             desc=f"Loading {split} files",
        #         )
        #     )
        audio_data = []
        for file in tqdm.tqdm(file_list, desc=f"Loading {split} files"):
            print(file)
            try:
                audio_data.append(load_audio_file(file))
            except AudioTooShort:
                print(f"Skipping {file} because it is too short.")
                continue

        audio_data = np.concatenate(audio_data, axis=0)

        out_dir.mkdir(exist_ok=True)
        output_file = out_dir / Path(f"{split}.bin")
        audio_data.tofile(output_file)
        print(f"Saved {split} data to {output_file}")

    create_meta(encoding=encoding, vocab_size=vocab_size, dtype=audio_data.dtype)


def create_meta(encoding: str, vocab_size: int, dtype: np.dtype):
    meta = {
        "vocab_size": vocab_size,
        "sample_rate": SAMPLE_RATE,
        "modality": "audio",
        "encoding": encoding,
        "dtype": str(dtype),
    }

    meta_file = SCRIPT_DIR / encoding / "meta.json"
    meta_file.parent.mkdir(exist_ok=True)

    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding", default="mu-law-256")
    args = parser.parse_args()

    assert args.encoding == "mu-law-256" or args.encoding.startswith("codec")

    main(args.encoding)
