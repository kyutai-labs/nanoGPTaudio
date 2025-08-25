import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import tqdm

SAMPLE_RATE = 16000
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parents[1]

# add to pythonpath because we're not a proper Python package
sys.path.append(str(REPO_ROOT))

from tokenizer import Tokenizer, audio_tokenizer_from_name


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


def main(tokenizer: Tokenizer[np.ndarray]):
    out_dir = SCRIPT_DIR / str(tokenizer)

    def load_audio_file(file: Path) -> np.ndarray:
        audio, sr = librosa.load(file, mono=True)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        return tokenizer.encode(audio).cpu().numpy()

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

    create_meta(tokenizer)


def create_meta(tokenizer: Tokenizer):
    meta = {
        "sample_rate": SAMPLE_RATE,
        "modality": "audio",
        "tokenizer": str(tokenizer),
        "vocab_size": tokenizer.vocab_size(),
        "dtype": str(tokenizer.dtype()).replace("torch.", ""),
    }

    meta_file = SCRIPT_DIR / str(tokenizer) / "meta.json"
    meta_file.parent.mkdir(exist_ok=True)

    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="mu-law-256")
    args = parser.parse_args()

    tokenizer = audio_tokenizer_from_name(args.tokenizer)

    main(tokenizer)
