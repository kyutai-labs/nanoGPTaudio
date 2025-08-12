import os
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import tqdm

SAMPLE_RATE = 16000
SCRIPT_DIR = Path(__file__).parent


def get_file_list() -> list[Path]:
    base_dir = Path(
        "/lustre/scwpod02/client/kyutai/vaclav/datasets/expresso/audio_48khz/conversational/"
    )

    # Be sure to shuffle reproducibly
    files = sorted(list(base_dir.glob("**/*.wav")))
    rng = np.random.default_rng(37)
    rng.shuffle(files)
    return files


def load_audio_file(file: Path) -> np.ndarray:
    audio, sr = librosa.load(file, mono=True)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    audio_mu_law = (librosa.mu_compress(audio, mu=255) + 128).astype(np.uint8)
    return audio_mu_law


def main():
    files = get_file_list()
    print(f"Found {len(files)} audio files.")

    train_fraction = 0.9
    splits = {
        "train": files[: int(len(files) * train_fraction)],
        "val": files[int(len(files) * train_fraction) :],
    }

    for split, file_list in splits.items():
        print(f"Processing {split} split with {len(file_list)} files.")
        with Pool(processes=os.cpu_count()) as pool:
            audio_data = list(
                tqdm.tqdm(
                    pool.imap(load_audio_file, file_list),
                    total=len(file_list),
                    desc=f"Loading {split} files",
                )
            )
        audio_data = np.concatenate(audio_data, axis=0)
        output_file = SCRIPT_DIR / Path(f"{split}.bin")
        audio_data.tofile(output_file)
        print(f"Saved {split} data to {output_file}")


def create_meta():
    meta = {
        "vocab_size": 256,  # mu-law encoding uses 256 values
        "sample_rate": SAMPLE_RATE,
        "modality": "audio",
        "encoding": "mu-law-256",
        "dtype": "uint8",
    }
    meta_file = SCRIPT_DIR / "meta.pkl"
    with open(meta_file, "wb") as f:
        import pickle

        pickle.dump(meta, f)
    print(f"Saved metadata to {meta_file}")


if __name__ == "__main__":
    main()
    create_meta()
