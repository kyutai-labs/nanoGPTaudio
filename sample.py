"""
Sample from a trained model
"""

import json
import os
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import soundfile as sf
import torch

from model import GPT, GPTConfig
from tokenizer import (
    CharTokenizer,
    CodecTokenizer,
    MuLawTokenizer,
    TiktokenTokenizer,
    Tokenizer,
)

# -----------------------------------------------------------------------------
init_from = (
    "resume"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
)
out_dir = "out"  # ignored if init_from is not 'resume'
start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10  # number of samples to draw
max_new_tokens = 500  # number of tokens generated in each sample
temperature = (
    0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
)
top_k = (
    200  # retain only the top_k most likely tokens, clamp others to have 0 probability
)
seed = 1337
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
exec(open("configurator.py").read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]

model: GPT

# model
if init_from == "resume":
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith("gpt2"):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
else:
    raise ValueError(
        f"Unknown init_from value: {init_from}. Expected 'resume' or 'gpt2-<size>'."
    )

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)


meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.json")
print(f"Loading meta from {meta_path}...")
with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

modality: Literal["text", "audio"] = meta.get("modality", "text")
tokenizer: Tokenizer

if modality == "text":
    tokenizer = CharTokenizer(meta)
    # TODO: when to load GPT-2 tokenizer?
elif modality == "audio":
    if meta["encoding"] == "mu-law-256":
        tokenizer = MuLawTokenizer()
    else:
        tokenizer = CodecTokenizer(meta, device=device)
else:
    raise ValueError(f"Unknown modality: {modality}. Expected 'text' or 'audio'.")

# encode the beginning of the prompt
if start.startswith("FILE:"):
    if modality == "text":
        with open(start[len("FILE:") :], "r", encoding="utf-8") as f:
            start = f.read()
    elif modality == "audio":
        # Read audio file using librosa
        start, _ = librosa.load(start[len("FILE:") :], sr=meta["sample_rate"])
else:
    if start == "\n":
        start = np.array([0.0])

start_ids = tokenizer.encode(start)
assert start_ids.ndim == 1, f"Expected 1D result from encode(), got {start_ids.shape=}"
x = torch.tensor(start_ids, dtype=torch.long, device=device)
x = x.repeat(num_samples, 1)

samples_dir = Path(out_dir) / "samples"
file_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

# run generation
with torch.no_grad():
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )
    with ctx:
        y = model.generate(
            x, max_new_tokens, temperature=temperature, top_k=top_k, progress_bar=True
        )
        samples_dir.mkdir(parents=True, exist_ok=True)
        for k in range(num_samples):
            if modality == "text":
                print(tokenizer.decode(y[k].tolist()))
                print("---------------")
            elif modality == "audio":
                audio = tokenizer.decode(y[k])

                output_path = samples_dir / f"{file_prefix}_{k}.wav"
                sf.write(output_path, audio, meta["sample_rate"])
                print(f"Generated {output_path}")
