import math
import os
import pickle
import time
from contextlib import nullcontext
from datetime import datetime

import librosa
import numpy as np
import torch
from einops import rearrange

import wandb
from codec import Codec, CodecConfig

# -----------------------------------------------------------------------------
# I/O
out_dir = "codec_" + datetime.now().strftime("%Y%m%d_%H%M%S")
eval_interval = 2000
log_interval = 100
eval_iters = 200
audio_sample_iters = 3

# wandb logging
wandb_log = True  # disabled by default
wandb_project = "vaclav-nanogpt-audio-codec"
wandb_run_name = out_dir

# data
dataset = "expresso"
batch_size = 64
block_size = 2 ** (14)  # 2**14 is roughly 1s of audio
sample_rate = 16000

# model
channels = 32
n_blocks = 7

# adamw optimizer
learning_rate = 3e-4  # Jukebox: 3e-4
max_iters = 100_000
weight_decay = 1e-1  # just mimicking train.py
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 100_000  # should be ~= max_iters per Chinchilla
min_lr = 3e-4  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# system
device = "cuda"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging

if out_dir == "out_default":
    raise ValueError("Please override out_dir")

os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
torch_dtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=torch_dtype)
)

data_dir = os.path.join("data", dataset)

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")
try:
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
except FileNotFoundError as f:
    raise FileNotFoundError(f"Meta file not found: {meta_path}") from f

meta_vocab_size = meta["vocab_size"]
meta_dtype = meta.get("dtype", "uint16")
print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")


# poor man's data loader
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    data = np.memmap(
        os.path.join(data_dir, "train.bin" if split == "train" else "val.bin"),
        dtype=meta_dtype,
        mode="r",
    )

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [
            torch.from_numpy(
                (
                    librosa.mu_expand(
                        data[i : i + block_size].astype(np.int32) - 128, mu=255
                    )
                )
            ).to(dtype=torch_dtype)
            for i in ix
        ]
    )
    x = rearrange(x, "b t -> b 1 t")
    if device_type == "cuda":
        # pin array x, which allows us to move it to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)

    return x


# model init
# start with model_args from command line
codec_config = CodecConfig(channels=channels, n_blocks=n_blocks)

model: Codec = Codec(codec_config)
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay=weight_decay, learning_rate=learning_rate, device_type=device_type
)

if compile:
    print("compiling the model...")
    unoptimized_model = model  # vv: Is this necessary?
    model = torch.compile(model)  # requires PyTorch 2.0


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    losses = {}
    audios = None
    model.eval()
    for split in ["train", "val"]:
        split_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x = get_batch(split)
            with ctx:
                reconstructed, loss = model(x)
            split_losses[k] = loss.item()

            if k == 0:
                audios = rearrange(
                    torch.concat([reconstructed, x], dim=-1), "b 1 t -> b t"
                )
                audios = audios[:audio_sample_iters, :]

        losses[split] = split_losses.mean()
    model.train()

    assert audios is not None
    return losses, audios


# learning rate decay scheduler (cosine with warmup)
# vv: Not sure the same makes sense for the codec too
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log:
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
x = get_batch("train")
t0 = time.time()

iter_num = 0
best_val_loss = 1e9


def create_spectrogram(audio: np.ndarray):
    import matplotlib.pyplot as plt

    assert audio.ndim == 1, f"Expected 1D array, got {audio.shape=}"

    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, ax=ax)
    fig.colorbar(img, ax=ax)
    return fig


while True:
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % eval_interval == 0:
        losses, audios = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        if wandb_log:
            wandb_audios = [
                {
                    "audio": wandb.Audio(
                        audio,
                        sample_rate=sample_rate,
                    ),
                    "spectrogram": create_spectrogram(audio),
                }
                for audio in audios.cpu().to(dtype=torch.float32).numpy()
            ]

            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "audio": [a["audio"] for a in wandb_audios],
                    "spectrogram": [wandb.Image(a["spectrogram"]) for a in wandb_audios],
                }
            )

    with ctx:
        reconstructed, loss = model(x)
    # immediately async prefetch next batch while model is doing the forward pass on the GPU
    x = get_batch("train")
    # backward pass, with gradient scaling if training in fp16
    scaler.scale(loss).backward()

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        lossf = loss.item()
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms")
    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
