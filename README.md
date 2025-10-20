
# nanoGPTaudio

Code for [Neural audio codecs: how to get audio into LLMs](https://kyutai.org/next/codec-explainer).
If you're looking for the code for the animations, you can find it [here](https://github.com/kyutai-labs/neural-audio-codecs-anims).

Disclaimer: This code is primarily published for curious readers who want to look at how exactly things were implemented,
but it will not work out of the box because of hardcoded paths.
I would accept PRs to make things more user-friendly, but I'm not planning to expand the features.

For more information, refer to the README of [the original nanoGPT](https://github.com/karpathy/nanoGPT).

## Installation

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

## Getting Libri-Light

First, you'll need the Libri-Light dataset.
Follow [this guide](https://github.com/facebookresearch/libri-light/blob/3fb5006a39e6f9e86daf3e5e52bc87630f3cdf3e/data_preparation/README.md) to download it.
You're going to want the segmented version, so also run `python cut_by_vad.py`.

Libri-Light is big: the train split contains 50k hours of audio, about 3 TB.
To train our codec, we only used 1000 hours.
Run this script to generate a list of files to be included in the dataset such that the length is 1000 hours:

```bash
python data/librilight/subsample.py path/to/your/librilight_segmented/train/ --target-hours 1000
```

Then to create a `.bin` file with the actual audio, shuffled and concatenated into one big lump, run:

```bash
python data/librilight/prepare.py data/librilight/librilight_1000h.jsonl --tokenizer mu-law-256 --batch-size 32
```

## Training a codec

Then, on a GPU machine, train the codec. I used a single H100 for this step.
The default parameters should be fine, so you can just run:

```bash
python train_codec.py
```

nanoGPT has a kind of crazy config system where variables are overridden using `exec`,
so if you want to override e.g. `dataset` to be `librilight/librilight_5000h_mu-law-256`,
you can use the flag `--dataset=librilight/librilight_5000h_mu-law-256`.

## Preparing a tokenized dataset

Now let's use the codec to tokenize Libri-Light more efficiently.
We're being a bit naughty because we're running the codec (partly) on the same data it was trained on.

We'll use 10k hours, so get a corresponding list of files:

```bash
python data/librilight/subsample.py path/to/your/librilight_segmented/train/ --target-hours 10000
```

This time, run `prepare.py` with our new codec:

```bash
python data/librilight/prepare.py data/librilight/librilight_10000h.jsonl --tokenizer codec_0123_012345 --batch-size 32
```

The tokenization can take a while, so you can use `--wandb-log` to track the progress.

You can also use `mimi_8_rvq` as the `--tokenizer` arg to tokenize with an 8-level Mimi.

## Training a language model

At this point, training is similar to the base nanoGPT model.

We run training on 8 H100 GPUs. In this setup, training takes about 4 days.
Assuming you have 8 GPUs available:

```bash
torchrun --standalone --nproc_per_node=8 train.py config/train_librilight_1e4h.py
```

This config trains on `librilight/librilight_10000h_mimi_8_rvq`, i.e. 10k hours from Libri-Light tokenized by a 8-level Mimi.

## Sampling

To sample from a trained model, use

```bash
python sample.py \
--out_dir=models/lm_librilight_1003_123345/ \
--max_new_tokens=4000 \
--num_samples=5 \
--temperature=0.8 \
--start=FILE:assets/prompt_librilight_uk_4s.wav
```

Here `--out-dir` should point to the directory of the model trained earlier.
The `--start` param is optional, and determines a prompt/prefix to start generating from.

Samples will be created under `{--out_dir}/samples/` and named sensibly.

## Audio sample sources

The poem prompt was extracted this way:

```bash
ffmpeg -i /lustre/scwpod02/client/kyutai/datasets/librilight_segmented/train/10038/10066/july_field_dcc_64kb_0000.flac -ss 00:00:09.6 -t 4s prompt_librilight_uk_4s.wav
```
