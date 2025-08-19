from pathlib import Path
from typing import Generic, Protocol, TypeVar

import librosa
import numpy as np
import torch
from einops import rearrange

from codec import Codec

T = TypeVar("T")


class Tokenizer(Protocol, Generic[T]):
    """Something that can encode data into tokens and then back.

    The type that we're encoding will be str for text and np.ndarray for audio.
    """

    def encode(self, raw: T) -> torch.Tensor:
        """Take type T and encode it into a 1D int tensor of tokens."""
        ...

    def decode(self, tokens: torch.Tensor) -> T:
        """Take a 1D int tensor of tokens and decode it into type T."""
        ...

    def vocab_size(self) -> int:
        """How many possible tokens are there?"""
        ...

    def dtype(self) -> torch.dtype:
        """What is the dtype of the tokens?"""
        ...


class CharTokenizer(Tokenizer[str]):
    def __init__(self, meta: dict):
        self.stoi = meta["stoi"]
        self.itos = meta["itos"]
        assert len(self.stoi) == len(self.itos)

    def encode(self, raw: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in raw], dtype=torch.int32)

    def decode(self, tokens: torch.Tensor) -> str:
        return "".join([self.itos[str(i)] for i in tokens.cpu().numpy()])

    def vocab_size(self):
        return len(self.stoi)

    def dtype(self):
        return torch.int32

    def __str__(self):
        return "char"


class TiktokenTokenizer(Tokenizer[str]):
    def __init__(self, encoding_name: str):
        import tiktoken

        self.encoding_name = encoding_name
        self.encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, raw: str) -> torch.Tensor:
        return torch.tensor(
            self.encoding.encode(raw, allowed_special={"<|endoftext|>"}),
            dtype=self.dtype(),
        )

    def decode(self, tokens: torch.Tensor) -> str:
        return self.encoding.decode(tokens.cpu().numpy().tolist())

    def vocab_size(self):
        return self.encoding.max_token_value + 1

    def dtype(self):
        return torch.int32

    def __str__(self):
        return f"tiktoken-{self.encoding_name}"


class MuLawTokenizer(Tokenizer[np.ndarray]):
    def encode(self, raw: np.ndarray) -> torch.Tensor:
        return torch.tensor(librosa.mu_compress(raw, mu=255) + 128).to(torch.uint8)

    def decode(self, tokens: torch.Tensor) -> np.ndarray:
        return librosa.mu_expand(
            tokens.to("cpu", dtype=torch.int32).numpy() - 128, mu=255
        )

    def vocab_size(self):
        return 256

    def dtype(self):
        # TODO: use int8 instead, we'd save ourselves the -128 and +128 reshuffling
        return torch.uint8

    def name(self):
        return "mu-law-256"


class CodecTokenizer(Tokenizer[np.ndarray]):
    def __init__(self, name: str, device: str = "cuda"):
        self.name = name
        self.codec = Codec.from_checkpoint(name, device=device)
        self.device = device

    def encode(self, raw: np.ndarray) -> torch.Tensor:
        audio = torch.Tensor(raw).to(self.device)
        factor = self.codec.encoder.downscaling_factor()
        audio = audio[None, None, : len(audio) // factor * factor]

        with torch.no_grad():
            codes, _reconstructed, _losses = self.codec(audio)

        # To avoid having to model multiple streams, flatten the levels of the RVQ
        # into one
        flat_codes = rearrange(codes, "1 n_codes t -> (t n_codes)")
        return flat_codes

    def decode(self, tokens: torch.Tensor) -> np.ndarray:
        codes = rearrange(
            tokens,
            "(t n_codebooks) -> n_codebooks t",
            n_codebooks=4,
        )
        decoded = self.codec.decode(codes[None, :, :])[0, 0]
        return decoded.to("cpu", dtype=self.dtype()).numpy()

    def vocab_size(self):
        return self.codec.config.codebook_size

    def dtype(self):
        return torch.int32

    def __str__(self):
        return self.name
