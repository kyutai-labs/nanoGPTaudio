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

    def encode_list(self, raw: list[T]) -> list[torch.Tensor]:
        """Encode a list of Ts of uneven lengths.

        Override for more efficient implementations.
        """
        return [self.encode(x) for x in raw]

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

        is_batched = audio.ndim == 2
        if not is_batched:
            audio = audio[None, :]

        # Make sure the length is divisible
        factor = self.codec.encoder.downscaling_factor()
        audio = audio[..., : audio.shape[-1] // factor * factor]

        audio = rearrange(audio, "b t -> b 1 t")

        with torch.no_grad():
            codes, _reconstructed, _losses = self.codec(audio)

        # To avoid having to model multiple streams, flatten the levels of the RVQ
        # into one
        flat_codes = rearrange(codes, "b n_codebooks t -> b (t n_codebooks)")

        if not is_batched:
            assert flat_codes.shape[0] == 1
            flat_codes = flat_codes[0]

        return flat_codes

    def encode_list(self, raw: list[np.ndarray]):
        for i in range(len(raw)):
            assert raw[i].ndim == 1, (
                f"Expected 1D array at index {i}, got {raw[i].ndim}D array"
            )

        max_len = max(a.shape[0] for a in raw)
        padded_audio = np.stack(
            [np.pad(a, (0, max_len - a.shape[0]), mode="constant") for a in raw]
        )  # shape [b, t]

        factor = self.codec.encoder.downscaling_factor()
        encoded = self.encode(padded_audio)  # shape [b, t // factor * n_codebooks]

        # Split encoded back into list with correct sizes
        encoded_list = [
            encoded[i, : len(raw[i]) // factor * self.n_codebooks()]
            for i in range(len(raw))
        ]
        return encoded_list

    def decode(self, tokens: torch.Tensor) -> np.ndarray:
        # The codes are flattened, so if there is an incomplete step, drop it
        tokens = tokens[: len(tokens) // self.n_codebooks() * self.n_codebooks()]

        codes = rearrange(
            tokens, "(t n_codebooks) -> n_codebooks t", n_codebooks=self.n_codebooks()
        )
        decoded = self.codec.decode(codes[None, :, :])[0, 0]
        return decoded.detach().to("cpu", dtype=torch.float32).numpy()

    def vocab_size(self):
        return self.codec.config.codebook_size

    def dtype(self):
        return torch.int32

    def __str__(self):
        return self.name

    def n_codebooks(self):
        return self.codec.config.n_codebooks
