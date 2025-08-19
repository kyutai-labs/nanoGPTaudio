from typing import Generic, Protocol, TypeVar

import librosa
import numpy as np
import torch
from einops import rearrange

from codec import Codec

T = TypeVar("T")


class Tokenizer(Protocol, Generic[T]):
    def encode(self, raw: T) -> torch.Tensor: ...

    def decode(self, tokens: torch.Tensor) -> T: ...


class CharTokenizer(Tokenizer[str]):
    def __init__(self, meta: dict):
        self.stoi = meta["stoi"]
        self.itos = meta["itos"]

    def encode(self, raw: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in raw], dtype=torch.int64)

    def decode(self, tokens: torch.Tensor) -> str:
        return "".join([self.itos[str(i)] for i in tokens.cpu().numpy()])


class TiktokenTokenizer(Tokenizer[str]):
    def __init__(self, encoding_name: str):
        import tiktoken

        self.encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, raw: str) -> torch.Tensor:
        return torch.tensor(
            self.encoding.encode(raw, allowed_special={"<|endoftext|>"}),
            dtype=torch.int64,
        )

    def decode(self, tokens: torch.Tensor) -> str:
        return self.encoding.decode(tokens.cpu().numpy().tolist())


class MuLawTokenizer(Tokenizer[np.ndarray]):
    def encode(self, raw: np.ndarray) -> torch.Tensor:
        return torch.tensor(librosa.mu_compress(raw, mu=255) + 128)

    def decode(self, tokens: torch.Tensor) -> np.ndarray:
        return librosa.mu_expand(tokens.cpu().numpy() - 128, mu=255)


class CodecTokenizer(Tokenizer[np.ndarray]):
    def __init__(self, meta: dict, device: str = "cuda"):
        self.codec = Codec.from_checkpoint(meta["encoding"], device=device)
        self.device = device

    def encode(self, raw: np.ndarray) -> torch.Tensor:
        audio = torch.Tensor(raw).to(self.device)
        factor = self.codec.encoder.downscaling_factor()
        audio = audio[None, None, : len(audio) // factor * factor]

        with torch.no_grad():
            codes, _reconstructed, _losses = self.codec(audio)
        flat_codes = rearrange(codes, "1 n_codes t -> (t n_codes)")
        return flat_codes

    def decode(self, tokens: torch.Tensor) -> np.ndarray:
        codes = rearrange(
            tokens,
            "(t n_codebooks) -> n_codebooks t",
            n_codebooks=4,
        )
        decoded = self.codec.decode(codes[None, :, :])[0, 0]
        return decoded.to("cpu", dtype=torch.float32).numpy()
