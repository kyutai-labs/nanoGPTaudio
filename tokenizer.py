import re
from pathlib import Path
from typing import Generic, Protocol, TypeVar, cast

import librosa
import numpy as np
import torch
from einops import rearrange
from transformers import MimiModel

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

    def sample_rate(self) -> int:
        """Sample rate of the audio expected by this tokenizer (if applicable)."""
        # The tokenizer abstraction is a bit leaky here - not applicable to text
        raise NotImplementedError()


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
        return torch.tensor(librosa.mu_compress(raw.clip(-1, 1), mu=255) + 128).to(
            torch.uint8
        )

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

    def sample_rate(self) -> int:
        return 16000

    def __str__(self):
        return self.name()


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
        decoded = self.codec.decode(codes[None, :, :])
        return decoded[0, 0].detach().to("cpu", dtype=torch.float32).numpy()

    def vocab_size(self):
        return self.codec.config.codebook_size

    def dtype(self):
        return torch.int32

    def __str__(self):
        return self.name

    def n_codebooks(self):
        return self.codec.config.n_codebooks

    def sample_rate(self) -> int:
        return 16000


class MimiTokenizer(Tokenizer[np.ndarray]):
    def __init__(
        self,
        semantic: bool = True,
        n_codebooks: int | None = None,
        device: str = "cuda",
    ):
        self.semantic = semantic
        self._n_codebooks = n_codebooks
        self.device = device
        self.mimi = MimiModel.from_pretrained("kyutai/mimi").to(device)

        if n_codebooks:
            max_codebooks = self.mimi.config.num_quantizers
            if not self.semantic:
                max_codebooks -= 1
            assert 1 <= n_codebooks <= self.mimi.config.num_quantizers, (
                f"levels must be between 1 and {self.mimi.config.num_quantizers}"
            )

    @staticmethod
    def from_name(name: str, device: str = "cuda") -> "MimiTokenizer":
        semantic = True
        n_codebooks = None
        # Match patterns like 'mimi', 'mimi_8_rvq', 'mimi_16_rvq_nosemantic'
        pattern = r"^mimi(?:_(\d+)_rvq)?(?:_nosemantic)?$"
        match = re.match(pattern, name)
        if not match:
            raise ValueError(f"Invalid MimiTokenizer name: {name}")
        if match.group(1):
            n_codebooks = int(match.group(1))
        if name.endswith("nosemantic"):
            semantic = False
        return MimiTokenizer(semantic=semantic, n_codebooks=n_codebooks, device=device)

    def encode(self, raw: np.ndarray) -> torch.Tensor:
        audio = torch.Tensor(raw).to(self.device)

        is_batched = audio.ndim == 2
        if not is_batched:
            audio = audio[None, :]

        audio = rearrange(audio, "b t -> b 1 t")

        with torch.no_grad():
            codes = cast(torch.Tensor, self.mimi.encode(audio).audio_codes)
            codes = codes.to(dtype=torch.int32)

            if not self.semantic:
                codes = codes[:, 1:]

            if self._n_codebooks:
                codes = codes[:, : self._n_codebooks]

            # To avoid having to model multiple streams, flatten the levels of the RVQ
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

        factor = self.downscaling_factor()
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
        if not self.semantic:
            # Prepend zeroes for the semantic codes
            zero_semantic = torch.zeros(
                (1, codes.shape[1]), dtype=codes.dtype, device=codes.device
            )
            codes = torch.cat([zero_semantic, codes], dim=0)

        decoded = self.mimi.decode(codes[None, :, :]).audio_values

        audio = decoded[0, 0].detach().to("cpu", dtype=torch.float32).numpy()

        return audio

    def vocab_size(self):
        return self.mimi.config.codebook_size

    def dtype(self):
        return torch.int32

    def __str__(self):
        name = "mimi"
        if self._n_codebooks:
            name += f"_{self._n_codebooks}_rvq"
        if not self.semantic:
            name += "_nosemantic"
        return name

    def n_codebooks(self):
        if self._n_codebooks:
            return self._n_codebooks
        elif self.semantic:
            return self.mimi.config.num_quantizers
        else:
            return self.mimi.config.num_quantizers - 1

    def sample_rate(self) -> int:
        # Always 24000 in practice, but let's not hardcode
        return self.mimi.config.sampling_rate

    def downscaling_factor(self):
        return int(self.sample_rate() / self.mimi.config._frame_rate)


def audio_tokenizer_from_name(name: str, device: str = "cuda"):
    if name.startswith("codec"):
        return CodecTokenizer(name, device=device)
    elif name.startswith("mimi"):
        return MimiTokenizer.from_name(name, device=device)
    elif name == "mu-law-256":
        return MuLawTokenizer()
    else:
        raise ValueError(f"Could not parse audio tokenizer name: {name}")
