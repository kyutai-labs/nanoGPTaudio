"""
Based on a single level VQ-VAE from Jukebox:
https://arxiv.org/abs/2005.00341
"""

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from einops import rearrange, repeat
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.channels = channels
        self.dilation = dilation

        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                dilation=dilation,
                padding=dilation,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, x: torch.Tensor):
        return x + self.model(x)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_residual_blocks: int,
    ):
        super().__init__()
        self.n_residual_blocks = n_residual_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsampling_conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.residual_blocks = nn.ModuleList([])
        for i in range(n_residual_blocks):
            self.residual_blocks.append(
                ResidualBlock(channels=self.out_channels, dilation=3 ** (1 + i))
            )

    def forward(self, x: torch.Tensor):
        x = self.downsampling_conv(x)
        for i in range(self.n_residual_blocks):
            x = self.residual_blocks[i](x)

        return x


class Encoder(nn.Module):
    def __init__(self, channels: int, n_blocks: int):
        super().__init__()
        self.channels = channels
        # Jukebox actually uses different widths for the EncoderBlock and adds a conv
        # at the end to get to an output embedding width, see
        # https://github.com/openai/jukebox/blob/master/jukebox/vqvae/encdec.py
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    in_channels=1 if i == 0 else channels,
                    out_channels=channels,
                    n_residual_blocks=4,
                )
                for i in range(n_blocks)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def downscaling_factor(self):
        return 2 ** len(self.blocks)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_residual_blocks: int,
    ):
        super().__init__()
        self.n_residual_blocks = n_residual_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.residual_blocks = nn.ModuleList([])
        for i in range(n_residual_blocks):
            self.residual_blocks.append(
                ResidualBlock(
                    channels=self.in_channels,
                    dilation=3 ** (i + 1),
                )
            )

        self.upsampling_conv = nn.ConvTranspose1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor):
        for i in range(self.n_residual_blocks):
            x = x + self.residual_blocks[i](x)
        x = self.upsampling_conv(x)

        return x


class Decoder(nn.Module):
    def __init__(self, channels: int, n_blocks: int):
        super().__init__()
        self.channels = channels
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    in_channels=channels,
                    out_channels=1 if i == n_blocks - 1 else channels,
                    n_residual_blocks=4,
                )
                for i in range(n_blocks)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def exponential_moving_average_update(
    old: torch.Tensor, new: torch.Tensor, update_speed: float
):
    old.mul_(1 - update_speed).add_(new * update_speed)


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        channels: int,
        codebook_size: int,
        codebook_update_speed: float = 0.01,
    ):
        super().__init__()
        self.channels = channels
        self.codebook_size = codebook_size
        self.codebook_update_speed = codebook_update_speed

        self.code_usage: torch.Tensor
        self.register_buffer("code_usage", torch.ones(codebook_size))
        self.code_embedding_sum: torch.Tensor
        self.register_buffer(
            "code_embedding_sum",
            torch.nn.init.kaiming_uniform_(torch.empty(codebook_size, channels)),
        )

    def codebook(self) -> torch.Tensor:
        """Compute the codebook from the moving average statistics."""
        return self.code_embedding_sum / self.code_usage.clamp(min=1e-5)[:, None]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the code indices for `x` of shape [batch, channels]."""
        assert x.dim() == 2
        distances = torch.cdist(self.codebook(), x)
        # int64 is default, save some space
        codes = distances.argmin(dim=0).to(dtype=torch.int32)
        return codes

    def decode(self, codes: torch.Tensor):
        """Return the continuous embeddings for `codes` of shape [batch]"""
        quantized = F.embedding(codes, self.codebook())
        return quantized

    def forward(self, embeddings: torch.Tensor):
        assert embeddings.dim() == 2, (
            f"Expected shape [batch, channels], got {embeddings.shape=}"
        )

        codes = self.encode(embeddings)
        embeddings_q = self.decode(codes)

        # Straight-through estimator: we pretend like we didn't quantize the embeddings.
        # We do this by treating quantization as the addition of a constant vector
        # TODO: why doesn't this work with torch.compile()?
        #   Getting "Trying to backward through the graph a second time" error
        embeddings_q = embeddings + (embeddings_q - embeddings).detach()

        commitment_loss = F.mse_loss(embeddings_q.detach(), embeddings)

        if self.training:
            # The no_grad() is needed here so that the computational graph doesn't keep
            # growing with all the EMA updates
            with torch.no_grad():
                cur_code_usage = torch.zeros_like(self.code_usage).scatter_add(
                    0, codes, torch.ones_like(codes, dtype=self.code_usage.dtype)
                )
                exponential_moving_average_update(
                    self.code_usage, cur_code_usage, self.codebook_update_speed
                )

                cur_code_embedding_sum = torch.zeros_like(
                    self.code_embedding_sum
                ).scatter_add(
                    0,
                    repeat(codes, "n -> n d", d=self.channels).to(dtype=torch.long),
                    embeddings,
                )

                exponential_moving_average_update(
                    self.code_embedding_sum,
                    cur_code_embedding_sum,
                    self.codebook_update_speed,
                )

        return codes, embeddings_q, commitment_loss

    def get_codebook_entropy(self):
        """Entropy in the information theory sense, normalized to [0, 1]."""
        proba = self.code_usage / self.code_usage.sum()
        p_log_p = torch.where(proba == 0, 0, proba * torch.log(proba))
        entropy = -p_log_p.sum()
        # The maximum entropy is reached by a uniform probability distribution
        max_possible_entropy = math.log(self.codebook_size)
        return float(entropy / max_possible_entropy)

    def get_fraction_unused_codes(self, threshold: float = 1):
        return torch.mean((self.code_usage < threshold).float()).item()

    def restart_unused_codes(self, batch: torch.Tensor):
        """Restart a codebook entry that is used very little, if there is one.

        Returns True if a code was restarted, False otherwise.
        """
        value, index = self.code_usage.min(dim=0)
        if value < 0.1:
            new_usage = 1.0
            self.code_usage[index] = (
                new_usage  # Fake value, to make sure it's not restarted again
            )
            self.code_embedding_sum[index] = (
                batch[torch.randint(0, len(batch), (1,))] * new_usage
            )
            return True

        return False


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        channels: int,
        codebook_size: int,
        n_codebooks: int,
    ):
        super().__init__()

        self.channels = channels
        self.codebook_size = codebook_size
        self.n_codebooks = n_codebooks

        self.bottlenecks = nn.ModuleList(
            [
                VectorQuantizer(
                    channels=channels,
                    codebook_size=codebook_size,
                )
                for _ in range(n_codebooks)
            ]
        )

    def decode(self, codes: torch.Tensor):
        """Return the continuous embeddings for `codes` of shape [batch, n_codebooks]"""
        embeddings_q = torch.zeros(codes.shape[0], self.channels, device=codes.device)
        for i in range(self.n_codebooks):
            embeddings_q += self.bottlenecks[i].decode(codes[:, i])
        return embeddings_q

    def forward(self, embeddings: torch.Tensor):
        assert embeddings.dim() == 2, (
            f"Expected shape [batch, channels], got {embeddings.shape=}"
        )

        codes = torch.zeros(
            embeddings.shape[0],
            self.n_codebooks,
            device=embeddings.device,
            dtype=torch.int32,
        )

        embeddings_q = torch.zeros_like(embeddings)

        commitment_loss = torch.scalar_tensor(0, device=embeddings.device)

        for i in range(self.n_codebooks):
            residual = embeddings - embeddings_q
            cur_codes, cur_embeddings_q, cur_commitment_loss = self.bottlenecks[i](
                residual
            )

            codes[:, i] = cur_codes
            embeddings_q += cur_embeddings_q
            commitment_loss += cur_commitment_loss

        return codes, embeddings_q, commitment_loss

    def get_codebook_entropy(self):
        entropies = [b.get_codebook_entropy() for b in self.bottlenecks]
        return sum(entropies) / len(entropies)

    def get_fraction_unused_codes(self, threshold: float = 1):
        fractions = [b.get_fraction_unused_codes() for b in self.bottlenecks]
        return sum(fractions) / len(fractions)


class MultiscaleSpectrogramLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.spectrograms = nn.ModuleList(
            [
                torchaudio.transforms.Spectrogram(
                    n_fft=2048, hop_length=240, win_length=1200, power=1
                ),
                torchaudio.transforms.Spectrogram(
                    n_fft=1024, hop_length=120, win_length=600, power=1
                ),
                torchaudio.transforms.Spectrogram(
                    n_fft=512, hop_length=50, win_length=240, power=1
                ),
            ]
        )

    def forward(self, audio: torch.Tensor, reconstructed: torch.Tensor):
        losses = []
        for spec in self.spectrograms:
            spec = spec.to(audio.device)
            diff = spec(audio) - spec(reconstructed)
            losses.append(torch.sqrt(torch.mean(diff**2)))
        return torch.mean(torch.stack(losses))


@dataclass
class CodecConfig:
    # Both in the encoder and decoder so that the downsampling/upsamling matches
    n_blocks: int
    channels: int
    codebook_size: int
    n_codebooks: int
    spectral_loss_weight: float
    commitment_loss_weight: float


class Codec(nn.Module):
    def __init__(self, config: CodecConfig):
        super().__init__()
        self.config = config

        self.encoder = Encoder(channels=config.channels, n_blocks=config.n_blocks)
        self.decoder = Decoder(channels=config.channels, n_blocks=config.n_blocks)
        self.bottleneck = ResidualVectorQuantizer(
            channels=config.channels,
            codebook_size=config.codebook_size,
            n_codebooks=config.n_codebooks,
        )
        self.multiscale_spectrogram_loss = MultiscaleSpectrogramLoss()

    def decode(self, codes: torch.Tensor):
        assert codes.dim() == 3, (
            f"Expected shape [batch, n_codebooks, time], got {codes.shape=}"
        )
        assert codes.shape[1] == self.config.n_codebooks, (
            f"Expected {self.config.n_codebooks} codebooks, got {codes.shape[1]}, "
            f"{codes.shape=}"
        )

        batch_size = codes.shape[0]
        flat_codes = rearrange(codes, "b n_codebooks t -> (b t) n_codebooks")

        flat_embeddings_q = self.bottleneck.decode(flat_codes)
        embeddings_q = rearrange(flat_embeddings_q, "(b t) c -> b c t", b=batch_size)
        reconstructed = self.decoder(embeddings_q)

        return reconstructed

    def forward(self, audio: torch.Tensor):
        assert audio.dim() == 3, f"Expected shape [batch, 1, time], got {audio.shape=}"
        assert audio.shape[1] == 1, (
            "Wrong number of channels. "
            f"Expected shape [batch, 1, time], got {audio.shape=}"
        )

        factor = self.encoder.downscaling_factor()
        assert audio.shape[2] % factor == 0, (
            "Audio length must be divisible by the downscaling factor of the encoder "
            f"({factor}), got length {audio.shape[2]}. "
            f"Try {audio.shape[2] // factor * factor}."
        )

        embeddings = self.encoder(audio)

        flat_embeddings = rearrange(embeddings, "b c t -> (b t) c")

        codes_flat, flat_embeddings_q, commitment_loss = self.bottleneck(
            flat_embeddings
        )
        embeddings_q = rearrange(
            flat_embeddings_q, "(b t) c -> b c t", b=audio.shape[0]
        )
        codes = rearrange(codes_flat, "(b t) n_codes -> b n_codes t", b=audio.shape[0])

        reconstructed = self.decoder(embeddings_q)

        losses = {}
        losses["mse"] = F.mse_loss(reconstructed, audio)
        losses["commitment"] = commitment_loss

        if self.config.spectral_loss_weight > 0:
            loss_spectral = self.multiscale_spectrogram_loss(audio, reconstructed)
            losses["spectral"] = loss_spectral
        else:
            losses["spectral"] = torch.Tensor(0.0, device=audio.device)

        losses["total"] = (
            losses["mse"]
            + losses["commitment"] * self.config.commitment_loss_weight
            + losses["spectral"] * self.config.spectral_loss_weight
        )
        # Not sure these detach_() calls are needed
        losses["mse"].detach_()
        losses["commitment"].detach_()
        losses["spectral"].detach_()

        return codes, reconstructed, losses

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            fused=device_type == "cuda",
        )
        return optimizer

    @staticmethod
    def from_checkpoint(checkpoint_path_or_name: Path | str, device: str = "cuda"):
        try:
            checkpoint = torch.load(checkpoint_path_or_name, map_location=device)
        except FileNotFoundError:
            checkpoint_path = (
                Path(__file__).parent
                / "models"
                / checkpoint_path_or_name
                / "codec_ckpt.pt"
            )
            checkpoint = torch.load(checkpoint_path, map_location=device)

        config = CodecConfig(**checkpoint["model_args"])
        model = Codec(config).to(device)

        state_dict = checkpoint["model"]
        model.load_state_dict(state_dict)

        return model
