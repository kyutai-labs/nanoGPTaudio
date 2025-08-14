"""
Based on a single level VQ-VAE from Jukebox:
https://arxiv.org/abs/2005.00341
"""

from dataclasses import dataclass

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


class Bottleneck(nn.Module):
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
        codes = distances.argmin(dim=0)
        return codes

    def decode(self, codes: torch.Tensor):
        quantized = F.embedding(codes, self.codebook())
        return quantized

    def forward(self, embeddings: torch.Tensor):
        assert embeddings.dim() == 2, (
            f"Expected shape [batch, channels], got {embeddings.shape=}"
        )

        codes = self.encode(embeddings)
        quantized = self.decode(codes)

        # Straight-through estimator: we pretend like we didn't quantize the embeddings.
        # We do this by treating quantization as the addition of a constant vector
        # TODO: why doesn't this work with torch.compile()?
        #   Getting "Trying to backward through the graph a second time" error
        quantized = embeddings + (quantized - embeddings).detach()

        commitment_loss = F.mse_loss(quantized.detach(), embeddings)

        if self.training:
            cur_code_usage = torch.zeros_like(self.code_usage).scatter_add(
                0, codes, torch.ones_like(codes, dtype=self.code_usage.dtype)
            )
            exponential_moving_average_update(
                self.code_usage, cur_code_usage, self.codebook_update_speed
            )

            cur_code_embedding_sum = torch.zeros_like(
                self.code_embedding_sum
            ).scatter_add(0, repeat(codes, "n -> n d", d=self.channels), embeddings)

            exponential_moving_average_update(
                self.code_embedding_sum,
                cur_code_embedding_sum,
                self.codebook_update_speed,
            )

        return codes, quantized, commitment_loss


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
    spectral_loss_weight: float
    commitment_loss_weight: float


class Codec(nn.Module):
    def __init__(self, config: CodecConfig):
        super().__init__()
        self.config = config

        self.encoder = Encoder(channels=config.channels, n_blocks=config.n_blocks)
        self.decoder = Decoder(channels=config.channels, n_blocks=config.n_blocks)
        self.bottleneck = Bottleneck(
            channels=config.channels, codebook_size=config.codebook_size
        )
        self.multiscale_spectrogram_loss = MultiscaleSpectrogramLoss()

    def forward(self, audio):
        embeddings = self.encoder(audio)

        flat_embeddings = rearrange(embeddings, "b c t -> (b t) c")

        _codes_flat, flat_embeddings_q, commitment_loss = self.bottleneck(
            flat_embeddings
        )
        embeddings_q = rearrange(
            flat_embeddings_q, "(b t) c -> b c t", b=audio.shape[0]
        )

        reconstructed = self.decoder(embeddings_q)

        loss = F.mse_loss(reconstructed, audio)
        loss += commitment_loss * self.config.commitment_loss_weight

        if self.config.spectral_loss_weight > 0:
            loss_spectral = self.multiscale_spectrogram_loss(audio, reconstructed)
            loss += self.config.spectral_loss_weight * loss_spectral

        return reconstructed, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            fused=device_type == "cuda",
        )
        return optimizer
