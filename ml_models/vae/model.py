"""Convolutional VAE with a diagonal Gaussian latent distribution."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import nn


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 128, image_size: int = 128, base_channels: int = 32) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive.")
        if image_size < 32 or image_size % 16:
            raise ValueError("image_size must be at least 32 and divisible by 16.")
        if base_channels < 1:
            raise ValueError("base_channels must be positive.")

        self.latent_dim = latent_dim
        self.image_size = image_size
        self.base_channels = base_channels
        self.feature_size = image_size // 16
        channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)

        encoder_layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            encoder_layers.extend(
                (
                    nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder_layers)

        flattened_features = channels[-1] * self.feature_size * self.feature_size
        self.to_mu = nn.Linear(flattened_features, latent_dim)
        self.to_logvar = nn.Linear(flattened_features, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flattened_features)

        decoder_layers: list[nn.Module] = []
        decoder_channels = (*channels[-2::-1], 3)
        in_channels = channels[-1]
        for index, out_channels in enumerate(decoder_channels):
            decoder_layers.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            )
            if index == len(decoder_channels) - 1:
                decoder_layers.append(nn.Sigmoid())
            else:
                decoder_layers.extend((nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))
            in_channels = out_channels
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected an image batch with shape [batch, 3, height, width].")
        if image.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"Expected {self.image_size}x{self.image_size} images.")
        features = self.encoder(image).flatten(start_dim=1)
        return self.to_mu(features), self.to_logvar(features).clamp(-20.0, 20.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"Expected latent vectors with shape [batch, {self.latent_dim}].")
        features = self.from_latent(latent)
        features = features.view(
            latent.shape[0],
            self.base_channels * 8,
            self.feature_size,
            self.feature_size,
        )
        return self.decoder(features)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(image)
        latent = self.reparameterize(mu, logvar)
        return self.decode(latent), mu, logvar

    def reconstruct(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deterministic reconstruction from the posterior mean for evaluation."""
        mu, logvar = self.encode(image)
        return self.decode(mu), mu, logvar

    def sample(
        self,
        count: int = 1,
        seed: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if count < 1:
            raise ValueError("count must be positive.")
        target_device = device or next(self.parameters()).device
        generator = torch.Generator(device=target_device)
        if seed is not None:
            generator.manual_seed(seed)
        latent = torch.randn(count, self.latent_dim, generator=generator, device=target_device)
        return self.decode(latent)


def image_to_tensor(image: Image.Image, image_size: int = 128) -> torch.Tensor:
    """Match the existing Autoencoder's RGB square resize and [0, 1] scaling."""
    if image_size < 32 or image_size % 16:
        raise ValueError("image_size must be at least 32 and divisible by 16.")
    resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    pixels = (
        tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")
