"""Skip-connected convolutional autoencoder and aspect-preserving image helpers."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import nn


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ImprovedAutoencoder(nn.Module):
    """A U-shaped autoencoder that retains spatial features through skip connections."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = _block(3, base_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        self.enc2 = _block(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)
        self.enc3 = _block(base_channels * 4, base_channels * 4)
        self.down3 = nn.Conv2d(base_channels * 4, base_channels * 8, 4, stride=2, padding=1)
        self.enc4 = _block(base_channels * 8, base_channels * 8)
        self.down4 = nn.Conv2d(base_channels * 8, base_channels * 8, 4, stride=2, padding=1)
        self.bottleneck = _block(base_channels * 8, base_channels * 8)

        self.dec4 = _block(base_channels * 16, base_channels * 8)
        self.dec3 = _block(base_channels * 12, base_channels * 4)
        self.dec2 = _block(base_channels * 6, base_channels * 2)
        self.dec1 = _block(base_channels * 3, base_channels)
        self.output = nn.Conv2d(base_channels, 3, kernel_size=1)

    @staticmethod
    def _up_and_join(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((x, skip), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Expected an image batch with shape [batch, 3, height, width].")
        if min(x.shape[-2:]) < 32 or x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError("Image height and width must both be divisible by 16.")

        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        z = self.bottleneck(self.down4(e4))

        d4 = self.dec4(self._up_and_join(z, e4))
        d3 = self.dec3(self._up_and_join(d4, e3))
        d2 = self.dec2(self._up_and_join(d3, e2))
        d1 = self.dec1(self._up_and_join(d2, e1))
        return torch.sigmoid(self.output(d1))


def image_to_tensor(image: Image.Image, image_size: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Fit an RGB image inside a square without changing its aspect ratio."""
    if image_size < 32 or image_size % 16:
        raise ValueError("image_size must be at least 32 and divisible by 16.")
    rgb = image.convert("RGB")
    scale = min(image_size / rgb.width, image_size / rgb.height)
    width = max(1, round(rgb.width * scale))
    height = max(1, round(rgb.height * scale))
    resized = rgb.resize((width, height), Image.Resampling.BILINEAR)
    left = (image_size - width) // 2
    top = (image_size - height) // 2
    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    canvas.paste(resized, (left, top))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor, (left, top, left + width, top + height)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    pixels = (
        tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def restore_image(
    tensor: torch.Tensor,
    content_box: tuple[int, int, int, int],
    original_size: tuple[int, int],
) -> Image.Image:
    """Remove letterbox padding and return the reconstruction at source dimensions."""
    return tensor_to_image(tensor).crop(content_box).resize(original_size, Image.Resampling.BILINEAR)
