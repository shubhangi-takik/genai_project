"""Conditional GAN (Pix2Pix) for structure-preserving interior generation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter, ImageOps


def extract_structural_edges(image: Image.Image, image_size: int = 128) -> Image.Image:
    """Extract structural architectural boundaries (walls, floor lines, windows)."""
    resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    gray = resized.convert("L")
    # Multi-pass gradient edge filter
    edges = gray.filter(ImageFilter.FIND_EDGES)
    # Enhance contrast to isolate structural contours
    edges = ImageOps.autocontrast(edges)
    return edges.convert("RGB")


def image_to_tensor(image: Image.Image, image_size: int = 128) -> torch.Tensor:
    """Convert RGB image to [3, H, W] float tensor in [0, 1]."""
    resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert [3, H, W] float tensor in [0, 1] to PIL Image."""
    pixels = (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


class UNetBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, down: bool = True, use_dropout: bool = False):
        super().__init__()
        self.down = down
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            layers = [
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            ]
            if use_dropout:
                layers.append(nn.Dropout(0.3))
            self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalGenerator(nn.Module):
    """U-Net Generator with skip connections to preserve room boundaries."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, features: int = 32):
        super().__init__()
        # Initial encoder (no batch norm)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = UNetBlock(features, features * 2, down=True)
        self.enc3 = UNetBlock(features * 2, features * 4, down=True)
        self.enc4 = UNetBlock(features * 4, features * 8, down=True)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder with skip connection concatenation
        self.dec4 = UNetBlock(features * 8, features * 8, down=False, use_dropout=True)
        self.dec3 = UNetBlock(features * 16, features * 4, down=False, use_dropout=True)
        self.dec2 = UNetBlock(features * 8, features * 2, down=False)
        self.dec1 = UNetBlock(features * 4, features, down=False)

        # Output projection
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder passes
        e1 = self.enc1(x)         # 128 -> 64
        e2 = self.enc2(e1)        # 64 -> 32
        e3 = self.enc3(e2)        # 32 -> 16
        e4 = self.enc4(e3)        # 16 -> 8
        b = self.bottleneck(e4)   # 8 -> 4

        # Decoder passes with skip connections
        d4 = self.dec4(b)
        d3 = self.dec3(torch.cat([d4, e4], dim=1))
        d2 = self.dec2(torch.cat([d3, e3], dim=1))
        d1 = self.dec1(torch.cat([d2, e2], dim=1))
        out = self.final(torch.cat([d1, e1], dim=1))
        return out


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN Discriminator evaluating realism of conditioned image pairs."""

    def __init__(self, in_channels: int = 6, features: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            # Concatenated [condition, real_or_fake] = 6 channels
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 4, 1, kernel_size=4, stride=1, padding=1)
            # Outputs raw patch logits
        )

    def forward(self, condition: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = torch.cat([condition, target], dim=1)
        return self.net(x)
