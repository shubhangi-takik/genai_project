"""Create original-versus-reconstructed images with a trained autoencoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from model import Autoencoder
from train import DEFAULT_DATA, OUTPUT_DIR, build_loader


def load_model(checkpoint: Path, device: torch.device) -> Autoencoder:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = Autoencoder(state.get("latent_channels", 256)).to(device)
    model.load_state_dict(state["model_state_dict"]); model.eval()
    return model


def save_comparison(original: torch.Tensor, reconstructed: torch.Tensor, output: Path) -> None:
    count = min(8, original.size(0)); size = original.shape[-1]
    result = Image.new("RGB", (count * size, 2 * size + 40), "white")
    for i in range(count):
        for row, tensor, label in ((0, original[i], "Original"), (1, reconstructed[i], "Reconstructed")):
            array = (tensor.permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255).astype("uint8")
            result.paste(Image.fromarray(array), (i * size, 20 + row * size))
            ImageDraw.Draw(result).text((i * size + 2, row * size + 2), label, fill="black")
    result.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=OUTPUT_DIR / "best_autoencoder.pt")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(); OUTPUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device); batch = next(iter(build_loader(args.data_root, args.split, args.batch_size, 0, False)))
    with torch.no_grad(): reconstructed = model(batch.to(device)).cpu()
    save_comparison(batch, reconstructed, OUTPUT_DIR / "original_vs_reconstructed.png")
    print(f"Saved {OUTPUT_DIR / 'original_vs_reconstructed.png'}")


if __name__ == "__main__":
    main()
