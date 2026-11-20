"""Calculate reconstruction metrics on the held-out test split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from inference import load_model
from train import DEFAULT_DATA, OUTPUT_DIR, build_loader, plot_history, plot_single_loss


def ssim_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Channel-averaged SSIM using an 11x11 Gaussian window on [0,1] images."""
    coords = torch.arange(11, device=prediction.device, dtype=prediction.dtype) - 5
    kernel_1d = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2)); kernel_1d /= kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).expand(3, 1, 11, 11)
    mu_x = F.conv2d(prediction, kernel, padding=5, groups=3); mu_y = F.conv2d(target, kernel, padding=5, groups=3)
    sigma_x = F.conv2d(prediction * prediction, kernel, padding=5, groups=3) - mu_x.square()
    sigma_y = F.conv2d(target * target, kernel, padding=5, groups=3) - mu_y.square()
    sigma_xy = F.conv2d(prediction * target, kernel, padding=5, groups=3) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + .01 ** 2) * (2 * sigma_xy + .03 ** 2)) / ((mu_x.square() + mu_y.square() + .01 ** 2) * (sigma_x + sigma_y + .03 ** 2))
    return score.mean(dim=(1, 2, 3))


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    pixels = (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(pixels)


def save_comparison_grid(
    pairs: list[tuple[Image.Image, Image.Image]], output_path: Path, columns: int = 4
) -> None:
    if not pairs:
        raise ValueError("No test images were available for the comparison grid.")
    width, height = pairs[0][0].size
    label_height = 22
    rows = math.ceil(len(pairs) / columns)
    grid = Image.new("RGB", (columns * width * 2, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(grid)
    for index, (original, reconstructed) in enumerate(pairs):
        row, column = divmod(index, columns)
        x = column * width * 2
        y = row * (height + label_height)
        draw.text((x + 2, y + 4), f"Original {index + 1}", fill="black")
        draw.text((x + width + 2, y + 4), "Reconstructed", fill="black")
        grid.paste(original, (x, y + label_height))
        grid.paste(reconstructed, (x + width, y + label_height))
    grid.save(output_path)


def save_training_graphs(output_dir: Path) -> list[Path]:
    history_path = OUTPUT_DIR / "training_history.json"
    if not history_path.is_file():
        print(f"Training history not found; skipping loss graphs: {history_path}")
        return []

    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = history.get("epoch", [])
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    if not epochs or len(epochs) != len(train_loss) or len(epochs) != len(val_loss):
        raise ValueError(f"Training history has inconsistent epoch/loss series: {history_path}")

    combined = output_dir / "loss_history.png"
    training = output_dir / "training_loss.png"
    validation = output_dir / "validation_loss.png"
    plot_history(history, combined)
    plot_single_loss(history, "train_loss", "Training loss", "#1f77b4", training)
    plot_single_loss(history, "val_loss", "Validation loss", "#d62728", validation)
    return [combined, training, validation]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=OUTPUT_DIR / "best_autoencoder.pt")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "evaluation_test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grid-count", type=int, default=32)
    args = parser.parse_args(); OUTPUT_DIR.mkdir(exist_ok=True)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.grid_count < 1:
        raise ValueError("--grid-count must be at least 1")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    loader = build_loader(args.data_root, "test", args.batch_size, 0, False)
    if len(loader.dataset) == 0:
        raise ValueError("The held-out test split is empty.")

    output_dir = args.output_dir.resolve()
    originals_dir = output_dir / "originals"
    reconstructions_dir = output_dir / "reconstructions"
    originals_dir.mkdir(parents=True, exist_ok=True)
    reconstructions_dir.mkdir(parents=True, exist_ok=True)
    source_paths = loader.dataset.paths
    metric_rows: list[dict] = []
    grid_pairs: list[tuple[Image.Image, Image.Image]] = []
    mse_sum = mae_sum = ssim_sum = 0.0
    pixels = images = 0
    inference_time = 0.0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            batch = batch.to(device, non_blocking=True)
            if device.type == "cuda": torch.cuda.synchronize()
            start = time.perf_counter()
            reconstruction = model(batch)
            if device.type == "cuda": torch.cuda.synchronize()
            inference_time += time.perf_counter() - start
            batch_mse = F.mse_loss(reconstruction, batch, reduction="none").flatten(1).mean(1)
            batch_mae = F.l1_loss(reconstruction, batch, reduction="none").flatten(1).mean(1)
            batch_ssim = ssim_per_image(reconstruction, batch)
            mse_sum += batch_mse.sum().item() * batch[0].numel()
            mae_sum += batch_mae.sum().item() * batch[0].numel()
            ssim_sum += batch_ssim.sum().item()
            pixels += batch.numel(); images += batch.size(0)

            for batch_offset in range(batch.size(0)):
                index = batch_index * args.batch_size + batch_offset
                source_path = source_paths[index]
                image_id = f"{index:06d}_{source_path.stem}"
                original_path = originals_dir / f"{image_id}{source_path.suffix.lower()}"
                reconstructed_path = reconstructions_dir / f"{image_id}_reconstructed.png"
                shutil.copy2(source_path, original_path)
                original_image = tensor_to_image(batch[batch_offset])
                reconstructed_image = tensor_to_image(reconstruction[batch_offset])
                reconstructed_image.save(reconstructed_path)
                if len(grid_pairs) < args.grid_count:
                    grid_pairs.append((original_image, reconstructed_image))
                image_mse = batch_mse[batch_offset].item()
                metric_rows.append(
                    {
                        "index": index,
                        "source_image": str(source_path),
                        "original_copy": str(original_path),
                        "reconstruction": str(reconstructed_path),
                        "mse": image_mse,
                        "mae": batch_mae[batch_offset].item(),
                        "psnr_db": 10 * math.log10(1.0 / max(image_mse, 1e-12)),
                        "ssim": batch_ssim[batch_offset].item(),
                    }
                )

    mse = mse_sum / pixels
    mae = mae_sum / pixels
    results = {
        "split": "test",
        "images": images,
        "mse": mse,
        "mae": mae,
        "psnr_db": 10 * math.log10(1.0 / max(mse, 1e-12)),
        "ssim": ssim_sum / images,
        "reconstruction_loss": mse,
        "inference_time_seconds": inference_time,
        "mean_inference_time_per_image_seconds": inference_time / images,
        "device": str(device),
        "checkpoint": str(args.checkpoint.resolve()),
    }

    per_image_csv = output_dir / "metrics_per_image.csv"
    with per_image_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    summary_csv = output_dir / "metrics_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results))
        writer.writeheader()
        writer.writerow(results)

    summary_json = output_dir / "evaluation_results.json"
    summary_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    grid_path = output_dir / "original_vs_reconstructed_grid.png"
    save_comparison_grid(grid_pairs, grid_path)
    graph_paths = save_training_graphs(output_dir)
    print(f"Saved {images} original test images to {originals_dir}")
    print(f"Saved {images} reconstructions to {reconstructions_dir}")
    print(f"Saved comparison grid to {grid_path} ({len(grid_pairs)} pairs)")
    print(f"Saved per-image metrics to {per_image_csv}")
    print(f"Saved summary metrics to {summary_csv}")
    print(f"Saved training graphs: {', '.join(str(path) for path in graph_paths) if graph_paths else 'none'}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
