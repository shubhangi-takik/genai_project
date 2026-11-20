"""Compare the original and improved Autoencoders on the same held-out test images."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

from evaluation import plot_history, plot_single_loss, ssim_per_image
from inference import load_model as load_original_model
from model_improved import ImprovedAutoencoder, image_to_tensor, restore_image, tensor_to_image
from train import DEFAULT_DATA, InteriorImageDataset

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
IMPROVED_CHECKPOINT = OUTPUT_ROOT / "improved_autoencoder" / "best_autoencoder_improved.pt"
IMPROVED_HISTORY = OUTPUT_ROOT / "improved_autoencoder" / "training_history.json"
ORIGINAL_CHECKPOINT = OUTPUT_ROOT / "best_autoencoder.pt"
COMPARISON_DIR = OUTPUT_ROOT / "improved_comparison"
METRIC_SIZE = 256


def _metric_tensor(image: Image.Image) -> torch.Tensor:
    normalized = ImageOps.pad(
        image.convert("RGB"),
        (METRIC_SIZE, METRIC_SIZE),
        method=Image.Resampling.BILINEAR,
        color=(0, 0, 0),
    )
    array = np.asarray(normalized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _reconstruction_metrics(reconstruction: Image.Image, original: Image.Image) -> tuple[float, float]:
    prediction = _metric_tensor(reconstruction)
    target = _metric_tensor(original)
    mse = F.mse_loss(prediction, target).item()
    ssim = ssim_per_image(prediction, target).item()
    return mse, ssim


def _load_improved(checkpoint_path: Path, device: torch.device) -> ImprovedAutoencoder:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if state.get("architecture") != "improved_skip_autoencoder_v1":
        raise ValueError(f"Checkpoint is not an improved Autoencoder checkpoint: {checkpoint_path}")
    model = ImprovedAutoencoder(state.get("base_channels", 32)).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    return model


def _tile(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.contain(image.convert("RGB"), size, method=Image.Resampling.BILINEAR)


def _save_grid(rows: list[tuple[Image.Image, Image.Image, Image.Image]], path: Path, count: int) -> None:
    selected = rows[:count]
    if not selected:
        raise ValueError("No test images were available for the comparison grid.")
    columns = 3
    cell_width, cell_height, label_height = 256, 256, 24
    grid = Image.new(
        "RGB",
        (columns * cell_width, len(selected) * (cell_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(grid)
    for row_index, images in enumerate(selected):
        y = row_index * (cell_height + label_height)
        for column, (image, label) in enumerate(zip(images, ("Original", "Existing AE", "Improved AE"))):
            x = column * cell_width
            draw.text((x + 4, y + 5), label, fill="black")
            grid.paste(_tile(image, (cell_width, cell_height)), (x, y + label_height))
    grid.save(path)


def _save_metric_graph(metrics: dict[str, dict], path: Path) -> None:
    width, height = 900, 520
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "Held-out test comparison (lower MSE / higher SSIM is better)", fill="black")
    names = ("Existing Autoencoder", "Improved Autoencoder")
    colors = ("#1f77b4", "#ff7f0e")
    for panel, (key, label, better) in enumerate(
        (("mse", "Mean MSE", "lower"), ("ssim", "Mean SSIM", "higher"))
    ):
        top = 70 + panel * 220
        draw.text((30, top), f"{label} ({better} is better)", fill="black")
        maximum = max(metrics[name][key] for name in names)
        scale_max = maximum * 1.15 if maximum > 0 else 1.0
        for index, (name, color) in enumerate(zip(names, colors)):
            x = 230 + index * 300
            bar_width = max(1, int(320 * metrics[name][key] / scale_max))
            y = top + 42
            draw.rectangle((x, y, x + bar_width, y + 34), fill=color)
            draw.text((x, y + 42), f"{name}: {metrics[name][key]:.6f}", fill="black")
    canvas.save(path)


def _save_history_graphs(output_dir: Path) -> None:
    original_history_path = OUTPUT_ROOT / "training_history.json"
    if original_history_path.is_file():
        history = json.loads(original_history_path.read_text(encoding="utf-8"))
        if history.get("epoch") and len(history.get("epoch", [])) == len(history.get("train_loss", [])) == len(history.get("val_loss", [])):
            plot_history(history, output_dir / "existing_loss_history.png")
            plot_single_loss(history, "train_loss", "Existing AE training MSE", "#1f77b4", output_dir / "existing_training_loss.png")
            plot_single_loss(history, "val_loss", "Existing AE validation MSE", "#d62728", output_dir / "existing_validation_loss.png")
    if IMPROVED_HISTORY.is_file():
        history = json.loads(IMPROVED_HISTORY.read_text(encoding="utf-8"))
        if history.get("epoch") and len(history.get("epoch", [])) == len(history.get("train_loss", [])) == len(history.get("val_loss", [])):
            plot_history(history, output_dir / "improved_loss_history.png")
            plot_single_loss(history, "train_loss", "Improved AE training combined loss", "#1f77b4", output_dir / "improved_training_loss.png")
            plot_single_loss(history, "val_loss", "Improved AE validation combined loss", "#d62728", output_dir / "improved_validation_loss.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--original-checkpoint", type=Path, default=ORIGINAL_CHECKPOINT)
    parser.add_argument("--improved-checkpoint", type=Path, default=IMPROVED_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=COMPARISON_DIR)
    parser.add_argument("--grid-count", type=int, default=24)
    args = parser.parse_args()
    if args.grid_count < 1:
        raise ValueError("--grid-count must be positive.")
    for checkpoint in (args.original_checkpoint, args.improved_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Required checkpoint is missing: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_model = load_original_model(args.original_checkpoint, device)
    improved_model = _load_improved(args.improved_checkpoint, device)
    dataset = InteriorImageDataset(args.data_root, "test", image_size=128)
    if not dataset:
        raise ValueError("The supplied held-out test split is empty.")

    output_dir = args.output_dir.resolve()
    originals_dir = output_dir / "originals"
    original_recon_dir = output_dir / "existing_autoencoder"
    improved_recon_dir = output_dir / "improved_autoencoder"
    for directory in (originals_dir, original_recon_dir, improved_recon_dir):
        directory.mkdir(parents=True, exist_ok=True)

    metric_values = {"Existing Autoencoder": {"mse": [], "ssim": []},
                     "Improved Autoencoder": {"mse": [], "ssim": []}}
    inference_seconds = {"Existing Autoencoder": 0.0, "Improved Autoencoder": 0.0}
    rows: list[dict] = []
    grid_rows: list[tuple[Image.Image, Image.Image, Image.Image]] = []
    with torch.inference_mode():
        for index, source_path in enumerate(dataset.paths):
            with Image.open(source_path) as opened:
                original = opened.convert("RGB")
                original.load()
            legacy_input = original.resize((128, 128), Image.Resampling.BILINEAR)
            new_input, content_box = image_to_tensor(original, 128)

            old_tensor = torch.from_numpy(
                np.asarray(legacy_input, dtype=np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            old_reconstruction = original_model(old_tensor).squeeze(0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds["Existing Autoencoder"] += time.perf_counter() - started
            old_image = tensor_to_image(old_reconstruction).resize(original.size, Image.Resampling.BILINEAR)

            new_tensor = new_input.unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            new_reconstruction = improved_model(new_tensor).squeeze(0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds["Improved Autoencoder"] += time.perf_counter() - started
            new_image = restore_image(new_reconstruction, content_box, original.size)

            old_mse, old_ssim = _reconstruction_metrics(old_image, original)
            new_mse, new_ssim = _reconstruction_metrics(new_image, original)
            metric_values["Existing Autoencoder"]["mse"].append(old_mse)
            metric_values["Existing Autoencoder"]["ssim"].append(old_ssim)
            metric_values["Improved Autoencoder"]["mse"].append(new_mse)
            metric_values["Improved Autoencoder"]["ssim"].append(new_ssim)

            filename = f"{index + 1:04d}.png"
            original.save(originals_dir / filename)
            old_image.save(original_recon_dir / filename)
            new_image.save(improved_recon_dir / filename)
            rows.append(
                {
                    "index": index + 1,
                    "source_path": str(source_path),
                    "width": original.width,
                    "height": original.height,
                    "existing_mse": old_mse,
                    "existing_ssim": old_ssim,
                    "improved_mse": new_mse,
                    "improved_ssim": new_ssim,
                }
            )
            if len(grid_rows) < args.grid_count:
                grid_rows.append((original.copy(), old_image.copy(), new_image.copy()))
            if (index + 1) % 100 == 0:
                print(f"Evaluated {index + 1}/{len(dataset)} test images.", flush=True)

    summary = {}
    for name, values in metric_values.items():
        summary[name] = {
            "mse": float(np.mean(values["mse"])),
            "ssim": float(np.mean(values["ssim"])),
            "inference_time_seconds": inference_seconds[name],
            "mean_inference_time_ms_per_image": inference_seconds[name] * 1000 / len(dataset),
        }
    summary_payload = {
        "test_images": len(dataset),
        "device": str(device),
        "metric_resolution": f"{METRIC_SIZE}x{METRIC_SIZE}, aspect-preserving fit with black padding",
        "metrics": summary,
        "notes": [
            "The original checkpoint uses its original square-stretch preprocessing.",
            "The improved checkpoint uses aspect-preserving letterbox preprocessing.",
            "Both restored reconstructions are compared against the same source test images.",
            "SSIM and MSE describe reconstruction fidelity, not perceptual sharpness by themselves.",
        ],
    }
    with (output_dir / "metrics_per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    with (output_dir / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("model", "test_images", "mse", "ssim",
                                                    "inference_time_seconds", "mean_inference_time_ms_per_image"))
        writer.writeheader()
        for name, values in summary.items():
            writer.writerow({"model": name, "test_images": len(dataset), **values})

    _save_grid(grid_rows, output_dir / "comparison_grid.png", args.grid_count)
    _save_metric_graph(summary, output_dir / "metric_comparison.png")
    _save_history_graphs(output_dir)
    print(f"Evaluated {len(dataset)} held-out test images on {device}.")
    print(f"Comparison outputs: {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
