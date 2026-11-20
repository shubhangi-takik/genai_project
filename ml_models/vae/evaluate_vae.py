"""Evaluate VAE reconstructions and prior samples on the held-out test split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AE_DIR = PROJECT_ROOT / "ml_models" / "autoencoder"
if str(AE_DIR) not in sys.path:
    sys.path.insert(0, str(AE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import ssim_per_image  # noqa: E402
from inference import load_model as load_autoencoder  # noqa: E402
from ml_models.vae.model import ConvVAE, tensor_to_image  # noqa: E402
from ml_models.vae.train_vae import DEFAULT_DATA  # noqa: E402
from train import InteriorImageDataset  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def load_vae(checkpoint_path: Path, device: torch.device) -> tuple[ConvVAE, dict]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("architecture") != "conv_vae_v1":
        raise ValueError(f"Unsupported VAE checkpoint architecture: {checkpoint.get('architecture')}")
    model = ConvVAE(
        latent_dim=checkpoint["latent_dim"],
        image_size=checkpoint["image_size"],
        base_channels=checkpoint.get("base_channels", 32),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def save_grid(
    entries: list[tuple[Image.Image, Image.Image, Image.Image]],
    output_path: Path,
    count: int,
    has_autoencoder: bool,
) -> None:
    selected = entries[:count]
    if not selected:
        raise ValueError("No test images were available for the comparison grid.")
    cell_size = 128
    label_height = 22
    columns = 3 if has_autoencoder else 2
    rows = len(selected)
    grid = Image.new("RGB", (columns * cell_size, rows * (cell_size + label_height)), "white")
    draw = ImageDraw.Draw(grid)
    labels = (
        ("Original", "VAE reconstruction", "Autoencoder reconstruction")
        if has_autoencoder
        else ("Original", "VAE reconstruction")
    )
    for row, images in enumerate(selected):
        y = row * (cell_size + label_height)
        for column, (image, label) in enumerate(zip(images[:columns], labels)):
            x = column * cell_size
            draw.text((x + 3, y + 4), label, fill="black")
            grid.paste(image.resize((cell_size, cell_size), Image.Resampling.BILINEAR), (x, y + label_height))
    grid.save(output_path)


def save_sample_grid(images: list[Image.Image], output_path: Path, columns: int = 4) -> None:
    if not images:
        raise ValueError("No latent samples were generated.")
    cell_size = 128
    label_height = 18
    rows = math.ceil(len(images) / columns)
    grid = Image.new("RGB", (columns * cell_size, rows * (cell_size + label_height)), "white")
    draw = ImageDraw.Draw(grid)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        x, y = column * cell_size, row * (cell_size + label_height)
        draw.text((x + 3, y + 3), f"Prior sample {index + 1}", fill="black")
        grid.paste(image.resize((cell_size, cell_size), Image.Resampling.BILINEAR), (x, y + label_height))
    grid.save(output_path)


def _write_report(
    output_path: Path,
    count: int,
    device: torch.device,
    vae_metrics: dict[str, float],
    autoencoder_metrics: dict[str, float] | None,
    generated_count: int,
    seed: int,
    checkpoint: Path,
) -> None:
    lines = [
        "# VAE evaluation",
        "",
        f"- Held-out split: test ({count} images)",
        f"- Device: {device}",
        f"- VAE checkpoint: `{checkpoint}`",
        "- Reconstruction uses the posterior mean for deterministic comparisons.",
        f"- Prior samples: {generated_count}, sampled from N(0, I) with seed {seed}.",
        "",
        "## Reconstruction metrics",
        "",
        "| Model | MSE | MAE | SSIM | Mean inference (ms/image) |",
        "|---|---:|---:|---:|---:|",
        (
            f"| VAE | {vae_metrics['mse']:.8f} | {vae_metrics['mae']:.8f} | "
            f"{vae_metrics['ssim']:.6f} | {vae_metrics['mean_inference_ms']:.3f} |"
        ),
    ]
    if autoencoder_metrics is not None:
        lines.append(
            f"| Existing Autoencoder | {autoencoder_metrics['mse']:.8f} | "
            f"{autoencoder_metrics['mae']:.8f} | {autoencoder_metrics['ssim']:.6f} | "
            f"{autoencoder_metrics['mean_inference_ms']:.3f} |"
        )
    lines.extend(
        (
            "",
            "MSE and SSIM quantify reconstruction fidelity only. They do not measure the quality, diversity, or realism of newly sampled prior images. Review the saved prior-sample grid separately; these samples are unconditional and are not controlled by prompts or room/style labels.",
            "",
            "The test split was used only for final evaluation, not for training or checkpoint selection.",
        )
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=OUTPUT_DIR / "best_vae.pt",
    )
    parser.add_argument(
        "--existing-checkpoint",
        type=Path,
        default=AE_DIR / "outputs" / "best_autoencoder.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "evaluation_test",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generated-count", type=int, default=16)
    parser.add_argument("--grid-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or args.generated_count < 1 or args.grid_count < 1:
        raise ValueError("batch-size, generated-count, and grid-count must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch installation.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.manual_seed(args.seed)

    vae, checkpoint = load_vae(args.checkpoint, device)
    if args.existing_checkpoint.is_file():
        autoencoder = load_autoencoder(args.existing_checkpoint, device)
    else:
        autoencoder = None
        print(f"Existing Autoencoder checkpoint not found; baseline comparison skipped: {args.existing_checkpoint}")

    dataset = InteriorImageDataset(args.data_root, "test", checkpoint["image_size"])
    if len(dataset) == 0:
        raise ValueError("The supplied test split is empty.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    output_dir = args.output_dir.resolve()
    originals_dir = output_dir / "originals"
    vae_dir = output_dir / "vae_reconstructions"
    autoencoder_dir = output_dir / "autoencoder_reconstructions"
    prior_dir = output_dir / "prior_samples"
    for directory in (originals_dir, vae_dir, prior_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if autoencoder is not None:
        autoencoder_dir.mkdir(parents=True, exist_ok=True)

    sums = {
        name: {
            "mse": 0.0,
            "mae": 0.0,
            "ssim": 0.0,
            "kl_divergence": 0.0,
            "reconstruction_loss": 0.0,
            "total_loss": 0.0,
            "seconds": 0.0,
        }
        for name in (("VAE", "Existing Autoencoder") if autoencoder is not None else ("VAE",))
    }
    rows: list[dict] = []
    comparison_entries: list[tuple[Image.Image, Image.Image, Image.Image]] = []
    sample_number = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            batch = batch.to(device, non_blocking=device.type == "cuda")
            start = time.perf_counter()
            vae_output, mu, logvar = vae.reconstruct(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            vae_seconds = time.perf_counter() - start
            vae_mse = F.mse_loss(vae_output, batch, reduction="none").flatten(1).mean(1)
            vae_mae = F.l1_loss(vae_output, batch, reduction="none").flatten(1).mean(1)
            vae_ssim = ssim_per_image(vae_output, batch)
            kl_values = -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp(), dim=1)
            per_image_bce = F.binary_cross_entropy(
                vae_output, batch, reduction="none"
            ).flatten(1).sum(1)
            per_image_total = per_image_bce + float(checkpoint.get("beta", 1.0)) * kl_values
            sums["VAE"]["mse"] += vae_mse.sum().item()
            sums["VAE"]["mae"] += vae_mae.sum().item()
            sums["VAE"]["ssim"] += vae_ssim.sum().item()
            sums["VAE"]["kl_divergence"] += kl_values.sum().item()
            sums["VAE"]["reconstruction_loss"] += per_image_bce.sum().item()
            sums["VAE"]["total_loss"] += per_image_total.sum().item()
            sums["VAE"]["seconds"] += vae_seconds

            if autoencoder is not None:
                start = time.perf_counter()
                ae_output = autoencoder(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                ae_seconds = time.perf_counter() - start
                ae_mse = F.mse_loss(ae_output, batch, reduction="none").flatten(1).mean(1)
                ae_mae = F.l1_loss(ae_output, batch, reduction="none").flatten(1).mean(1)
                ae_ssim = ssim_per_image(ae_output, batch)
                sums["Existing Autoencoder"]["mse"] += ae_mse.sum().item()
                sums["Existing Autoencoder"]["mae"] += ae_mae.sum().item()
                sums["Existing Autoencoder"]["ssim"] += ae_ssim.sum().item()
                sums["Existing Autoencoder"]["seconds"] += ae_seconds

            for offset in range(batch.shape[0]):
                index = sample_number
                source_path = dataset.paths[index]
                original_image = tensor_to_image(batch[offset])
                vae_image = tensor_to_image(vae_output[offset])
                original_name = f"{index + 1:04d}_{source_path.stem}.png"
                original_image.save(originals_dir / original_name)
                vae_image.save(vae_dir / f"{index + 1:04d}_reconstruction.png")
                row = {
                    "index": index + 1,
                    "source_path": str(source_path),
                    "vae_mse": float(vae_mse[offset].item()),
                    "vae_mae": float(vae_mae[offset].item()),
                    "vae_ssim": float(vae_ssim[offset].item()),
                    "vae_kl_divergence": float(kl_values[offset].item()),
                    "vae_total_loss": float(per_image_total[offset].item()),
                }
                if autoencoder is not None:
                    ae_image = tensor_to_image(ae_output[offset])
                    ae_image.save(autoencoder_dir / f"{index + 1:04d}_reconstruction.png")
                    row.update(
                        {
                            "autoencoder_mse": float(ae_mse[offset].item()),
                            "autoencoder_mae": float(ae_mae[offset].item()),
                            "autoencoder_ssim": float(ae_ssim[offset].item()),
                        }
                    )
                    if len(comparison_entries) < args.grid_count:
                        comparison_entries.append((original_image.copy(), vae_image.copy(), ae_image.copy()))
                elif len(comparison_entries) < args.grid_count:
                    comparison_entries.append((original_image.copy(), vae_image.copy(), vae_image.copy()))
                rows.append(row)
                sample_number += 1
            if sample_number % 100 == 0:
                print(f"Evaluated {sample_number}/{len(dataset)} test images.", flush=True)

    summaries: dict[str, dict[str, float]] = {}
    for name, values in sums.items():
        summary = {
            "mse": values["mse"] / len(dataset),
            "mae": values["mae"] / len(dataset),
            "ssim": values["ssim"] / len(dataset),
            "inference_time_seconds": values["seconds"],
            "mean_inference_ms": values["seconds"] * 1000.0 / len(dataset),
        }
        if name == "VAE":
            summary["mean_posterior_kl_divergence"] = values["kl_divergence"] / len(dataset)
            summary["mean_bce_reconstruction_loss"] = values["reconstruction_loss"] / len(dataset)
            summary["mean_vae_total_loss"] = values["total_loss"] / len(dataset)
        summaries[name] = summary

    generated = vae.sample(args.generated_count, seed=args.seed, device=device)
    sample_images: list[Image.Image] = []
    for index, tensor in enumerate(generated):
        image = tensor_to_image(tensor)
        image.save(prior_dir / f"prior_sample_{index + 1:03d}_seed_{args.seed}.png")
        sample_images.append(image)

    with (output_dir / "metrics_per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "model",
                "test_images",
                "mse",
                "mae",
                "ssim",
                "mean_inference_ms",
                "mean_posterior_kl_divergence",
                "mean_bce_reconstruction_loss",
                "mean_vae_total_loss",
            ),
        )
        writer.writeheader()
        for name, values in summaries.items():
            writer.writerow(
                {
                    "model": name,
                    "test_images": len(dataset),
                    "mse": values["mse"],
                    "mae": values["mae"],
                    "ssim": values["ssim"],
                    "mean_inference_ms": values["mean_inference_ms"],
                    "mean_posterior_kl_divergence": values.get("mean_posterior_kl_divergence", ""),
                    "mean_bce_reconstruction_loss": values.get("mean_bce_reconstruction_loss", ""),
                    "mean_vae_total_loss": values.get("mean_vae_total_loss", ""),
                }
            )

    summary_payload = {
        "split": "test",
        "test_images": len(dataset),
        "device": str(device),
        "vae_checkpoint": str(args.checkpoint.resolve()),
        "generated_prior_samples": args.generated_count,
        "sample_seed": args.seed,
        "metrics": summaries,
        "metric_note": "MSE and SSIM quantify reconstruction fidelity only, not prior-sample realism or diversity.",
        "preprocessing": "RGB, bilinear square resize to checkpoint image_size, float values in [0, 1].",
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    save_grid(
        comparison_entries,
        output_dir / "reconstruction_comparison_grid.png",
        args.grid_count,
        autoencoder is not None,
    )
    save_sample_grid(sample_images, output_dir / "prior_samples_grid.png")
    _write_report(
        output_dir / "evaluation_report.md",
        len(dataset),
        device,
        summaries["VAE"],
        summaries.get("Existing Autoencoder"),
        args.generated_count,
        args.seed,
        args.checkpoint.resolve(),
    )
    print(f"Evaluated {len(dataset)} held-out test images on {device}.")
    print(f"VAE evaluation artifacts: {output_dir}")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
