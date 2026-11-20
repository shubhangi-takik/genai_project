"""Train the VAE on supplied train/validation CSV splits; never uses test data."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AE_DIR = PROJECT_ROOT / "ml_models" / "autoencoder"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(AE_DIR) not in sys.path:
    sys.path.insert(0, str(AE_DIR))

from ml_models.vae.model import ConvVAE  # noqa: E402
from train import DEFAULT_DATA, InteriorImageDataset  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def loss_components(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = target.shape[0]
    reconstruction_loss = F.binary_cross_entropy(reconstruction, target, reduction="sum") / batch_size
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp()) / batch_size
    total_loss = reconstruction_loss + beta * kl_divergence
    return total_loss, reconstruction_loss, kl_divergence


def run_epoch(
    model: ConvVAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    totals = {"total": 0.0, "reconstruction": 0.0, "kl": 0.0}
    seen = 0
    with torch.set_grad_enabled(is_training):
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            reconstruction, mu, logvar = model(batch)
            total, reconstruction_loss, kl_divergence = loss_components(
                reconstruction, batch, mu, logvar, beta
            )
            if optimizer is not None:
                total.backward()
                optimizer.step()
            count = batch.shape[0]
            totals["total"] += total.item() * count
            totals["reconstruction"] += reconstruction_loss.item() * count
            totals["kl"] += kl_divergence.item() * count
            seen += count
    if seen == 0:
        raise ValueError("Cannot run an epoch on an empty data split.")
    return {key: value / seen for key, value in totals.items()}


def plot_train_validation(
    history: dict[str, list[float]], train_key: str, val_key: str, title: str, output: Path
) -> None:
    train_values = history[train_key]
    val_values = history[val_key]
    values = train_values + val_values
    low, high = min(values), max(values)
    if low == high:
        high = low + 1e-8

    canvas = Image.new("RGB", (900, 550), "white")
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = 80, 55, 860, 470
    draw.line((left, top, left, bottom, right, bottom), fill="black", width=2)
    draw.text((left, 20), title, fill="black")
    x_span = right - left
    y_span = bottom - top

    def points(series: list[float]) -> list[tuple[float, float]]:
        return [
            (
                left + index * x_span / max(1, len(series) - 1),
                bottom - (value - low) * y_span / (high - low),
            )
            for index, value in enumerate(series)
        ]

    if len(train_values) > 1:
        draw.line(points(train_values), fill="#1f77b4", width=3)
    else:
        x, y = points(train_values)[0]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#1f77b4")
    if len(val_values) > 1:
        draw.line(points(val_values), fill="#d62728", width=3)
    else:
        x, y = points(val_values)[0]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#d62728")
    draw.text((left, 500), "Blue: training     Red: validation", fill="black")
    canvas.save(output)


def save_training_artifacts(history: dict, output_dir: Path) -> None:
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    fields = (
        "epoch",
        "beta",
        "train_total_loss",
        "val_total_loss",
        "train_reconstruction_loss",
        "val_reconstruction_loss",
        "train_kl_divergence",
        "val_kl_divergence",
    )
    with (output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, epoch in enumerate(history["epoch"]):
            writer.writerow({key: history[key][index] if key != "epoch" else epoch for key in fields})

    plot_train_validation(history, "train_total_loss", "val_total_loss", "Total VAE loss", output_dir / "total_loss.png")
    plot_train_validation(
        history,
        "train_reconstruction_loss",
        "val_reconstruction_loss",
        "Binary cross-entropy reconstruction loss",
        output_dir / "reconstruction_loss.png",
    )
    plot_train_validation(
        history,
        "train_kl_divergence",
        "val_kl_divergence",
        "KL divergence",
        output_dir / "kl_divergence_loss.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--beta-warmup-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.latent_dim < 1:
        raise ValueError("epochs, batch-size, and latent-dim must be positive.")
    if args.base_channels < 1 or args.patience < 1 or args.workers < 0:
        raise ValueError("base-channels and patience must be positive; workers cannot be negative.")
    if args.beta < 0 or args.beta_warmup_epochs < 0:
        raise ValueError("beta and beta-warmup-epochs cannot be negative.")
    if args.image_size < 32 or args.image_size % 16:
        raise ValueError("--image-size must be at least 32 and divisible by 16.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch installation.")

    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_set = InteriorImageDataset(args.data_root, "train", args.image_size)
    val_set = InteriorImageDataset(args.data_root, "val", args.image_size)
    if not train_set or not val_set:
        raise ValueError("The supplied train and validation splits must both be non-empty.")
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = ConvVAE(args.latent_dim, args.image_size, args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    checkpoint_path = output_dir / "best_vae.pt"
    history: dict[str, list[float] | str | int | float] = {
        "epoch": [],
        "beta": [],
        "train_total_loss": [],
        "val_total_loss": [],
        "train_reconstruction_loss": [],
        "val_reconstruction_loss": [],
        "train_kl_divergence": [],
        "val_kl_divergence": [],
        "device": str(device),
        "image_size": args.image_size,
        "latent_dim": args.latent_dim,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "beta_max": args.beta,
        "beta_warmup_epochs": args.beta_warmup_epochs,
        "loss_definition": "BCE sum per image + beta * KL sum per image",
    }
    best_val_loss = float("inf")
    waiting = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        beta = args.beta if args.beta_warmup_epochs == 0 else args.beta * min(
            1.0, epoch / args.beta_warmup_epochs
        )
        train_metrics = run_epoch(model, train_loader, device, beta, optimizer)
        # Keep checkpoint selection comparable while beta is being annealed for training.
        val_metrics = run_epoch(model, val_loader, device, args.beta, None)
        history["epoch"].append(epoch)
        history["beta"].append(beta)
        for key, value in train_metrics.items():
            history[f"train_{'kl_divergence' if key == 'kl' else key + '_loss'}"].append(value)
        for key, value in val_metrics.items():
            history[f"val_{'kl_divergence' if key == 'kl' else key + '_loss'}"].append(value)
        history["elapsed_seconds"] = time.perf_counter() - started

        print(
            f"Epoch {epoch:03d}/{args.epochs} beta={beta:.4f}: "
            f"train={train_metrics['total']:.4f} (recon={train_metrics['reconstruction']:.4f}, "
            f"kl={train_metrics['kl']:.4f}) val={val_metrics['total']:.4f} "
            f"(recon={val_metrics['reconstruction']:.4f}, kl={val_metrics['kl']:.4f})",
            flush=True,
        )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            waiting = 0
            temporary_path = checkpoint_path.with_suffix(".tmp")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": "conv_vae_v1",
                    "latent_dim": args.latent_dim,
                    "image_size": args.image_size,
                    "base_channels": args.base_channels,
                    "beta": args.beta,
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                },
                temporary_path,
            )
            temporary_path.replace(checkpoint_path)
        elif epoch >= args.beta_warmup_epochs:
            waiting += 1
        else:
            waiting = 0

        save_training_artifacts(history, output_dir)
        if epoch >= args.beta_warmup_epochs and waiting >= args.patience:
            print(f"Early stopping at epoch {epoch}; best validation loss: {best_val_loss:.4f}")
            break

    history["elapsed_seconds"] = time.perf_counter() - started
    save_training_artifacts(history, output_dir)
    print(f"Best VAE checkpoint: {checkpoint_path}")
    print(f"Training artifacts: {output_dir}")
    print(f"Training elapsed: {history['elapsed_seconds']:.2f}s on {device}")


if __name__ == "__main__":
    main()
