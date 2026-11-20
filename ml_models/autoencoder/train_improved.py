"""Train the skip-connected Autoencoder using only supplied train and validation splits."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from evaluation import plot_history, plot_single_loss, ssim_per_image
from model_improved import ImprovedAutoencoder, image_to_tensor
from train import DEFAULT_DATA, InteriorImageDataset

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "improved_autoencoder"


class AspectPreservingDataset(Dataset):
    def __init__(self, data_root: Path, split: str, image_size: int) -> None:
        source = InteriorImageDataset(data_root, split, image_size)
        self.paths = source.paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            tensor, _ = image_to_tensor(image, self.image_size)
        return tensor


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pixel_loss = nn.functional.mse_loss(prediction, target)
    structural_loss = 1.0 - ssim_per_image(prediction, target).mean()
    return 0.8 * pixel_loss + 0.2 * structural_loss


def run_epoch(
    model: ImprovedAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    image_count = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = reconstruction_loss(model(batch), batch)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.size(0)
            image_count += batch.size(0)
    if image_count == 0:
        raise ValueError("Cannot run an epoch with an empty data split.")
    return total_loss / image_count


def save_history(history: dict, output_dir: Path) -> None:
    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    if history["epoch"]:
        plot_history(history, output_dir / "loss_history.png")
        plot_single_loss(history, "train_loss", "Training combined reconstruction loss",
                         "#1f77b4", output_dir / "training_loss.png")
        plot_single_loss(history, "val_loss", "Validation combined reconstruction loss",
                         "#d62728", output_dir / "validation_loss.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.workers < 0:
        raise ValueError("epochs, batch-size, and patience must be positive; workers cannot be negative.")
    if args.image_size < 32 or args.image_size % 16:
        raise ValueError("--image-size must be at least 32 and divisible by 16.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch installation has no available CUDA device.")

    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = AspectPreservingDataset(args.data_root, "train", args.image_size)
    val_dataset = AspectPreservingDataset(args.data_root, "val", args.image_size)
    if not train_dataset or not val_dataset:
        raise ValueError("Both supplied train and validation splits must contain images.")
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = ImprovedAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    checkpoint_path = output_dir / "best_autoencoder_improved.pt"
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "loss_definition": "0.8 * MSE + 0.2 * (1 - SSIM)",
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(device),
        "early_stopping_patience": args.patience,
    }
    best_loss = float("inf")
    epochs_without_improvement = 0
    started_at = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        val_loss = run_epoch(model, val_loader, None, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["training_time_seconds"] = time.perf_counter() - started_at
        print(f"Epoch {epoch:03d}/{args.epochs}: train={train_loss:.7f} val={val_loss:.7f}", flush=True)

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
            temporary_path = checkpoint_path.with_suffix(".tmp")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "base_channels": 32,
                    "image_size": args.image_size,
                    "best_val_loss": best_loss,
                    "epoch": epoch,
                    "architecture": "improved_skip_autoencoder_v1",
                },
                temporary_path,
            )
            temporary_path.replace(checkpoint_path)
        else:
            epochs_without_improvement += 1

        save_history(history, output_dir)
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs; best validation loss: {best_loss:.7f}")
            break

    history["training_time_seconds"] = time.perf_counter() - started_at
    save_history(history, output_dir)
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training history and graphs: {output_dir}")
    print(f"Elapsed training time: {history['training_time_seconds']:.2f}s on {device}")


if __name__ == "__main__":
    main()
