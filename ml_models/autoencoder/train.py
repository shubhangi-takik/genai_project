"""Train the Phase 2 PyTorch convolutional autoencoder."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import Autoencoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "Pinterest Interior Design Images and Metadata"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


class InteriorImageDataset(Dataset):
    """Uses a supplied CSV split; labels are intentionally ignored for reconstruction."""

    def __init__(self, data_root: str | Path, split: str, image_size: int = 128) -> None:
        self.root = Path(data_root)
        self.image_size = image_size
        csv_path = self.root / f"{split}_data.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing supplied split file: {csv_path}")
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.paths = [self._resolve(row["image_path"]) for row in rows]
        missing = [p for p in self.paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images from {csv_path.name} are missing; e.g. {missing[0]}")

    def _resolve(self, csv_value: str) -> Path:
        direct = (self.root / csv_value).resolve()
        if direct.is_file():
            return direct
        # Archive CSVs use ../data/raw/<room>/<style>/<file>; archive stores <room>/<style>/<file>.
        parts = Path(csv_value.replace("\\", "/")).parts
        return self.root.joinpath(*parts[-3:])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            rgb = image.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)


def build_loader(data_root: str | Path, split: str, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(InteriorImageDataset(data_root, split), batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=torch.cuda.is_available())


def mean_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total, images = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            total += criterion(model(batch), batch).item() * batch.size(0)
            images += batch.size(0)
    return total / images


def plot_history(history: dict, path: Path) -> None:
    """Render loss curves without an additional plotting dependency."""
    canvas = Image.new("RGB", (900, 550), "white"); draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = 80, 45, 860, 480
    values = history["train_loss"] + history["val_loss"]
    low, high = min(values), max(values)
    if high == low: high = low + 1e-8
    draw.line((left, top, left, bottom, right, bottom), fill="black", width=2)
    def points(series: list[float]) -> list[tuple[float, float]]:
        return [(left + i * (right - left) / max(1, len(series) - 1),
                 bottom - (v - low) * (bottom - top) / (high - low)) for i, v in enumerate(series)]
    draw.line(points(history["train_loss"]), fill="#1f77b4", width=3)
    draw.line(points(history["val_loss"]), fill="#d62728", width=3)
    draw.text((left, 12), "MSE reconstruction loss", fill="black")
    draw.text((left, 500), "blue: training loss     red: validation loss", fill="black")
    canvas.save(path)


def plot_single_loss(history: dict, key: str, label: str, color: str, path: Path) -> None:
    series = history[key]
    canvas = Image.new("RGB", (900, 550), "white"); draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = 80, 45, 860, 480
    low, high = min(series), max(series)
    if high == low: high = low + 1e-8
    points = [(left + i * (right - left) / max(1, len(series) - 1),
               bottom - (value - low) * (bottom - top) / (high - low)) for i, value in enumerate(series)]
    draw.line((left, top, left, bottom, right, bottom), fill="black", width=2)
    draw.line(points, fill=color, width=3); draw.text((left, 12), label, fill="black")
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0, help="Use 0 on Windows unless worker spawning is configured.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1: raise ValueError("--epochs must be at least 1")
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(exist_ok=True)
    train_loader = build_loader(args.data_root, "train", args.batch_size, args.workers, True)
    val_loader = build_loader(args.data_root, "val", args.batch_size, args.workers, False)
    model = Autoencoder().to(device); optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss(); history = {"epoch": [], "train_loss": [], "val_loss": []}
    training_start = time.perf_counter()
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); total, images = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch)
            loss.backward(); optimizer.step()
            total += loss.item() * batch.size(0); images += batch.size(0)
        train_loss = total / images; val_loss = mean_loss(model, val_loader, criterion, device)
        history["epoch"].append(epoch); history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:03d}/{args.epochs}: train={train_loss:.7f} val={val_loss:.7f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(), "latent_channels": 256, "image_size": 128,
                        "best_val_loss": best_loss, "epoch": epoch}, OUTPUT_DIR / "best_autoencoder.pt")
    elapsed = time.perf_counter() - training_start
    history["training_time_seconds"] = elapsed; history["device"] = str(device)
    (OUTPUT_DIR / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_history(history, OUTPUT_DIR / "loss_history.png")
    plot_single_loss(history, "train_loss", "Training loss", "#1f77b4", OUTPUT_DIR / "training_loss.png")
    plot_single_loss(history, "val_loss", "Validation loss", "#d62728", OUTPUT_DIR / "validation_loss.png")
    print(f"Training complete in {elapsed:.2f}s. Best validation loss: {best_loss:.7f}")


if __name__ == "__main__":
    main()
