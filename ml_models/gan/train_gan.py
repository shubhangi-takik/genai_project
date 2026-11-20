"""Train the Conditional GAN (Pix2Pix) for interior design synthesis."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_models.gan.dataset import DEFAULT_DATA, PairedInteriorDataset
from ml_models.gan.model import ConditionalGenerator, PatchGANDiscriminator

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def plot_series(series_dict: dict[str, list[float]], colors: dict[str, str], title: str, output: Path) -> None:
    canvas = Image.new("RGB", (900, 550), "white")
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = 80, 55, 860, 470
    draw.line((left, top, left, bottom, right, bottom), fill="black", width=2)
    draw.text((left, 20), title, fill="black")

    all_vals = [v for s in series_dict.values() for v in s]
    if not all_vals:
        return
    low, high = min(all_vals), max(all_vals)
    if low == high:
        high = low + 1e-8

    x_span = right - left
    y_span = bottom - top

    legend_items = []
    for name, series in series_dict.items():
        color = colors.get(name, "black")
        legend_items.append(f"{name}: {color}")
        if len(series) > 1:
            pts = [
                (
                    left + idx * x_span / max(1, len(series) - 1),
                    bottom - (val - low) * y_span / (high - low)
                )
                for idx, val in enumerate(series)
            ]
            draw.line(pts, fill=color, width=3)
        elif len(series) == 1:
            cx = left + x_span / 2
            cy = bottom - (series[0] - low) * y_span / (high - low)
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=color)

    draw.text((left, 495), " | ".join(legend_items), fill="black")
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = PairedInteriorDataset(args.data_root, split="train", image_size=128)
    val_dataset = PairedInteriorDataset(args.data_root, split="val", image_size=128)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    gen = ConditionalGenerator().to(device)
    disc = PatchGANDiscriminator().to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    history = {
        "epoch": [],
        "g_loss": [],
        "d_loss": [],
        "l1_loss": [],
        "val_g_loss": [],
        "val_d_loss": [],
        "val_l1_loss": [],
        "device": str(device),
        "lambda_l1": args.lambda_l1,
    }

    checkpoint_path = output_dir / "best_gan.pt"
    started = time.perf_counter()
    best_val_loss = float("inf")

    print(f"Starting GAN training on {device} ({len(train_dataset)} train, {len(val_dataset)} val)...", flush=True)

    for epoch in range(1, args.epochs + 1):
        gen.train()
        disc.train()
        g_loss_sum, d_loss_sum, l1_loss_sum = 0.0, 0.0, 0.0
        n_batches = 0

        for cond, real in train_loader:
            cond, real = cond.to(device), real.to(device)

            # --- Train Discriminator ---
            opt_d.zero_grad(set_to_none=True)
            fake = gen(cond)

            pred_real = disc(cond, real)
            loss_d_real = criterion_bce(pred_real, torch.ones_like(pred_real))

            pred_fake = disc(cond, fake.detach())
            loss_d_fake = criterion_bce(pred_fake, torch.zeros_like(pred_fake))

            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            loss_d.backward()
            opt_d.step()

            # --- Train Generator ---
            opt_g.zero_grad(set_to_none=True)
            pred_fake_for_g = disc(cond, fake)
            loss_g_adv = criterion_bce(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
            loss_g_l1 = criterion_l1(fake, real) * args.lambda_l1
            loss_g = loss_g_adv + loss_g_l1
            loss_g.backward()
            opt_g.step()

            g_loss_sum += loss_g.item()
            d_loss_sum += loss_d.item()
            l1_loss_sum += (loss_g_l1 / args.lambda_l1).item()
            n_batches += 1

        avg_g = g_loss_sum / n_batches
        avg_d = d_loss_sum / n_batches
        avg_l1 = l1_loss_sum / n_batches

        # Validation pass
        gen.eval()
        disc.eval()
        val_g_sum, val_d_sum, val_l1_sum = 0.0, 0.0, 0.0
        val_batches = 0

        with torch.no_grad():
            for cond, real in val_loader:
                cond, real = cond.to(device), real.to(device)
                fake = gen(cond)

                pred_real = disc(cond, real)
                loss_d_real = criterion_bce(pred_real, torch.ones_like(pred_real))
                pred_fake = disc(cond, fake)
                loss_d_fake = criterion_bce(pred_fake, torch.zeros_like(pred_fake))
                loss_d = 0.5 * (loss_d_real + loss_d_fake)

                pred_fake_for_g = disc(cond, fake)
                loss_g_adv = criterion_bce(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
                loss_g_l1 = criterion_l1(fake, real) * args.lambda_l1
                loss_g = loss_g_adv + loss_g_l1

                val_g_sum += loss_g.item()
                val_d_sum += loss_d.item()
                val_l1_sum += (loss_g_l1 / args.lambda_l1).item()
                val_batches += 1

        avg_val_g = val_g_sum / val_batches
        avg_val_d = val_d_sum / val_batches
        avg_val_l1 = val_l1_sum / val_batches

        history["epoch"].append(epoch)
        history["g_loss"].append(avg_g)
        history["d_loss"].append(avg_d)
        history["l1_loss"].append(avg_l1)
        history["val_g_loss"].append(avg_val_g)
        history["val_d_loss"].append(avg_val_d)
        history["val_l1_loss"].append(avg_val_l1)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d}: "
            f"Train [G: {avg_g:.4f}, D: {avg_d:.4f}, L1: {avg_l1:.4f}] | "
            f"Val [G: {avg_val_g:.4f}, D: {avg_val_d:.4f}, L1: {avg_val_l1:.4f}]",
            flush=True
        )

        # Save checkpoint
        if avg_val_g < best_val_loss:
            best_val_loss = avg_val_g
            torch.save(
                {
                    "gen_state_dict": gen.state_dict(),
                    "disc_state_dict": disc.state_dict(),
                    "epoch": epoch,
                    "val_g_loss": avg_val_g,
                    "val_d_loss": avg_val_d,
                    "image_size": 128,
                    "architecture": "conditional_pix2pix_v1"
                },
                checkpoint_path
            )

        # Plot graphs
        (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        plot_series(
            {"Train G Loss": history["g_loss"], "Val G Loss": history["val_g_loss"]},
            {"Train G Loss": "#3b82f6", "Val G Loss": "#8b5cf6"},
            "Generator Loss",
            output_dir / "generator_loss.png"
        )
        plot_series(
            {"Train D Loss": history["d_loss"], "Val D Loss": history["val_d_loss"]},
            {"Train D Loss": "#ef4444", "Val D Loss": "#f97316"},
            "Discriminator Loss",
            output_dir / "discriminator_loss.png"
        )

    elapsed = time.perf_counter() - started
    print(f"GAN training complete in {elapsed:.2f}s. Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
