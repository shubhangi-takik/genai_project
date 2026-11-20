"""Load trained Conditional GAN and run structure-preserving interior generation."""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GAN_DIR = PROJECT_ROOT / "ml_models" / "gan"
AE_DIR = PROJECT_ROOT / "ml_models" / "autoencoder"

for p in (PROJECT_ROOT, GAN_DIR, AE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluation import ssim_per_image
from ml_models.gan.model import (
    ConditionalGenerator,
    extract_structural_edges,
    image_to_tensor,
    tensor_to_image,
)

CHECKPOINT_PATH = GAN_DIR / "outputs" / "best_gan.pt"
HISTORY_PATH = GAN_DIR / "outputs" / "training_history.json"
METRICS_PATH = GAN_DIR / "outputs" / "evaluation_test" / "metrics_summary.json"
RESULTS_DIR = PROJECT_ROOT / "backend" / "results" / "gan"

logger = logging.getLogger(__name__)


class GANService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator: ConditionalGenerator | None = None
        self.checkpoint_path: Path | None = None
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.reload()

    def reload(self) -> None:
        if not CHECKPOINT_PATH.is_file():
            self.generator = None
            self.checkpoint_path = None
            return
        try:
            state = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=False)
            gen = ConditionalGenerator().to(self.device)
            gen.load_state_dict(state["gen_state_dict"], strict=True)
            gen.eval()
            self.generator = gen
            self.checkpoint_path = CHECKPOINT_PATH
        except Exception:
            logger.exception("Failed to load GAN checkpoint at %s", CHECKPOINT_PATH)
            self.generator = None
            self.checkpoint_path = None

    def status(self) -> dict:
        checkpoint_exists = CHECKPOINT_PATH.is_file()
        if checkpoint_exists and self.generator is None:
            self.reload()

        return {
            "model": "Conditional GAN (Pix2Pix)",
            "available": CHECKPOINT_PATH.is_file() and self.generator is not None,
            "loaded": self.generator is not None,
            "device": str(self.device),
            "checkpoint": str(CHECKPOINT_PATH) if checkpoint_exists else None,
            "architecture": "U-Net Generator + PatchGAN Discriminator",
        }

    def decode_upload(self, data: bytes) -> Image.Image:
        if not data:
            raise ValueError("No image data provided.")
        try:
            img = Image.open(BytesIO(data))
            img.load()
            return img.convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError("Uploaded file is not a valid image.") from exc

    def predict(self, image_bytes: bytes, filename: str | None) -> dict:
        if self.generator is None:
            self.reload()
        if self.generator is None:
            raise FileNotFoundError("GAN checkpoint is missing. Train the model first to generate best_gan.pt.")

        orig_img = self.decode_upload(image_bytes)
        # Extract structural room condition
        cond_img = extract_structural_edges(orig_img, image_size=128)
        cond_tensor = image_to_tensor(cond_img, image_size=128).unsqueeze(0).to(self.device)
        real_tensor = image_to_tensor(orig_img, image_size=128).unsqueeze(0).to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            generated_tensor = self.generator(cond_tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) * 1000.0

        # Calculate live comparison metrics
        mse = F.mse_loss(generated_tensor, real_tensor).item()
        psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
        ssim = float(ssim_per_image(generated_tensor, real_tensor).mean().item())

        # Load global test metrics if available
        fid_val = None
        is_val = None
        g_loss_val = None
        d_loss_val = None
        if METRICS_PATH.is_file():
            try:
                metrics_data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
                fid_val = metrics_data.get("fid")
                is_val = metrics_data.get("inception_score")
            except Exception:
                pass

        if HISTORY_PATH.is_file():
            try:
                hist_data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
                g_loss_val = hist_data.get("g_loss", [None])[-1]
                d_loss_val = hist_data.get("d_loss", [None])[-1]
            except Exception:
                pass

        # Save results
        req_id = uuid.uuid4().hex
        orig_name = f"{req_id}_original.png"
        cond_name = f"{req_id}_condition.png"
        gen_name = f"{req_id}_generated.png"

        orig_img.resize((128, 128), Image.Resampling.BILINEAR).save(RESULTS_DIR / orig_name)
        cond_img.save(RESULTS_DIR / cond_name)
        tensor_to_image(generated_tensor.squeeze(0)).save(RESULTS_DIR / gen_name)

        return {
            "model": "GAN",
            "input_image": f"/results/gan/{orig_name}",
            "condition_image": f"/results/gan/{cond_name}",
            "output_image": f"/results/gan/{gen_name}",
            "metrics": {
                "fid": fid_val,
                "inception_score": is_val,
                "ssim": round(ssim, 4),
                "psnr": round(psnr, 2),
                "generator_loss": round(g_loss_val, 4) if g_loss_val else None,
                "discriminator_loss": round(d_loss_val, 4) if d_loss_val else None,
                "inference_time_ms": round(infer_ms, 2),
            },
        }

    def training_history(self) -> list[dict]:
        if not HISTORY_PATH.is_file():
            return []
        try:
            data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
            epochs = data.get("epoch", [])
            g_loss = data.get("g_loss", [])
            d_loss = data.get("d_loss", [])
            val_g = data.get("val_g_loss", [])
            val_d = data.get("val_d_loss", [])

            rows = []
            for i, ep in enumerate(epochs):
                rows.append({
                    "name": str(ep),
                    "generator_loss": g_loss[i] if i < len(g_loss) else None,
                    "discriminator_loss": d_loss[i] if i < len(d_loss) else None,
                    "val_g_loss": val_g[i] if i < len(val_g) else None,
                    "val_d_loss": val_d[i] if i < len(val_d) else None,
                })
            return rows
        except Exception:
            return []


service = GANService()
