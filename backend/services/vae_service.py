from __future__ import annotations

import logging
import secrets
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
VAE_DIR = PROJECT_ROOT / "ml_models" / "vae"
AE_DIR = PROJECT_ROOT / "ml_models" / "autoencoder"
for import_path in (PROJECT_ROOT, AE_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from evaluation import ssim_per_image  # noqa: E402
from ml_models.vae.model import ConvVAE, image_to_tensor, tensor_to_image  # noqa: E402
from ml_models.vae.train_vae import loss_components  # noqa: E402

CHECKPOINT_PATH = VAE_DIR / "outputs" / "best_vae.pt"
RESULTS_DIR = PROJECT_ROOT / "backend" / "results" / "vae"
logger = logging.getLogger(__name__)


class VAEUnavailableError(Exception):
    """Raised when no trained VAE checkpoint is available or it cannot be loaded."""


class VAEInputError(Exception):
    """Raised when an uploaded reconstruction image is invalid."""


class VAEService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: ConvVAE | None = None
        self.beta = 1.0
        self.checkpoint_path: Path | None = None
        self.checkpoint_modified_at: int | None = None
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def reload(self) -> None:
        if not CHECKPOINT_PATH.is_file():
            self.model = None
            self.checkpoint_path = None
            self.checkpoint_modified_at = None
            return
        try:
            state = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=False)
            if state.get("architecture") != "conv_vae_v1":
                raise ValueError("Checkpoint architecture is not conv_vae_v1.")
            model = ConvVAE(
                latent_dim=state["latent_dim"],
                image_size=state["image_size"],
                base_channels=state.get("base_channels", 32),
            ).to(self.device)
            model.load_state_dict(state["model_state_dict"], strict=True)
            model.eval()
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            logger.exception("Could not load VAE checkpoint at %s.", CHECKPOINT_PATH)
            self.model = None
            self.checkpoint_path = None
            self.checkpoint_modified_at = None
            raise VAEUnavailableError(
                "The VAE checkpoint exists but could not be loaded. Review the backend log."
            ) from exc
        self.model = model
        self.beta = float(state.get("beta", 1.0))
        self.checkpoint_path = CHECKPOINT_PATH
        self.checkpoint_modified_at = CHECKPOINT_PATH.stat().st_mtime_ns

    def _require_model(self) -> ConvVAE:
        if (
            self.model is None
            or not CHECKPOINT_PATH.is_file()
            or self.checkpoint_modified_at != CHECKPOINT_PATH.stat().st_mtime_ns
        ):
            self.reload()
        if self.model is None:
            raise VAEUnavailableError(
                "VAE is not available yet. Train the model to create "
                "ml_models/vae/outputs/best_vae.pt."
            )
        return self.model

    def status(self) -> dict:
        checkpoint_exists = CHECKPOINT_PATH.is_file()
        return {
            "model": "VAE",
            "available": checkpoint_exists,
            "loaded": self.model is not None,
            "device": str(self.device),
            "checkpoint": str(CHECKPOINT_PATH) if checkpoint_exists else None,
            "latent_dim": self.model.latent_dim if self.model is not None else None,
            "image_size": self.model.image_size if self.model is not None else None,
            "generation_mode": "unconditional prior sampling",
        }

    @staticmethod
    def _decode_upload(data: bytes, filename: str | None) -> Image.Image:
        if not data:
            raise VAEInputError("No image uploaded.")
        suffix = Path(filename or "").suffix.lower()
        if suffix and suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            raise VAEInputError("Unsupported image format. Use JPG, PNG, BMP, or WEBP.")
        try:
            with Image.open(BytesIO(data)) as image:
                image.load()
                if image.width < 1 or image.height < 1:
                    raise VAEInputError("The uploaded image has invalid dimensions.")
                return image.convert("RGB")
        except UnidentifiedImageError as exc:
            raise VAEInputError("The uploaded file is not a valid image.") from exc
        except OSError as exc:
            raise VAEInputError("The uploaded image could not be read.") from exc

    def reconstruct(self, data: bytes, filename: str | None) -> dict:
        model = self._require_model()
        original = self._decode_upload(data, filename)
        tensor = image_to_tensor(original, model.image_size).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            reconstruction, mu, logvar = model.reconstruct(tensor)
            mse = F.mse_loss(reconstruction, tensor).item()
            mae = F.l1_loss(reconstruction, tensor).item()
            ssim = ssim_per_image(reconstruction, tensor).mean().item()
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp()).item()
            total_loss, _, _ = loss_components(reconstruction, tensor, mu, logvar, self.beta)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - started) * 1000.0

        result_id = uuid.uuid4().hex
        input_name = f"{result_id}_input.png"
        output_name = f"{result_id}_reconstruction.png"
        original.resize((model.image_size, model.image_size), Image.Resampling.BILINEAR).save(
            RESULTS_DIR / input_name
        )
        tensor_to_image(reconstruction.squeeze(0)).save(RESULTS_DIR / output_name)
        return {
            "model": "VAE",
            "mode": "reconstruction",
            "input_image": f"/results/vae/{input_name}",
            "output_image": f"/results/vae/{output_name}",
            "seed": None,
            "metrics": {
                "mse": mse,
                "mae": mae,
                "psnr": float(10.0 * np.log10(1.0 / max(mse, 1e-12))),
                "ssim": ssim,
                "kl_divergence": kl_divergence,
                "total_loss": total_loss.item(),
                "inference_time_ms": inference_ms,
            },
        }

    def generate(self, seed: int | None = None) -> dict:
        model = self._require_model()
        actual_seed = seed if seed is not None else secrets.randbelow(2**31)
        with torch.inference_mode():
            generated = model.sample(1, seed=actual_seed, device=self.device)
        filename = f"{uuid.uuid4().hex}_prior_sample.png"
        tensor_to_image(generated.squeeze(0)).save(RESULTS_DIR / filename)
        return {
            "model": "VAE",
            "mode": "generation",
            "input_image": None,
            "output_image": f"/results/vae/{filename}",
            "seed": actual_seed,
            "metrics": None,
        }

    @staticmethod
    def training_history() -> list[dict]:
        history_path = VAE_DIR / "outputs" / "training_history.json"
        if not history_path.is_file():
            return []
        try:
            import json

            history = json.loads(history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.exception("Could not read VAE training history at %s.", history_path)
            raise VAEUnavailableError("VAE training history could not be read.") from exc

        rows = []
        for index, epoch in enumerate(history.get("epoch", [])):
            rows.append(
                {
                    "name": str(epoch),
                    "train_total": _history_value(history, "train_total_loss", index),
                    "val_total": _history_value(history, "val_total_loss", index),
                    "train_reconstruction": _history_value(history, "train_reconstruction_loss", index),
                    "val_reconstruction": _history_value(history, "val_reconstruction_loss", index),
                    "train_kl": _history_value(history, "train_kl_divergence", index),
                    "val_kl": _history_value(history, "val_kl_divergence", index),
                }
            )
        return rows


def _history_value(history: dict, key: str, index: int) -> float | None:
    values = history.get(key, [])
    return float(values[index]) if index < len(values) else None


service = VAEService()
