"""Load the existing Autoencoder, run reconstruction, and compute real metrics."""

from __future__ import annotations

import json
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
AE_DIR = PROJECT_ROOT / "ml_models" / "autoencoder"
if str(AE_DIR) not in sys.path:
    sys.path.insert(0, str(AE_DIR))

from evaluation import ssim_per_image  # noqa: E402
from model import Autoencoder  # noqa: E402

IMAGE_SIZE = 128
CHECKPOINT_CANDIDATES = [
    PROJECT_ROOT / "backend" / "checkpoints" / "best_autoencoder.pt",
    AE_DIR / "outputs" / "best_autoencoder.pt",
]
HISTORY_PATH = AE_DIR / "outputs" / "training_history.json"
RESULTS_DIR = PROJECT_ROOT / "backend" / "results" / "autoencoder"


def find_checkpoint() -> Path | None:
    for path in CHECKPOINT_CANDIDATES:
        if path.is_file():
            return path
    return None


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    rgb = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array)


class AutoencoderService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Autoencoder | None = None
        self.checkpoint_path: Path | None = None
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (PROJECT_ROOT / "backend" / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.reload()

    def reload(self) -> None:
        checkpoint = find_checkpoint()
        if checkpoint is None:
            self.model = None
            self.checkpoint_path = None
            return
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        model = Autoencoder(state.get("latent_channels", 256)).to(self.device)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        self.model = model
        self.checkpoint_path = checkpoint

    def require_model(self) -> Autoencoder:
        if self.model is None:
            self.reload()
        if self.model is None:
            raise FileNotFoundError(
                "Autoencoder checkpoint is missing. Train the model first so "
                "ml_models/autoencoder/outputs/best_autoencoder.pt exists."
            )
        return self.model

    def decode_upload(self, data: bytes, filename: str | None) -> Image.Image:
        if not data:
            raise ValueError("No image uploaded.")
        try:
            image = Image.open(BytesIO(data))
            image.load()
        except UnidentifiedImageError as exc:
            raise ValueError("The uploaded file is not a valid image.") from exc
        except OSError as exc:
            raise ValueError("The uploaded image could not be read.") from exc
        if image.width < 1 or image.height < 1:
            raise ValueError("The uploaded image has invalid dimensions.")
        suffix = Path(filename or "").suffix.lower()
        if suffix and suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            raise ValueError("Unsupported image format. Use JPG, PNG, BMP, or WEBP.")
        return image.convert("RGB")

    def predict(
        self,
        image_bytes: bytes,
        filename: str | None,
        room_type: str | None,
        interior_style: str | None,
        color_preference: str | None,
        furniture_preference: str | None,
        design_prompt: str | None,
    ) -> dict:
        model = self.require_model()
        original = self.decode_upload(image_bytes, filename)
        tensor = image_to_tensor(original).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            reconstruction = model(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - start) * 1000.0

        mse = F.mse_loss(reconstruction, tensor).item()
        mae = F.l1_loss(reconstruction, tensor).item()
        psnr = float(10.0 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item())
        ssim = float(ssim_per_image(reconstruction, tensor).mean().item())

        experiment_id = uuid.uuid4().hex
        input_name = f"{experiment_id}_input.png"
        output_name = f"{experiment_id}_reconstructed.png"
        original.save(RESULTS_DIR / input_name)
        tensor_to_image(reconstruction.squeeze(0)).save(RESULTS_DIR / output_name)

        return {
            "model": "Autoencoder",
            "output_image": f"/results/autoencoder/{output_name}",
            "input_image": f"/results/autoencoder/{input_name}",
            "room_type": room_type,
            "interior_style": interior_style,
            "color_preference": color_preference,
            "furniture_preference": furniture_preference,
            "design_prompt": design_prompt,
            "metrics": {
                "mse": mse,
                "mae": mae,
                "psnr": psnr,
                "ssim": ssim,
                "reconstruction_loss": mse,
                "inference_time_ms": inference_ms,
            },
        }

    def training_history(self) -> list[dict]:
        if not HISTORY_PATH.is_file():
            return []
        payload = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        epochs = payload.get("epoch") or []
        train_loss = payload.get("train_loss") or []
        val_loss = payload.get("val_loss") or []
        rows = []
        for i, epoch in enumerate(epochs):
            rows.append(
                {
                    "name": str(epoch),
                    "train": train_loss[i] if i < len(train_loss) else None,
                    "val": val_loss[i] if i < len(val_loss) else None,
                }
            )
        return rows


service = AutoencoderService()
