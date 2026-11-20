from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from fastapi import APIRouter, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTOENCODER_METRICS = (
    PROJECT_ROOT
    / "ml_models"
    / "autoencoder"
    / "outputs"
    / "evaluation_test"
    / "evaluation_results.json"
)
VAE_METRICS = (
    PROJECT_ROOT
    / "ml_models"
    / "vae"
    / "outputs"
    / "evaluation_test"
    / "metrics_summary.json"
)
GAN_METRICS = (
    PROJECT_ROOT
    / "ml_models"
    / "gan"
    / "outputs"
    / "evaluation_test"
    / "metrics_summary.json"
)
router = APIRouter(prefix="/api/comparison", tags=["model comparison"])
logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception("Could not load model evaluation data from %s.", path)
        raise HTTPException(
            status_code=500,
            detail=f"Could not read evaluation results for {path.parent.parent.parent.name}.",
        ) from exc


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _model_row(
    name: str,
    task: str,
    source: Path | None,
    metrics: dict | None = None,
    images: object = None,
    note: str | None = None,
) -> dict:
    clean_metrics = {
        key: _finite_number(value)
        for key, value in (metrics or {}).items()
    }
    if name == "GAN" and clean_metrics.get("fid") is not None and clean_metrics["fid"] < 0:
        clean_metrics["fid"] = None
        note = "The saved FID value was negative and is omitted because FID cannot be negative."

    evaluated = bool(source and source.is_file() and any(value is not None for value in clean_metrics.values()))
    return {
        "name": name,
        "task": task,
        "status": "evaluated" if evaluated else "not_evaluated",
        "evaluation_split": "test" if evaluated else None,
        "images": _finite_number(images),
        "metrics": clean_metrics,
        "source": str(source) if source and source.is_file() else None,
        "note": note,
    }


@router.get("/metrics")
def comparison_metrics() -> dict:
    autoencoder = _load_json(AUTOENCODER_METRICS)
    vae_summary = _load_json(VAE_METRICS)
    gan = _load_json(GAN_METRICS)
    vae = vae_summary.get("metrics", {}).get("VAE", {})

    models = [
        _model_row(
            "Autoencoder",
            "Image reconstruction",
            AUTOENCODER_METRICS,
            {
                "mse": autoencoder.get("mse"),
                "mae": autoencoder.get("mae"),
                "psnr": autoencoder.get("psnr_db"),
                "ssim": autoencoder.get("ssim"),
                "inference_time_ms": (
                    _finite_number(autoencoder.get("mean_inference_time_per_image_seconds")) * 1000
                    if _finite_number(autoencoder.get("mean_inference_time_per_image_seconds")) is not None
                    else None
                ),
            },
            autoencoder.get("images"),
        ),
        _model_row(
            "VAE",
            "Image reconstruction / latent sampling",
            VAE_METRICS,
            {
                "mse": vae.get("mse"),
                "mae": vae.get("mae"),
                "psnr": (
                    10 * math.log10(1 / max(_finite_number(vae.get("mse")), 1e-12))
                    if _finite_number(vae.get("mse")) is not None
                    else None
                ),
                "ssim": vae.get("ssim"),
                "inference_time_ms": vae.get("mean_inference_ms"),
            },
            vae_summary.get("test_images"),
        ),
        _model_row(
            "GAN",
            "Conditional image translation",
            GAN_METRICS,
            {
                "mse": gan.get("mse"),
                "mae": gan.get("mae"),
                "psnr": gan.get("psnr"),
                "ssim": gan.get("ssim"),
                "fid": gan.get("fid"),
                "inference_time_ms": gan.get("mean_inference_time_ms"),
            },
            gan.get("images"),
        ),
        _model_row(
            "Transformer",
            "Rule-based design requirement extraction",
            None,
            note="No labeled NLP test set is available, so precision, recall, and F1 have not been measured.",
        ),
        _model_row(
            "Diffusion",
            "Prompt-conditioned room inpainting",
            None,
            note="A feasibility image was generated, but no held-out image-quality benchmark has been run.",
        ),
    ]

    return {
        "evaluation_split": "test",
        "metric_note": (
            "Image metrics describe similarity to the test images, not how well furniture is added. "
            "Models use different architectures and preprocessing, so these are descriptive results, not an overall ranking."
        ),
        "models": models,
    }
