from __future__ import annotations

from pydantic import BaseModel


class AutoencoderMetrics(BaseModel):
    mse: float
    mae: float
    psnr: float
    ssim: float
    reconstruction_loss: float
    inference_time_ms: float


class AutoencoderPredictResponse(BaseModel):
    model: str = "Autoencoder"
    output_image: str
    input_image: str
    room_type: str | None = None
    interior_style: str | None = None
    color_preference: str | None = None
    furniture_preference: str | None = None
    design_prompt: str | None = None
    metrics: AutoencoderMetrics


class VAEMetrics(BaseModel):
    mse: float | None = None
    mae: float | None = None
    psnr: float | None = None
    ssim: float | None = None
    kl_divergence: float | None = None
    total_loss: float | None = None
    inference_time_ms: float | None = None


class VAEOutputResponse(BaseModel):
    model: str = "VAE"
    mode: str
    input_image: str | None = None
    output_image: str
    seed: int | None = None
    metrics: VAEMetrics | None = None


class GANMetrics(BaseModel):
    fid: float | None = None
    inception_score: float | None = None
    ssim: float | None = None
    psnr: float | None = None
    generator_loss: float | None = None
    discriminator_loss: float | None = None
    inference_time_ms: float | None = None


class GANResponse(BaseModel):
    model: str = "GAN"
    condition_image: str
    input_image: str
    output_image: str
    metrics: GANMetrics
