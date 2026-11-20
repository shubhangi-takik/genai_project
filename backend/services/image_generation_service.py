from __future__ import annotations

import base64
import binascii
import logging
import os
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoPipelineForInpainting
from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI, OpenAIError, RateLimitError
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATED_DIR = PROJECT_ROOT / "backend" / "results" / "generated"
DIFFUSION_DIR = PROJECT_ROOT / "backend" / "results" / "diffusion"
MODEL = "gpt-image-2.5-flare"
LOCAL_DIFFUSION_MODEL_PATH = os.getenv("LOCAL_DIFFUSION_MODEL_PATH") or os.getenv("DIFFUSION_MODEL_PATH")
DIFFUSION_MODEL_ID = os.getenv("DIFFUSION_MODEL_ID", "sd-legacy/stable-diffusion-inpainting")
logger = logging.getLogger(__name__)
DIFFUSION_PIPELINE = None


class GenerationError(Exception):
    """Base class for errors safe to return to API clients."""


class GenerationConfigurationError(GenerationError):
    pass


class GenerationRateLimitError(GenerationError):
    pass


class GenerationProviderError(GenerationError):
    pass


def compose_prompt(
    prompt: str,
    room_type: str = "",
    interior_style: str = "",
    color_preference: str = "",
    furniture_preference: str = "",
) -> str:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Enter a design prompt before generating an image.")

    selections = [
        ("Room type", room_type),
        ("Interior style", interior_style),
        ("Color palette", color_preference),
        ("Furniture", furniture_preference),
    ]
    details = [f"{label}: {value.strip()}" for label, value in selections if value.strip()]
    if details:
        return f"{prompt}\n\nDesign requirements:\n" + "\n".join(details)
    return prompt


def _build_inpaint_mask(width: int, height: int, *, x_ratio: tuple[float, float] = (0.15, 0.85), y_ratio: tuple[float, float] = (0.28, 0.9)) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    x0 = int(width * x_ratio[0])
    x1 = int(width * x_ratio[1])
    y0 = int(height * y_ratio[0])
    y1 = int(height * y_ratio[1])
    mask_data = np.asarray(mask).copy()
    mask_data[y0:y1, x0:x1] = 255
    return Image.fromarray(mask_data, mode="L")


def _get_diffusion_pipeline():
    global DIFFUSION_PIPELINE
    if DIFFUSION_PIPELINE is not None:
        return DIFFUSION_PIPELINE

    model_path = LOCAL_DIFFUSION_MODEL_PATH or None
    try:
        if model_path and Path(model_path).exists():
            pipe = AutoPipelineForInpainting.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
        else:
            pipe = AutoPipelineForInpainting.from_pretrained(
                DIFFUSION_MODEL_ID,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise GenerationConfigurationError(
            "The local diffusion inpainting model is not available. Set LOCAL_DIFFUSION_MODEL_PATH or allow the pipeline to download the model in the configured environment."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)
    DIFFUSION_PIPELINE = pipe
    return pipe


def generate_diffusion_interior_image(
    image: Image.Image,
    prompt: str,
    room_type: str = "",
    interior_style: str = "",
    color_preference: str = "",
    furniture_preference: str = "",
) -> dict[str, str]:
    full_prompt = compose_prompt(prompt, room_type, interior_style, color_preference, furniture_preference)
    image = image.convert("RGB")
    max_dimension = 1024
    scale = min(1.0, max_dimension / max(image.size))
    if scale < 1.0:
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)

    mask = _build_inpaint_mask(image.width, image.height)
    pipe = _get_diffusion_pipeline()
    device = pipe.device if hasattr(pipe, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = torch.Generator(device=str(device)).manual_seed(42)
    with torch.inference_mode():
        result = pipe(
            prompt=full_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=25,
            guidance_scale=7.5,
            generator=generator,
        )
    generated = result.images[0]

    DIFFUSION_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.png"
    output_path = DIFFUSION_DIR / filename
    generated.save(output_path)
    return {
        "model": DIFFUSION_MODEL_ID,
        "output_image": f"/results/diffusion/{filename}",
        "prompt": full_prompt,
        "room_type": room_type,
        "interior_style": interior_style,
        "color_preference": color_preference,
        "furniture_preference": furniture_preference,
    }


def generate_image(
    prompt: str,
    room_type: str = "",
    interior_style: str = "",
    color_preference: str = "",
    furniture_preference: str = "",
) -> dict[str, str]:
    full_prompt = compose_prompt(
        prompt, room_type, interior_style, color_preference, furniture_preference
    )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise GenerationConfigurationError(
            "Text-to-image generation is not configured. Set OPENAI_API_KEY in the backend environment."
        )

    client = OpenAI(api_key=api_key, timeout=120.0, max_retries=0)
    try:
        response = client.images.generate(
            model=MODEL,
            prompt=full_prompt,
            size="1024x1024",
            quality="medium",
            output_format="png",
        )
    except AuthenticationError as exc:
        logger.warning("Image provider rejected backend credentials.")
        raise GenerationConfigurationError(
            "The image-generation provider rejected the backend API credentials."
        ) from exc
    except RateLimitError as exc:
        logger.warning("Image provider rate or usage limit reached.")
        raise GenerationRateLimitError(
            "The image-generation provider rate or usage limit was reached. Check API billing and limits."
        ) from exc
    except APIConnectionError as exc:
        logger.warning("Could not connect to image-generation provider.")
        raise GenerationProviderError(
            "Could not connect to the image-generation provider. Try again later."
        ) from exc
    except APIStatusError as exc:
        logger.warning("Image provider returned HTTP %s.", exc.status_code)
        raise GenerationProviderError(
            f"The image-generation provider returned an error (HTTP {exc.status_code}). "
            "Review the prompt and provider status, then try again."
        ) from exc
    except OpenAIError as exc:
        logger.warning("Image provider request failed (%s).", type(exc).__name__)
        raise GenerationProviderError(
            "The image-generation request failed. Try again later."
        ) from exc

    if not response.data or not response.data[0].b64_json:
        raise GenerationProviderError(
            "The image-generation provider returned no image. No output file was saved."
        )

    try:
        image_bytes = base64.b64decode(response.data[0].b64_json, validate=True)
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except (binascii.Error, UnidentifiedImageError, OSError, ValueError) as exc:
        raise GenerationProviderError(
            "The image-generation provider returned an invalid image. No output file was saved."
        ) from exc

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.png"
    output_path = GENERATED_DIR / filename
    output_path.write_bytes(image_bytes)
    return {
        "model": MODEL,
        "output_image": f"/results/generated/{filename}",
        "prompt": full_prompt,
        "room_type": room_type,
        "interior_style": interior_style,
        "color_preference": color_preference,
        "furniture_preference": furniture_preference,
    }
