from __future__ import annotations

from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from backend.models.generation_schemas import ImageGenerationRequest, ImageGenerationResponse
from backend.services.image_generation_service import (
    GenerationConfigurationError,
    GenerationProviderError,
    GenerationRateLimitError,
    generate_diffusion_interior_image,
    generate_image,
)

router = APIRouter(prefix="/api/generation", tags=["image generation"])


@router.post("/image", response_model=ImageGenerationResponse)
def generate_interior_design(request: ImageGenerationRequest) -> ImageGenerationResponse:
    try:
        result = generate_image(
            prompt=request.prompt,
            room_type=request.room_type,
            interior_style=request.interior_style,
            color_preference=request.color_preference,
            furniture_preference=request.furniture_preference,
        )
    except GenerationConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GenerationRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GenerationProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ImageGenerationResponse.model_validate(result)


@router.post("/diffuse", response_model=ImageGenerationResponse)
async def generate_diffusion_interior_design(
    prompt: str = Form(..., min_length=1, max_length=4000),
    room_type: str = Form(default="", max_length=120),
    interior_style: str = Form(default="", max_length=120),
    color_preference: str = Form(default="", max_length=300),
    furniture_preference: str = Form(default="", max_length=300),
    image: UploadFile = File(...),
) -> ImageGenerationResponse:
    if not image.filename:
        raise HTTPException(status_code=400, detail="Upload an empty-room image before generating a furnished version.")

    try:
        file_bytes = await image.read()
        uploaded_image = Image.open(BytesIO(file_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc

    try:
        result = generate_diffusion_interior_image(
            image=uploaded_image,
            prompt=prompt,
            room_type=room_type,
            interior_style=interior_style,
            color_preference=color_preference,
            furniture_preference=furniture_preference,
        )
    except GenerationConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GenerationRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GenerationProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ImageGenerationResponse.model_validate(result)
