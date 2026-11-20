from __future__ import annotations

import logging
from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.models.schemas import GANResponse
from backend.services.gan_service import service

router = APIRouter(prefix="/api/gan", tags=["gan"])
logger = logging.getLogger(__name__)


@router.get("/status")
def gan_status() -> dict:
    return service.status()


@router.get("/history")
def gan_history() -> dict:
    return {"data": service.training_history()}


@router.post("/generate", response_model=GANResponse)
async def generate_gan(image: UploadFile = File(...)) -> GANResponse:
    if not image.filename:
        raise HTTPException(status_code=400, detail="No image uploaded.")
    try:
        data = await image.read()
        result = service.predict(data, image.filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("GAN generation failed.")
        raise HTTPException(status_code=500, detail=f"GAN generation failed: {exc}") from exc
    return GANResponse.model_validate(result)
