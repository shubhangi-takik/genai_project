from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.models.schemas import VAEOutputResponse
from backend.services.vae_service import VAEInputError, VAEUnavailableError, service

router = APIRouter(prefix="/api/vae", tags=["vae"])
logger = logging.getLogger(__name__)


class VAEGenerateRequest(BaseModel):
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)


@router.get("/status")
def vae_status() -> dict:
    return service.status()


@router.get("/history")
def vae_history() -> dict:
    try:
        return {"data": service.training_history()}
    except VAEUnavailableError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/reconstruct", response_model=VAEOutputResponse)
async def reconstruct_vae(image: UploadFile = File(...)) -> VAEOutputResponse:
    if not image.filename:
        raise HTTPException(status_code=400, detail="No image uploaded.")
    try:
        result = service.reconstruct(await image.read(), image.filename)
    except VAEUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except VAEInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("VAE reconstruction failed.")
        raise HTTPException(status_code=500, detail="VAE reconstruction failed. Review the backend log.") from exc
    return VAEOutputResponse.model_validate(result)


@router.post("/generate", response_model=VAEOutputResponse)
def generate_vae_sample(request: VAEGenerateRequest) -> VAEOutputResponse:
    try:
        result = service.generate(seed=request.seed)
    except VAEUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("VAE prior sampling failed.")
        raise HTTPException(status_code=500, detail="VAE generation failed. Review the backend log.") from exc
    return VAEOutputResponse.model_validate(result)
