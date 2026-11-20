from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.models.schemas import AutoencoderPredictResponse
from backend.services.autoencoder_service import service

router = APIRouter(prefix="/api/autoencoder", tags=["autoencoder"])


@router.post("/predict", response_model=AutoencoderPredictResponse)
async def predict_autoencoder(
    image: UploadFile = File(...),
    room_type: str = Form(""),
    interior_style: str = Form(""),
    color_preference: str = Form(""),
    furniture_preference: str = Form(""),
    design_prompt: str = Form(""),
) -> AutoencoderPredictResponse:
    if image is None or not image.filename:
        raise HTTPException(status_code=400, detail="No image uploaded.")
    try:
        data = await image.read()
        result = service.predict(
            image_bytes=data,
            filename=image.filename,
            room_type=room_type or None,
            interior_style=interior_style or None,
            color_preference=color_preference or None,
            furniture_preference=furniture_preference or None,
            design_prompt=design_prompt or None,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Autoencoder inference failed: {exc}") from exc
    return AutoencoderPredictResponse.model_validate(result)


@router.get("/history")
def autoencoder_history() -> dict:
    return {"data": service.training_history()}
