from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routes.autoencoder import router as autoencoder_router
from backend.routes.generation import router as generation_router
from backend.routes.vae import router as vae_router
from backend.routes.gan import router as gan_router
from backend.routes.transformer import router as transformer_router
from backend.routes.comparison import router as comparison_router
RESULTS_DIR = PROJECT_ROOT / "backend" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "autoencoder").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "vae").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "gan").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "diffusion").mkdir(parents=True, exist_ok=True)

app = FastAPI(title="GenAI Interior Design API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(autoencoder_router)
app.include_router(generation_router)
app.include_router(vae_router)
app.include_router(gan_router)
app.include_router(transformer_router)
app.include_router(comparison_router)
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


@app.get("/api/health")
def health() -> dict:
    from backend.services.autoencoder_service import find_checkpoint, service

    checkpoint = find_checkpoint()
    return {
        "status": "ok",
        "device": str(service.device),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "model_loaded": service.model is not None,
    }
