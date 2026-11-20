from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from backend.services.transformer_service import analyze_design_prompt

router = APIRouter(prefix="/api/transformer", tags=["transformer"])


class AnalyzePromptRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Design prompt must not be blank.")
        return value


@router.post("/analyze")
def analyze_prompt(request: AnalyzePromptRequest) -> dict[str, object]:
    """Extract known interior-design terms using a deterministic rule-based baseline."""
    return analyze_design_prompt(request.prompt)
