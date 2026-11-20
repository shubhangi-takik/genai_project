from __future__ import annotations

from pydantic import BaseModel, Field


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    room_type: str = Field(default="", max_length=120)
    interior_style: str = Field(default="", max_length=120)
    color_preference: str = Field(default="", max_length=300)
    furniture_preference: str = Field(default="", max_length=300)


class ImageGenerationResponse(BaseModel):
    model: str
    output_image: str
    prompt: str
    room_type: str
    interior_style: str
    color_preference: str
    furniture_preference: str
