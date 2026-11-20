"""Rule-based extraction of structured interior-design requirements."""

from __future__ import annotations

import re


ROOM_TYPES = {
    "living room": "living_room",
    "living-room": "living_room",
    "bedroom": "bedroom",
    "kitchen": "kitchen",
    "bathroom": "bathroom",
    "dining room": "dining_room",
    "dining-room": "dining_room",
    "office": "office",
    "home office": "office",
    "study": "office",
}

INTERIOR_STYLES = (
    "scandinavian",
    "minimalist",
    "contemporary",
    "traditional",
    "industrial",
    "farmhouse",
    "mid-century modern",
    "mid century modern",
    "modern",
    "boho",
    "rustic",
    "coastal",
    "japandi",
    "eclectic",
    "transitional",
)

COLOR_TERMS = (
    "warm tones",
    "cool tones",
    "neutral tones",
    "earth tones",
    "black and white",
    "off-white",
    "off white",
    "navy blue",
    "sage green",
    "forest green",
    "light blue",
    "dark blue",
    "warm white",
    "cool white",
    "beige",
    "cream",
    "ivory",
    "white",
    "black",
    "gray",
    "grey",
    "brown",
    "tan",
    "taupe",
    "terracotta",
    "green",
    "blue",
    "yellow",
    "orange",
    "red",
    "pink",
    "purple",
    "teal",
    "gold",
    "silver",
    "neutral",
    "colorful",
    "colourful",
)

FURNITURE_TERMS = (
    "coffee table",
    "dining table",
    "side table",
    "bedside table",
    "bookshelf",
    "bookcase",
    "armchair",
    "sectional",
    "sofa",
    "couch",
    "bed",
    "dresser",
    "wardrobe",
    "nightstand",
    "desk",
    "chair",
    "table",
    "cabinet",
    "stool",
    "ottoman",
    "console",
    "shelving",
    "rug",
    "carpet",
    "curtains",
    "curtain",
    "blinds",
)

LIGHTING_TERMS = (
    "warm lighting",
    "natural lighting",
    "ambient lighting",
    "task lighting",
    "soft lighting",
    "bright lighting",
    "pendant light",
    "pendant lights",
    "chandelier",
    "floor lamp",
    "table lamp",
    "wall sconces",
    "sconce",
    "recessed lighting",
    "led lighting",
    "natural light",
    "warm light",
    "soft light",
    "bright light",
    "lighting",
    "lamps",
    "lamp",
)

DESIGN_KEYWORDS = (
    "natural materials",
    "open plan",
    "open-plan",
    "large windows",
    "high ceiling",
    "high ceilings",
    "wooden flooring",
    "wood floors",
    "hardwood floor",
    "marble",
    "brass",
    "wood",
    "rattan",
    "linen",
    "plants",
    "plant",
    "greenery",
    "fireplace",
    "cozy",
    "spacious",
    "airy",
    "elegant",
    "functional",
    "storage",
    "textured",
    "minimal",
)


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


def _matches(text: str, terms: tuple[str, ...]) -> list[str]:
    matches = [term for term in terms if _contains_phrase(text, term)]
    return [
        term
        for term in matches
        if not any(
            term != other and _contains_phrase(other, term)
            for other in matches
        )
    ]


def analyze_design_prompt(prompt: str) -> dict[str, object]:
    """Extract known room-design terms without implying a trained NLP model."""
    normalized = " ".join(prompt.casefold().split())
    room_type = next(
        (room for phrase, room in ROOM_TYPES.items() if _contains_phrase(normalized, phrase)),
        None,
    )
    interior_style = next(
        (style for style in INTERIOR_STYLES if _contains_phrase(normalized, style)),
        None,
    )

    return {
        "extraction_method": "rule_based",
        "requirements": {
            "room_type": room_type,
            "interior_style": interior_style,
            "color_preferences": _matches(normalized, COLOR_TERMS),
            "furniture_preferences": _matches(normalized, FURNITURE_TERMS),
            "lighting_preferences": _matches(normalized, LIGHTING_TERMS),
            "design_keywords": _matches(normalized, DESIGN_KEYWORDS),
        },
    }
