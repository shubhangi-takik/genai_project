"""Dataset loader for Conditional GAN interior design generation."""

from __future__ import annotations

import csv
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

from ml_models.gan.model import extract_structural_edges, image_to_tensor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "Pinterest Interior Design Images and Metadata"


class PairedInteriorDataset(Dataset):
    """Loads room images and dynamically generates structural wireframe/edge conditions."""

    def __init__(self, data_root: str | Path = DEFAULT_DATA, split: str = "train", image_size: int = 128):
        self.root = Path(data_root)
        self.image_size = image_size
        csv_path = self.root / f"{split}_data.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV split file: {csv_path}")
        
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        
        self.paths = [self._resolve(row["image_path"]) for row in rows]
        missing = [p for p in self.paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images missing from {csv_path.name}")

    def _resolve(self, csv_value: str) -> Path:
        direct = (self.root / csv_value).resolve()
        if direct.is_file():
            return direct
        parts = Path(csv_value.replace("\\", "/")).parts
        return self.root.joinpath(*parts[-3:])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[index]) as img:
            rgb_img = img.convert("RGB")
            # Extract structural architectural condition (walls, windows, perspective)
            condition_img = extract_structural_edges(rgb_img, self.image_size)
            
            real_tensor = image_to_tensor(rgb_img, self.image_size)
            cond_tensor = image_to_tensor(condition_img, self.image_size)
            
        return cond_tensor, real_tensor
