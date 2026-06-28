from __future__ import annotations

from dataclasses import dataclass
from PIL import Image


@dataclass
class InferenceSettings:
  prompt: str
  strength: float
  running: bool


class InferenceEngine:
  name = "base"

  @property
  def model_ready(self) -> bool:
    return False

  @property
  def detail(self) -> str:
    return "Engine not initialized."

  def transform(self, image: Image.Image, settings: InferenceSettings, frame_index: int) -> Image.Image:
    raise NotImplementedError
