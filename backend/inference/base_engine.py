from __future__ import annotations

from dataclasses import dataclass
from PIL import Image


@dataclass
class InferenceSettings:
  prompt: str
  strength: float
  running: bool
  anti_stagnation_enabled: bool = True
  stagnation_threshold: float = 0.012
  stagnation_window: int = 6
  variation_strength: float = 0.35
  prompt_enhancement_enabled: bool = False
  prompt_enhancement_interval: int = 12
  prompt_enhancement_strength: float = 0.55
  study_frames_enabled: bool = True
  study_frame_count: int = 2
  study_frame_strength: float = 0.2
  study_frame_effort: int = 1
  study_frame_delay: float = 0.6


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
