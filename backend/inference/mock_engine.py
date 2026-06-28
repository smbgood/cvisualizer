from __future__ import annotations

import numpy as np
from PIL import Image

from .base_engine import InferenceEngine, InferenceSettings


class MockEngine(InferenceEngine):
    name = "mock-engine"

    @property
    def model_ready(self) -> bool:
        return True

    @property
    def detail(self) -> str:
        return "Mock mode active. Install StreamDiffusion dependencies for real model output."

    def transform(self, image: Image.Image, settings: InferenceSettings, frame_index: int) -> Image.Image:
        base = image.convert("RGB").resize((512, 512))
        array = np.asarray(base).astype(np.float32)

        phase = frame_index * 0.08
        shift_x = int(8 * np.sin(phase))
        shift_y = int(8 * np.cos(phase * 1.3))
        rolled = np.roll(array, shift=(shift_y, shift_x), axis=(0, 1))

        tint = np.array(
            [
                64 + 40 * np.sin(phase * 0.7),
                80 + 45 * np.sin(phase * 1.1 + 1.2),
                96 + 60 * np.sin(phase * 1.6 + 2.4),
            ],
            dtype=np.float32,
        )

        prompt_gain = min(max(len(settings.prompt) / 80.0, 0.0), 1.0)
        mix = min(max(settings.strength, 0.05), 1.0)

        overlay = (rolled * (0.65 + 0.2 * prompt_gain)) + tint
        result = array * (1.0 - mix * 0.65) + overlay * (mix * 0.65)
        result = np.clip(result, 0, 255).astype(np.uint8)

        return Image.fromarray(result, mode="RGB")
