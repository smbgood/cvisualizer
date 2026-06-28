from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Optional

from PIL import Image

from .base_engine import InferenceEngine, InferenceSettings

logger = logging.getLogger("uvicorn.error")


@dataclass
class StreamRuntime:
    stream: object
    loaded_model: str
    t_index_list: list[int]
    guidance_scale: float
    prepared_prompt: Optional[str] = None


def parse_t_index_list(raw_value: str) -> list[int]:
    values = []
    for item in raw_value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(min(max(int(stripped), 0), 49))

    if not values:
        raise ValueError("At least one timestep index is required.")
    return values


class StreamDiffusionEngine(InferenceEngine):
    name = "streamdiffusion"

    def __init__(self, runtime: StreamRuntime):
        self._runtime = runtime
        self._detail = (
            f"Running {runtime.loaded_model} via StreamDiffusion "
            f"(t_index_list={runtime.t_index_list}, guidance_scale={runtime.guidance_scale})."
        )

    @property
    def model_ready(self) -> bool:
        return True

    @property
    def detail(self) -> str:
        return self._detail

    @classmethod
    def try_create_from_env(cls) -> Optional["StreamDiffusionEngine"]:
        model_id = os.getenv("CVIS_MODEL_ID", "stabilityai/sd-turbo")
        device = os.getenv("CVIS_DEVICE", "cuda")
        try:
            t_index_list = parse_t_index_list(os.getenv("CVIS_T_INDEX_LIST", "0,16,32"))
            guidance_scale = float(os.getenv("CVIS_GUIDANCE_SCALE", "0.0"))
            cfg_type = os.getenv("CVIS_CFG_TYPE", "none")
        except Exception:
            logger.exception("Invalid StreamDiffusion environment settings; falling back to mock engine")
            return None
        logger.info(
            "Initializing StreamDiffusion engine (model=%s, device=%s, t_index_list=%s, guidance_scale=%.2f)",
            model_id,
            device,
            t_index_list,
            guidance_scale,
        )

        try:
            import torch
            from diffusers import StableDiffusionImg2ImgPipeline
            from streamdiffusion import StreamDiffusion
        except Exception:
            logger.exception("StreamDiffusion dependencies are unavailable; falling back to mock engine")
            return None

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to mock engine")
            return None

        try:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16
            ).to(device)
            stream_kwargs = {
                "t_index_list": t_index_list,
                "torch_dtype": torch.float16,
                "width": 512,
                "height": 512,
            }
            try:
                stream = StreamDiffusion(pipe, cfg_type=cfg_type, **stream_kwargs)
            except TypeError:
                logger.warning("StreamDiffusion cfg_type argument unsupported; retrying without it")
                stream = StreamDiffusion(pipe, **stream_kwargs)

            try:
                stream.load_lcm_lora()
                stream.fuse_lora()
            except Exception:
                # SD-Turbo can still run without loading extra LoRA.
                pass

            return cls(
                StreamRuntime(
                    stream=stream,
                    loaded_model=model_id,
                    t_index_list=t_index_list,
                    guidance_scale=guidance_scale,
                )
            )
        except Exception:
            logger.exception("StreamDiffusion initialization failed; falling back to mock engine")
            return None

    def transform(self, image: Image.Image, settings: InferenceSettings, frame_index: int) -> Image.Image:
        # Keep real-time constraints conservative for 8GB cards.
        source = image.convert("RGB").resize((512, 512))

        stream = self._runtime.stream
        self._ensure_prompt(stream, settings.prompt)
        output = stream(source)

        if isinstance(output, list):
            candidate = output[0]
        else:
            candidate = output

        if hasattr(candidate, "images"):
            return candidate.images[0].convert("RGB")
        if isinstance(candidate, Image.Image):
            return candidate.convert("RGB")
        if hasattr(stream, "image_processor"):
            images = stream.image_processor.postprocess(candidate, output_type="pil")
            if images:
                return images[0].convert("RGB")
        return source

    def _ensure_prompt(self, stream: object, prompt: str) -> None:
        prompt = prompt.strip() or "radical surreal neon transformation"
        if self._runtime.prepared_prompt is None:
            stream.prepare(prompt=prompt, guidance_scale=self._runtime.guidance_scale)
        elif prompt != self._runtime.prepared_prompt:
            if hasattr(stream, "update_prompt"):
                stream.update_prompt(prompt)
            else:
                stream.prepare(prompt=prompt, guidance_scale=self._runtime.guidance_scale)
        else:
            return

        self._runtime.prepared_prompt = prompt
