from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("uvicorn.error")

DEFAULT_MODEL_ID = "google/flan-t5-small"
DEFAULT_TASK = "text2text-generation"
DEFAULT_MAX_NEW_TOKENS = 60
DEFAULT_MIN_INTERVAL = 1
MAX_PROMPT_LENGTH = 320
MAX_INPUT_PROMPT_LENGTH = 240
FALLBACK_SUFFIX = (
    ", ultra detailed, coherent composition, crisp focus, rich textures, balanced lighting, high quality"
)


def clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def clamp_interval(value: int) -> int:
    return max(value, DEFAULT_MIN_INTERVAL)


def normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def truncate_prompt(value: str, limit: int = MAX_PROMPT_LENGTH) -> str:
    text = normalize_whitespace(value).strip(", ")
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip(", ") + "…"


def fallback_prompt(base_prompt: str, strength: float) -> str:
    normalized_base = truncate_prompt(base_prompt or "abstract image")
    amount = clamp01(strength)
    if amount <= 0:
        return normalized_base
    if amount < 0.34:
        return truncate_prompt(f"{normalized_base}, refined details, improved clarity")
    if amount < 0.67:
        return truncate_prompt(f"{normalized_base}, enhanced detail, better lighting, vivid textures")
    return truncate_prompt(f"{normalized_base}{FALLBACK_SUFFIX}")


@dataclass
class PromptEnhancerConfig:
    model_id: str = DEFAULT_MODEL_ID
    task: str = DEFAULT_TASK
    device: str = "auto"
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    local_files_only: bool = False

    @classmethod
    def from_env(cls) -> "PromptEnhancerConfig":
        model_id = os.getenv("CVIS_PROMPT_ENHANCER_MODEL", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
        task = os.getenv("CVIS_PROMPT_ENHANCER_TASK", DEFAULT_TASK).strip() or DEFAULT_TASK
        device = os.getenv("CVIS_PROMPT_ENHANCER_DEVICE", "auto").strip() or "auto"
        max_new_tokens = int(os.getenv("CVIS_PROMPT_ENHANCER_MAX_NEW_TOKENS", str(DEFAULT_MAX_NEW_TOKENS)))
        local_files_only = os.getenv("CVIS_PROMPT_ENHANCER_LOCAL_ONLY", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return cls(
            model_id=model_id,
            task=task,
            device=device,
            max_new_tokens=max(12, min(max_new_tokens, 120)),
            local_files_only=local_files_only,
        )


class PromptEnhancer:
    def __init__(self, config: PromptEnhancerConfig):
        self._config = config
        self._generator: Optional[Any] = None
        self._disabled_reason: Optional[str] = None

    @classmethod
    def from_env(cls) -> "PromptEnhancer":
        return cls(PromptEnhancerConfig.from_env())

    @property
    def detail(self) -> str:
        if self._generator is not None:
            return f"prompt enhancer active ({self._config.model_id})"
        if self._disabled_reason:
            return f"prompt enhancer fallback ({self._disabled_reason})"
        return f"prompt enhancer ready ({self._config.model_id})"

    def enhance_if_due(
        self,
        *,
        base_prompt: str,
        strength: float,
        frame_index: int,
        interval: int,
        enabled: bool,
        cached_prompt: Optional[str],
        last_enhanced_frame: int,
    ) -> tuple[str, Optional[str], int, bool]:
        normalized_base = truncate_prompt(base_prompt or "abstract image")
        if not enabled:
            return normalized_base, None, 0, False

        safe_interval = clamp_interval(interval)
        needs_refresh = cached_prompt is None or (frame_index - last_enhanced_frame) >= safe_interval
        if not needs_refresh:
            return truncate_prompt(cached_prompt), cached_prompt, last_enhanced_frame, False

        refreshed = self.enhance_prompt(normalized_base, strength)
        return refreshed, refreshed, frame_index, True

    def enhance_prompt(self, base_prompt: str, strength: float) -> str:
        prompt = truncate_prompt(base_prompt or "abstract image")
        if clamp01(strength) <= 0:
            return prompt

        generator = self._ensure_generator()
        if generator is None:
            return fallback_prompt(prompt, strength)

        instruction = (
            "Rewrite this image prompt to be concise, descriptive, and quality-focused. "
            "Return one single comma-separated prompt only, with no quotes or explanations.\n"
            f"Prompt: {prompt[:MAX_INPUT_PROMPT_LENGTH]}"
        )
        try:
            output = generator(
                instruction,
                max_new_tokens=self._config.max_new_tokens,
                num_return_sequences=1,
                do_sample=False,
                truncation=True,
            )
            candidate = self._extract_text(output)
            if not candidate:
                return fallback_prompt(prompt, strength)
            merged = self._blend_prompt(prompt, candidate, strength)
            return truncate_prompt(merged)
        except Exception as exc:
            self._disabled_reason = f"inference_failed:{type(exc).__name__}"
            logger.warning("Prompt enhancer inference failed, using fallback: %s", exc)
            return fallback_prompt(prompt, strength)

    def _ensure_generator(self) -> Optional[Any]:
        if self._generator is not None:
            return self._generator
        if self._disabled_reason:
            return None
        try:
            from transformers import pipeline

            pipeline_kwargs: dict[str, Any] = {
                "task": self._config.task,
                "model": self._config.model_id,
                "local_files_only": self._config.local_files_only,
            }
            normalized_device = self._config.device.lower()
            if normalized_device == "cpu":
                pipeline_kwargs["device"] = -1
            elif normalized_device in {"cuda", "cuda:0", "gpu"}:
                pipeline_kwargs["device"] = 0
            self._generator = pipeline(**pipeline_kwargs)
            logger.info("Prompt enhancer model loaded: %s", self._config.model_id)
            return self._generator
        except Exception as exc:
            self._disabled_reason = f"load_failed:{type(exc).__name__}"
            logger.warning("Prompt enhancer model unavailable, using fallback only: %s", exc)
            return None

    @staticmethod
    def _extract_text(output: Any) -> str:
        if isinstance(output, list) and output:
            first = output[0]
            if isinstance(first, dict):
                text = first.get("generated_text") or first.get("summary_text") or ""
                return normalize_whitespace(str(text))
            return normalize_whitespace(str(first))
        return normalize_whitespace(str(output))

    @staticmethod
    def _blend_prompt(base_prompt: str, enhanced: str, strength: float) -> str:
        base_terms = [item.strip() for item in base_prompt.split(",") if item.strip()]
        enhanced_terms = [item.strip() for item in enhanced.split(",") if item.strip()]
        if not enhanced_terms:
            return base_prompt

        amount = clamp01(strength)
        keep_base = max(1, int(round(len(base_terms) * (1.0 - (amount * 0.45)))))
        use_enhanced = max(1, int(round(2 + (amount * 6))))
        merged_terms: list[str] = []
        merged_terms.extend(base_terms[:keep_base])
        for term in enhanced_terms:
            if term.lower() not in {item.lower() for item in merged_terms}:
                merged_terms.append(term)
            if len(merged_terms) >= keep_base + use_enhanced:
                break
        return ", ".join(merged_terms)
