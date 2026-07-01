from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

try:
    from backend.inference import InferenceSettings, MockEngine, StreamDiffusionEngine
except Exception:
    try:
        from .inference import InferenceSettings, MockEngine, StreamDiffusionEngine
    except Exception:
        raise

try:
    from backend.prompt_enhancer import PromptEnhancer
except Exception:
    try:
        from .prompt_enhancer import PromptEnhancer
    except Exception:
        raise


def image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{payload}"


def data_url_to_image(data_url: str) -> Image.Image:
    if "," not in data_url:
        raise ValueError("Invalid data URL.")
    encoded = data_url.split(",", 1)[1]
    raw = base64.b64decode(encoded)
    image = Image.open(BytesIO(raw))
    return image.convert("RGB")


@dataclass
class SessionState:
    seed_image: Optional[Image.Image]
    settings: InferenceSettings
    session_id: str
    session_dir: Path
    created_at: str
    frames: list[dict[str, object]]
    seeds: list[dict[str, object]]
    frame_index: int = 0
    saved_frame_index: int = 0
    saved_seed_index: int = 0
    current_seed_index: Optional[int] = None
    current_seed_filename: Optional[str] = None
    previous_frame_small: Optional[np.ndarray] = None
    stagnant_frames: int = 0
    variation_pulse_remaining: int = 0
    enhanced_prompt: Optional[str] = None
    prompt_enhancement_last_frame: int = 0


DEFAULT_PROMPT = "radical surreal neon transformation"
VARIATION_SUFFIX = ", fresh composition, altered color palette, unexpected details"


def clamp_strength(value: float) -> float:
    return min(max(value, 0.05), 1.0)


def clamp_threshold(value: float) -> float:
    return min(max(value, 0.001), 0.08)


def clamp_window(value: int) -> int:
    return min(max(value, 2), 30)


def clamp_variation_strength(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def clamp_prompt_enhancement_interval(value: int) -> int:
    return min(max(value, 1), 120)


def clamp_prompt_enhancement_strength(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def clamp_study_frame_count(value: int) -> int:
    return min(max(value, 0), 6)


def clamp_study_frame_strength(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def clamp_study_frame_effort(value: int) -> int:
    return min(max(value, 1), 8)


def clamp_study_frame_delay(value: float) -> float:
    return min(max(value, 0.05), 3.0)


def image_feedback_source(previous: Image.Image, generated: Image.Image, strength: float) -> Image.Image:
    base = previous.convert("RGB").resize(generated.size)
    feedback_weight = clamp_strength(strength)
    return Image.blend(base, generated.convert("RGB"), feedback_weight)


def downsample_for_delta(image: Image.Image, size: int = 64) -> np.ndarray:
    reduced = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(reduced, dtype=np.float32) / 255.0


def frame_delta_score(previous_small: np.ndarray, current_small: np.ndarray) -> float:
    return float(np.mean(np.abs(current_small - previous_small)))


def build_effective_prompt(prompt: str, variation_applied: bool) -> str:
    base_prompt = prompt.strip() or DEFAULT_PROMPT
    if not variation_applied:
        return base_prompt
    return f"{base_prompt}{VARIATION_SUFFIX}"


def inject_feedback_noise(source: Image.Image, frame_index: int, variation_strength: float) -> Image.Image:
    if variation_strength <= 0.0:
        return source
    sigma = 4.0 + (14.0 * clamp_variation_strength(variation_strength))
    rng = np.random.default_rng(frame_index * 7919)
    array = np.asarray(source.convert("RGB"), dtype=np.float32)
    noisy = array + rng.normal(0.0, sigma, size=array.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), mode="RGB")


def create_study_frame(
    source: Image.Image,
    generation_index: int,
    step_index: int,
    total_steps: int,
    strength: float,
    effort: int,
) -> Image.Image:
    alteration_strength = clamp_study_frame_strength(strength)
    if alteration_strength <= 0.0:
        return source.convert("RGB")

    effort_count = clamp_study_frame_effort(effort)
    progress = step_index / max(total_steps, 1)
    working = source.convert("RGB")
    width, height = working.size

    for pass_index in range(effort_count):
        pass_progress = (pass_index + 1) / effort_count
        pass_seed = (generation_index * 1009) + step_index + (pass_index * 131)
        rng = np.random.default_rng(pass_seed)

        scale = 1.0 + (0.18 * alteration_strength * progress * pass_progress)
        enlarged = working.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
        max_left = max(enlarged.width - width, 0)
        max_top = max(enlarged.height - height, 0)
        jitter_x = 0
        jitter_y = 0
        if effort_count > 1:
            jitter_x = int(round((rng.random() - 0.5) * max_left * alteration_strength))
            jitter_y = int(round((rng.random() - 0.5) * max_top * alteration_strength))
        left = min(max((max_left // 2) + jitter_x, 0), max_left)
        top = min(max((max_top // 2) + jitter_y, 0), max_top)
        reframed = enlarged.crop((left, top, left + width, top + height))
        sharpened = reframed.filter(
            ImageFilter.UnsharpMask(
                radius=1.1 + (0.8 * alteration_strength),
                percent=55 + round(85 * alteration_strength * progress * pass_progress),
                threshold=3,
            )
        )
        color_shifted = ImageEnhance.Color(sharpened).enhance(
            1.0 + (0.05 * alteration_strength * progress * pass_progress)
        )
        contrasted = ImageEnhance.Contrast(color_shifted).enhance(
            1.0 + (0.04 * alteration_strength * progress * pass_progress)
        )

        sigma = 1.0 + (2.0 * alteration_strength * progress * pass_progress)
        array = np.asarray(contrasted, dtype=np.float32)
        altered = array + rng.normal(0.0, sigma, size=array.shape)
        if effort_count > 1:
            channel_shift = max(1, round(2 * alteration_strength * pass_progress))
            altered[:, :, pass_index % 3] = np.roll(altered[:, :, pass_index % 3], channel_shift, axis=1)
        altered_image = Image.fromarray(np.clip(altered, 0, 255).astype(np.uint8), mode="RGB")
        working = Image.blend(working, altered_image, 0.35 + (0.45 * progress * pass_progress))

    return working


def variation_pulse_frames(variation_strength: float) -> int:
    return 2 + int(round(clamp_variation_strength(variation_strength) * 3))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_session_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid4().hex[:8]}"


OUTPUT_ROOT = Path(os.environ.get("CVIS_OUTPUT_DIR", Path(__file__).resolve().parents[1] / "outputs")).resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def session_output_url(session_id: str, filename: str) -> str:
    return f"/outputs/{session_id}/{filename}"


def session_dir_for(session_id: str) -> Path:
    session_dir = (OUTPUT_ROOT / session_id).resolve()
    if OUTPUT_ROOT not in session_dir.parents:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session_dir


def write_manifest(state: SessionState, ended_at: str | None = None) -> None:
    manifest = {
        "id": state.session_id,
        "created_at": state.created_at,
        "ended_at": ended_at,
        "engine": engine.name,
        "detail": engine.detail,
        "frames": state.frames,
        "seeds": state.seeds,
    }
    (state.session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_manifest(session_id: str) -> dict[str, object]:
    session_dir = session_dir_for(session_id)
    manifest_path = session_dir / "manifest.json"
    if not manifest_path.exists() or not session_dir.is_dir():
        raise HTTPException(status_code=404, detail="Session not found.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = manifest.get("frames", [])
    if isinstance(frames, list):
        for frame in frames:
            if isinstance(frame, dict) and isinstance(frame.get("filename"), str):
                frame["url"] = session_output_url(session_id, frame["filename"])
            if isinstance(frame, dict) and isinstance(frame.get("seed_filename"), str):
                frame["seed_url"] = session_output_url(session_id, frame["seed_filename"])
    seeds = manifest.get("seeds", [])
    if isinstance(seeds, list):
        for seed in seeds:
            if isinstance(seed, dict) and isinstance(seed.get("filename"), str):
                seed["url"] = session_output_url(session_id, seed["filename"])
    return manifest


app = FastAPI(title="cvisualizer-backend")
logger = logging.getLogger("uvicorn.error")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_ROOT)), name="outputs")

engine = StreamDiffusionEngine.try_create_from_env() or MockEngine()
prompt_enhancer = PromptEnhancer.from_env()
logger.info("Inference engine selected: %s (%s)", engine.name, engine.detail)
logger.info("Prompt enhancer status: %s", prompt_enhancer.detail)


@app.get("/api/status")
def get_status() -> dict[str, object]:
    return {
        "engine": engine.name,
        "model_ready": engine.model_ready,
        "detail": engine.detail,
    }


@app.get("/api/sessions")
def list_sessions() -> dict[str, list[dict[str, object]]]:
    sessions: list[dict[str, object]] = []
    for session_dir in OUTPUT_ROOT.iterdir():
        if not session_dir.is_dir():
            continue
        manifest_path = session_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = read_manifest(session_dir.name)
        except Exception:
            logger.exception("Failed to read session manifest: %s", manifest_path)
            continue

        frames = manifest.get("frames", [])
        latest_frame = frames[-1] if frames else None
        sessions.append(
            {
                "id": manifest.get("id", session_dir.name),
                "created_at": manifest.get("created_at"),
                "ended_at": manifest.get("ended_at"),
                "engine": manifest.get("engine", "unknown"),
                "frame_count": len(frames) if isinstance(frames, list) else 0,
                "thumbnail_url": latest_frame.get("url") if isinstance(latest_frame, dict) else None,
            }
        )

    sessions.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, object]:
    return read_manifest(session_id)


@app.websocket("/ws/stream")
async def stream_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    client = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown-client"
    logger.info("Stream socket accepted from %s using %s", client, engine.name)
    session_id = create_session_id()
    session_dir = OUTPUT_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    state = SessionState(
        seed_image=None,
        settings=InferenceSettings(
            prompt=DEFAULT_PROMPT,
            strength=0.85,
            running=True,
            prompt_enhancement_enabled=False,
            prompt_enhancement_interval=12,
            prompt_enhancement_strength=0.55,
            study_frames_enabled=True,
            study_frame_count=2,
            study_frame_strength=0.2,
            study_frame_effort=1,
            study_frame_delay=0.6,
        ),
        session_id=session_id,
        session_dir=session_dir,
        created_at=utc_now_iso(),
        frames=[],
        seeds=[],
    )
    write_manifest(state)

    async def produce_frames() -> None:
        while True:
            await asyncio.sleep(0.28)
            if not state.settings.running or state.seed_image is None:
                continue

            state.frame_index += 1
            logger.info(
                "Starting frame generation %s for %s (prompt=%r, strength=%.2f)",
                state.frame_index,
                client,
                state.settings.prompt,
                state.settings.strength,
            )
            variation_applied = (
                state.settings.anti_stagnation_enabled
                and state.variation_pulse_remaining > 0
                and state.settings.variation_strength > 0
            )
            (
                enhancement_base_prompt,
                refreshed_enhanced_prompt,
                prompt_enhancement_last_frame,
                prompt_enhancement_refreshed,
            ) = prompt_enhancer.enhance_if_due(
                base_prompt=state.settings.prompt,
                strength=state.settings.prompt_enhancement_strength,
                frame_index=state.frame_index,
                interval=state.settings.prompt_enhancement_interval,
                enabled=state.settings.prompt_enhancement_enabled,
                cached_prompt=state.enhanced_prompt,
                last_enhanced_frame=state.prompt_enhancement_last_frame,
            )
            state.enhanced_prompt = refreshed_enhanced_prompt
            state.prompt_enhancement_last_frame = prompt_enhancement_last_frame
            effective_prompt = build_effective_prompt(enhancement_base_prompt, variation_applied)
            transform_settings = replace(state.settings, prompt=effective_prompt)
            try:
                frame = engine.transform(state.seed_image, transform_settings, state.frame_index)
            except Exception as exc:
                logger.exception("Frame generation %s failed for %s", state.frame_index, client)
                state.settings.running = False
                await websocket.send_json(
                    {
                        "type": "error",
                        "engine": engine.name,
                        "detail": f"Frame generation failed: {exc}",
                    }
                )
                continue

            current_small = downsample_for_delta(frame)
            delta_from_previous: float | None = None
            variation_triggered = False
            if state.previous_frame_small is not None:
                delta_from_previous = frame_delta_score(state.previous_frame_small, current_small)
                if (
                    state.settings.anti_stagnation_enabled
                    and delta_from_previous < clamp_threshold(state.settings.stagnation_threshold)
                ):
                    state.stagnant_frames += 1
                else:
                    state.stagnant_frames = 0
            else:
                state.stagnant_frames = 0
            state.previous_frame_small = current_small

            if (
                state.settings.anti_stagnation_enabled
                and state.stagnant_frames >= clamp_window(state.settings.stagnation_window)
                and state.variation_pulse_remaining <= 0
            ):
                state.variation_pulse_remaining = variation_pulse_frames(state.settings.variation_strength)
                variation_triggered = True

            feedback_strength = state.settings.strength
            if variation_applied:
                modulation = 0.2 + (0.45 * clamp_variation_strength(state.settings.variation_strength))
                feedback_strength = clamp_strength(state.settings.strength * (1.0 - modulation))

            state.saved_frame_index += 1
            filename = f"frame_{state.saved_frame_index:06d}.png"
            frame_path = state.session_dir / filename
            frame.save(frame_path, format="PNG")
            frame_record = {
                "index": state.saved_frame_index,
                "kind": "generated",
                "generation_index": state.frame_index,
                "filename": filename,
                "url": session_output_url(state.session_id, filename),
                "seed_index": state.current_seed_index,
                "seed_filename": state.current_seed_filename,
                "seed_url": (
                    session_output_url(state.session_id, state.current_seed_filename)
                    if state.current_seed_filename
                    else None
                ),
                "created_at": utc_now_iso(),
                "prompt": state.settings.prompt,
                "effective_prompt": effective_prompt,
                "strength": state.settings.strength,
                "feedback_strength": feedback_strength,
                "study_frame_effort": state.settings.study_frame_effort,
                "delta_from_previous": delta_from_previous,
                "stagnant_frames": state.stagnant_frames,
                "variation_applied": variation_applied,
                "variation_triggered": variation_triggered,
                "variation_pulse_remaining": state.variation_pulse_remaining,
                "prompt_enhancement_enabled": state.settings.prompt_enhancement_enabled,
                "prompt_enhancement_interval": state.settings.prompt_enhancement_interval,
                "prompt_enhancement_strength": state.settings.prompt_enhancement_strength,
                "prompt_enhancement_refreshed": prompt_enhancement_refreshed,
                "enhanced_prompt": state.enhanced_prompt,
                "prompt_enhancement_last_frame": state.prompt_enhancement_last_frame,
            }
            state.frames.append(frame_record)
            write_manifest(state)
            next_seed = image_feedback_source(state.seed_image, frame, feedback_strength)
            if variation_applied:
                next_seed = inject_feedback_noise(next_seed, state.frame_index, state.settings.variation_strength)
                state.variation_pulse_remaining = max(state.variation_pulse_remaining - 1, 0)
            state.seed_image = next_seed
            await websocket.send_json(
                {
                    "type": "frame",
                    "frame_kind": "generated",
                    "frame": image_to_data_url(frame),
                    "session_id": state.session_id,
                    "frame_index": state.saved_frame_index,
                    "generation_index": state.frame_index,
                    "frame_url": frame_record["url"],
                    "created_at": frame_record["created_at"],
                    "engine": engine.name,
                    "detail": engine.detail,
                    "study_frame_effort": state.settings.study_frame_effort,
                    "delta_from_previous": delta_from_previous,
                    "stagnant_frames": state.stagnant_frames,
                    "variation_applied": variation_applied,
                    "variation_triggered": variation_triggered,
                    "variation_pulse_remaining": state.variation_pulse_remaining,
                    "prompt_enhancement_enabled": state.settings.prompt_enhancement_enabled,
                    "prompt_enhancement_interval": state.settings.prompt_enhancement_interval,
                    "prompt_enhancement_strength": state.settings.prompt_enhancement_strength,
                    "prompt_enhancement_refreshed": prompt_enhancement_refreshed,
                    "enhanced_prompt": state.enhanced_prompt,
                    "prompt_enhancement_last_frame": state.prompt_enhancement_last_frame,
                    "effective_prompt": effective_prompt,
                }
            )
            logger.info("Sent generated frame %s to %s", state.frame_index, client)

            if state.settings.study_frames_enabled:
                study_frame_count = clamp_study_frame_count(state.settings.study_frame_count)
                for study_step in range(1, study_frame_count + 1):
                    await asyncio.sleep(clamp_study_frame_delay(state.settings.study_frame_delay))
                    if not state.settings.running or state.seed_image is None:
                        break

                    study_frame = create_study_frame(
                        state.seed_image,
                        state.frame_index,
                        study_step,
                        study_frame_count,
                        state.settings.study_frame_strength,
                        state.settings.study_frame_effort,
                    )
                    state.seed_image = study_frame
                    state.saved_frame_index += 1
                    study_filename = f"frame_{state.saved_frame_index:06d}.png"
                    study_frame.save(state.session_dir / study_filename, format="PNG")
                    study_record = {
                        "index": state.saved_frame_index,
                        "kind": "study",
                        "generation_index": state.frame_index,
                        "study_step": study_step,
                        "study_total": study_frame_count,
                        "filename": study_filename,
                        "url": session_output_url(state.session_id, study_filename),
                        "seed_index": state.current_seed_index,
                        "seed_filename": state.current_seed_filename,
                        "seed_url": (
                            session_output_url(state.session_id, state.current_seed_filename)
                            if state.current_seed_filename
                            else None
                        ),
                        "created_at": utc_now_iso(),
                        "prompt": state.settings.prompt,
                        "effective_prompt": effective_prompt,
                        "strength": state.settings.strength,
                        "feedback_strength": feedback_strength,
                        "study_frame_effort": state.settings.study_frame_effort,
                        "delta_from_previous": None,
                        "stagnant_frames": state.stagnant_frames,
                        "variation_applied": variation_applied,
                        "variation_triggered": False,
                        "variation_pulse_remaining": state.variation_pulse_remaining,
                        "prompt_enhancement_enabled": state.settings.prompt_enhancement_enabled,
                        "prompt_enhancement_interval": state.settings.prompt_enhancement_interval,
                        "prompt_enhancement_strength": state.settings.prompt_enhancement_strength,
                        "prompt_enhancement_refreshed": False,
                        "enhanced_prompt": state.enhanced_prompt,
                        "prompt_enhancement_last_frame": state.prompt_enhancement_last_frame,
                    }
                    state.frames.append(study_record)
                    write_manifest(state)
                    await websocket.send_json(
                        {
                            "type": "frame",
                            "frame_kind": "study",
                            "frame": image_to_data_url(study_frame),
                            "session_id": state.session_id,
                            "frame_index": state.saved_frame_index,
                            "generation_index": state.frame_index,
                            "study_step": study_step,
                            "study_total": study_frame_count,
                            "frame_url": study_record["url"],
                            "created_at": study_record["created_at"],
                            "engine": engine.name,
                            "detail": engine.detail,
                            "study_frame_effort": state.settings.study_frame_effort,
                            "delta_from_previous": None,
                            "stagnant_frames": state.stagnant_frames,
                            "variation_applied": variation_applied,
                            "variation_triggered": False,
                            "variation_pulse_remaining": state.variation_pulse_remaining,
                            "prompt_enhancement_enabled": state.settings.prompt_enhancement_enabled,
                            "prompt_enhancement_interval": state.settings.prompt_enhancement_interval,
                            "prompt_enhancement_strength": state.settings.prompt_enhancement_strength,
                            "prompt_enhancement_refreshed": False,
                            "enhanced_prompt": state.enhanced_prompt,
                            "prompt_enhancement_last_frame": state.prompt_enhancement_last_frame,
                            "effective_prompt": effective_prompt,
                        }
                    )
                    logger.info(
                        "Sent study frame %s/%s after generation %s to %s",
                        study_step,
                        study_frame_count,
                        state.frame_index,
                        client,
                    )

    producer_task = asyncio.create_task(produce_frames())
    await websocket.send_json(
        {
            "type": "session",
            "session_id": state.session_id,
            "output_url": f"/outputs/{state.session_id}",
            "engine": engine.name,
            "detail": engine.detail,
        }
    )

    try:
        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")

            if message_type == "seed_frame":
                image_data = message.get("image")
                if isinstance(image_data, str):
                    try:
                        state.seed_image = data_url_to_image(image_data)
                        state.frame_index = 0
                        state.previous_frame_small = None
                        state.stagnant_frames = 0
                        state.variation_pulse_remaining = 0
                        state.enhanced_prompt = None
                        state.prompt_enhancement_last_frame = 0
                        state.saved_seed_index += 1
                        seed_filename = f"seed_{state.saved_seed_index:06d}.png"
                        state.seed_image.save(state.session_dir / seed_filename, format="PNG")
                        state.seed_image.save(state.session_dir / "seed.png", format="PNG")
                        seed_record = {
                            "index": state.saved_seed_index,
                            "filename": seed_filename,
                            "url": session_output_url(state.session_id, seed_filename),
                            "created_at": utc_now_iso(),
                        }
                        state.seeds.append(seed_record)
                        state.current_seed_index = state.saved_seed_index
                        state.current_seed_filename = seed_filename
                        write_manifest(state)
                    except Exception:
                        logger.exception("Failed to decode seed frame from %s", client)
                        continue
                    logger.info("Received seed frame from %s (%sx%s)", client, *state.seed_image.size)
                else:
                    logger.warning("Ignored seed_frame from %s with missing image data", client)

            elif message_type == "settings":
                payload = message.get("payload", {})
                previous_prompt = state.settings.prompt
                state.settings = InferenceSettings(
                    prompt=str(payload.get("prompt", state.settings.prompt)),
                    strength=clamp_strength(float(payload.get("strength", state.settings.strength))),
                    running=bool(payload.get("running", state.settings.running)),
                    anti_stagnation_enabled=bool(
                        payload.get("anti_stagnation_enabled", state.settings.anti_stagnation_enabled)
                    ),
                    stagnation_threshold=clamp_threshold(
                        float(payload.get("stagnation_threshold", state.settings.stagnation_threshold))
                    ),
                    stagnation_window=clamp_window(
                        int(payload.get("stagnation_window", state.settings.stagnation_window))
                    ),
                    variation_strength=clamp_variation_strength(
                        float(payload.get("variation_strength", state.settings.variation_strength))
                    ),
                    prompt_enhancement_enabled=bool(
                        payload.get("prompt_enhancement_enabled", state.settings.prompt_enhancement_enabled)
                    ),
                    prompt_enhancement_interval=clamp_prompt_enhancement_interval(
                        int(
                            payload.get(
                                "prompt_enhancement_interval", state.settings.prompt_enhancement_interval
                            )
                        )
                    ),
                    prompt_enhancement_strength=clamp_prompt_enhancement_strength(
                        float(
                            payload.get(
                                "prompt_enhancement_strength", state.settings.prompt_enhancement_strength
                            )
                        )
                    ),
                    study_frames_enabled=bool(
                        payload.get("study_frames_enabled", state.settings.study_frames_enabled)
                    ),
                    study_frame_count=clamp_study_frame_count(
                        int(payload.get("study_frame_count", state.settings.study_frame_count))
                    ),
                    study_frame_strength=clamp_study_frame_strength(
                        float(payload.get("study_frame_strength", state.settings.study_frame_strength))
                    ),
                    study_frame_effort=clamp_study_frame_effort(
                        int(payload.get("study_frame_effort", state.settings.study_frame_effort))
                    ),
                    study_frame_delay=clamp_study_frame_delay(
                        float(payload.get("study_frame_delay", state.settings.study_frame_delay))
                    ),
                )
                if not state.settings.anti_stagnation_enabled:
                    state.stagnant_frames = 0
                    state.variation_pulse_remaining = 0
                if (
                    not state.settings.prompt_enhancement_enabled
                    or state.settings.prompt != previous_prompt
                ):
                    state.enhanced_prompt = None
                    state.prompt_enhancement_last_frame = 0
                logger.info(
                    (
                        "Received settings from %s (running=%s, prompt=%r, strength=%.2f, "
                        "anti_stagnation=%s, threshold=%.4f, window=%s, variation=%.2f, "
                        "prompt_enhancement=%s, interval=%s, enhancer_strength=%.2f, "
                        "study_frames=%s, study_count=%s, study_strength=%.2f, "
                        "study_effort=%s, study_delay=%.2f)"
                    ),
                    client,
                    state.settings.running,
                    state.settings.prompt,
                    state.settings.strength,
                    state.settings.anti_stagnation_enabled,
                    state.settings.stagnation_threshold,
                    state.settings.stagnation_window,
                    state.settings.variation_strength,
                    state.settings.prompt_enhancement_enabled,
                    state.settings.prompt_enhancement_interval,
                    state.settings.prompt_enhancement_strength,
                    state.settings.study_frames_enabled,
                    state.settings.study_frame_count,
                    state.settings.study_frame_strength,
                    state.settings.study_frame_effort,
                    state.settings.study_frame_delay,
                )
            else:
                logger.warning("Ignored unknown message type from %s: %r", client, message_type)
    except WebSocketDisconnect:
        logger.info("Stream socket disconnected: %s", client)
    finally:
        producer_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await producer_task
        write_manifest(state, ended_at=utc_now_iso())
        logger.info("Stream producer stopped for %s", client)
