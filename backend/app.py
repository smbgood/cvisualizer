from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

try:
    from backend.inference import InferenceSettings, MockEngine, StreamDiffusionEngine
except Exception:
    try:
        from .inference import InferenceSettings, MockEngine, StreamDiffusionEngine
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
    frame_index: int = 0


def clamp_strength(value: float) -> float:
    return min(max(value, 0.05), 1.0)


def image_feedback_source(previous: Image.Image, generated: Image.Image, strength: float) -> Image.Image:
    base = previous.convert("RGB").resize(generated.size)
    feedback_weight = clamp_strength(strength)
    return Image.blend(base, generated.convert("RGB"), feedback_weight)


app = FastAPI(title="cvisualizer-backend")
logger = logging.getLogger("uvicorn.error")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = StreamDiffusionEngine.try_create_from_env() or MockEngine()
logger.info("Inference engine selected: %s (%s)", engine.name, engine.detail)


@app.get("/api/status")
def get_status() -> dict[str, object]:
    return {
        "engine": engine.name,
        "model_ready": engine.model_ready,
        "detail": engine.detail,
    }


@app.websocket("/ws/stream")
async def stream_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    client = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown-client"
    logger.info("Stream socket accepted from %s using %s", client, engine.name)
    state = SessionState(
        seed_image=None,
        settings=InferenceSettings(prompt="radical surreal neon transformation", strength=0.85, running=True),
    )

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
            try:
                frame = engine.transform(state.seed_image, state.settings, state.frame_index)
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

            state.seed_image = image_feedback_source(state.seed_image, frame, state.settings.strength)
            await websocket.send_json(
                {
                    "type": "frame",
                    "frame": image_to_data_url(frame),
                    "engine": engine.name,
                    "detail": engine.detail,
                }
            )
            logger.info("Sent generated frame %s to %s", state.frame_index, client)

    producer_task = asyncio.create_task(produce_frames())
    await websocket.send_json({"engine": engine.name, "detail": engine.detail})

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
                    except Exception:
                        logger.exception("Failed to decode seed frame from %s", client)
                        continue
                    logger.info("Received seed frame from %s (%sx%s)", client, *state.seed_image.size)
                else:
                    logger.warning("Ignored seed_frame from %s with missing image data", client)

            elif message_type == "settings":
                payload = message.get("payload", {})
                state.settings = InferenceSettings(
                    prompt=str(payload.get("prompt", state.settings.prompt)),
                    strength=clamp_strength(float(payload.get("strength", state.settings.strength))),
                    running=bool(payload.get("running", state.settings.running)),
                )
                logger.info(
                    "Received settings from %s (running=%s, prompt=%r, strength=%.2f)",
                    client,
                    state.settings.running,
                    state.settings.prompt,
                    state.settings.strength,
                )
            else:
                logger.warning("Ignored unknown message type from %s: %r", client, message_type)
    except WebSocketDisconnect:
        logger.info("Stream socket disconnected: %s", client)
    finally:
        producer_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await producer_task
        logger.info("Stream producer stopped for %s", client)
