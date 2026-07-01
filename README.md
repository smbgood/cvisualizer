# cvisualizer

Local Windows prototype for interactive AI image visualization:

- Draw pixels or import a seed image.
- Stream continuously transforming frames in the main display.
- Scrub through generated frames on a timeline.
- Capture snapshots and review them as thumbnails.
- Re-open saved output sessions from separate local folders.

## Architecture

- Frontend: React + Vite + TypeScript (`src/`)
- Backend: FastAPI + WebSocket stream (`backend/`)
- Inference engine:
  - Primary: StreamDiffusion + SD-Turbo (`backend/inference/streamdiffusion_engine.py`)
  - Fallback: deterministic mock engine (`backend/inference/mock_engine.py`)

## Requirements (Windows)

- Python 3.11
- Node.js 20+
- NVIDIA GPU (8GB VRAM target)
- Latest NVIDIA driver + CUDA-compatible PyTorch install

## Quick Start

### 1) Frontend

```powershell
npm install
npm run dev
```

Frontend runs at <http://127.0.0.1:5173>.

### 2) Backend

Create a virtual environment and install backend dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

Run FastAPI:

```powershell
uvicorn backend.app:app --reload --port 8000
```

Backend status endpoint: <http://127.0.0.1:8000/api/status>

Generated frames are saved under `outputs/<session-id>/` by default. Each folder includes `manifest.json`,
versioned seed snapshots (`seed_000001.png`, `seed_000002.png`, ...), a latest-seed alias (`seed.png`),
and numbered frame PNGs. The manifest links each frame to the active seed snapshot so restoring earlier
timeline points can also restore the corresponding input seed.

Optional output location:

```powershell
$env:CVIS_OUTPUT_DIR="D:\cvisualizer-runs"
uvicorn backend.app:app --reload --port 8000
```

## StreamDiffusion Setup

The backend automatically tries to initialize StreamDiffusion. If initialization fails, it falls back to `mock-engine` so the UI still works.

Optional environment variables:

- `CVIS_MODEL_ID` (default: `stabilityai/sd-turbo`)
- `CVIS_DEVICE` (default: `cuda`)
- `CVIS_T_INDEX_LIST` (default: `0,16,32`; lower values reinterpret the source more aggressively)
- `CVIS_CFG_TYPE` (default: `none`)
- `CVIS_GUIDANCE_SCALE` (default: `0.0`)
- `CVIS_PROMPT_ENHANCER_MODEL` (default: `google/flan-t5-small`)
- `CVIS_PROMPT_ENHANCER_DEVICE` (`auto`, `cpu`, or `cuda`)
- `CVIS_PROMPT_ENHANCER_MAX_NEW_TOKENS` (default: `60`)
- `CVIS_PROMPT_ENHANCER_LOCAL_ONLY` (`1` to avoid model downloads and use local cache only)

Example:

```powershell
$env:CVIS_MODEL_ID="stabilityai/sd-turbo"
$env:CVIS_DEVICE="cuda"
$env:CVIS_T_INDEX_LIST="0,16,32"
uvicorn backend.app:app --reload --port 8000
```

## 8GB VRAM Defaults

Current conservative defaults for local streaming:

- Resolution: 512x512
- Batch: 1
- Precision: FP16
- Inference steps: `0,16,32` by default for stronger image-to-image reinterpretation
- Feedback: each generated frame becomes the next source image, weighted by the Strength control

## Anti-Stagnation Controls

The generation loop can naturally settle into a low-change "burn-in" state. The app now includes a bounded anti-stagnation pulse that activates when frame-to-frame change stays below a threshold for several frames.

- **Enable anti-stagnation**: toggles automatic intervention
- **Stagnation threshold**: lower values trigger less often, higher values trigger sooner
- **Detection window**: number of consecutive low-delta frames before intervention
- **Variation strength**: controls how strong each intervention pulse is

When triggered, the backend briefly applies:

- a temporary prompt variation suffix
- reduced feedback strength (so new output is not overly anchored)
- small deterministic noise on the feedback seed

Per-frame diagnostics (`delta_from_previous`, `stagnant_frames`, `variation_applied`) are written to each session `manifest.json`.

## Prompt Enhancement Controls

The streaming loop can now periodically rewrite your prompt to keep outputs more descriptive and quality-focused.

- **Enable prompt enhancement**: turns periodic prompt rewriting on/off
- **Prompt refresh interval**: number of generated frames between prompt refreshes
- **Prompt enhancement strength**: controls how aggressively the rewritten prompt diverges from your base prompt

Behavior notes:

- Prompt enhancement runs locally in the backend through `transformers` and lazy-loads only when enabled.
- If the configured model is unavailable, the backend falls back to a deterministic quality/descriptiveness suffix strategy.
- `effective_prompt` and prompt-enhancement diagnostics are recorded per frame in each session `manifest.json`.

## Troubleshooting

- **`engine: mock-engine` in `/api/status`**
  - StreamDiffusion or model dependencies are missing, failed to import, or failed to initialize.
  - Confirm your Python environment has `torch`, `diffusers`, and `streamdiffusion`.
- **CUDA unavailable**
  - Verify NVIDIA drivers and a CUDA-enabled PyTorch build.
  - Check `python -c "import torch; print(torch.cuda.is_available())"`.
- **OOM on 8GB GPU**
  - Keep model to SD-Turbo/SD1.5 class and avoid increasing resolution above 512 for live loop.
  - Close other GPU-heavy applications.
- **Frontend not receiving frames**
  - Confirm backend is running on port `8000`.
  - Confirm browser can connect to `ws://127.0.0.1:8000/ws/stream`.

## Notes

- Snapshot capture stores the current displayed frame, not the original seed.
- Snapshot gallery is session-local (in-memory) in this MVP.
- Timeline frames are persisted to local session folders and served by the backend.
