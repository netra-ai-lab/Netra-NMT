"""
netra_nmt.server — FastAPI backend + web UI for the netra-nmt translator.

Exposes a small REST API and serves a two-pane translation website
(source on the left, output on the right, with an EN⇄KM swap button).

Run with the console script::

    netra-web                          # http://127.0.0.1:8000
    netra-web --port 8080 --device cpu
    netra-web --local-dir export       # load weights from a local export dir

    # also serve the experimental diffusion model (opt-in):
    netra-web --diffusion-local-dir export_diffusion

REST API
--------
    POST /api/translate
        {"text": "Hello", "direction": "en2km", "mode": "greedy"}
      → {"translation": "...", "direction": "en2km"}

        {"text": "Hello", "direction": "en2km", "model_type": "diffusion", "decode_steps": 10}
      → {"translation": "...", "direction": "en2km"}
      (400 if the diffusion model wasn't configured at startup)

    GET /api/health  →  {"status": "ok", "device": "cpu", "diffusion_available": false}

Configuration is read from the environment at startup
(``NETRA_NMT_REPO_ID`` / ``NETRA_NMT_LOCAL_DIR`` / ``NETRA_NMT_DEVICE`` for the
AR model; ``NETRA_NMT_DIFFUSION_REPO_ID`` / ``NETRA_NMT_DIFFUSION_LOCAL_DIR``
for the optional diffusion model — only loaded if one of these is set).
"""

from __future__ import annotations

import argparse
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import DEFAULT_DIRECTION, DIRECTIONS
from .diffusion_translator import DEFAULT_DECODE_STEPS, DiffusionTranslator
from .translator import NetraTranslator

_STATIC = Path(__file__).resolve().parent / "static"

# Loaded once on startup (see lifespan).
_state: dict = {"translator": None, "diffusion_translator": None}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["translator"] = NetraTranslator(
        repo_id=os.environ.get("NETRA_NMT_REPO_ID") or None,
        local_dir=os.environ.get("NETRA_NMT_LOCAL_DIR") or None,
        device=os.environ.get("NETRA_NMT_DEVICE") or None,
    )

    # Diffusion model is opt-in (extra memory + startup time for a model most
    # deployments won't use): loaded only if explicitly enabled, even though
    # weights are published at the same default-repo pattern as the AR model.
    diff_repo = os.environ.get("NETRA_NMT_DIFFUSION_REPO_ID") or None
    diff_dir = os.environ.get("NETRA_NMT_DIFFUSION_LOCAL_DIR") or None
    diff_enabled = bool(diff_repo or diff_dir) or _env_flag("NETRA_NMT_ENABLE_DIFFUSION")
    if diff_enabled:
        _state["diffusion_translator"] = DiffusionTranslator(
            repo_id=diff_repo, local_dir=diff_dir,
            device=os.environ.get("NETRA_NMT_DEVICE") or None,
        )

    yield
    _state["translator"] = None
    _state["diffusion_translator"] = None


app = FastAPI(title="netra-nmt", version="0.1.0", lifespan=lifespan)


class TranslateRequest(BaseModel):
    text: str
    direction: str = DEFAULT_DIRECTION
    model_type: str = Field(default="ar", pattern="^(ar|diffusion)$")
    mode: str = Field(default="greedy", pattern="^(greedy|beam|sample)$")
    beam_size: int = Field(default=5, ge=1, le=10)
    max_new_tokens: int = Field(default=128, ge=1, le=256)
    decode_steps: int = Field(default=DEFAULT_DECODE_STEPS, ge=1, le=64)


class TranslateResponse(BaseModel):
    translation: str
    direction: str


def _translator() -> NetraTranslator:
    t = _state["translator"]
    if t is None:  # pragma: no cover - only if called before startup
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return t


def _diffusion_translator() -> DiffusionTranslator:
    t = _state["diffusion_translator"]
    if t is None:
        raise HTTPException(
            status_code=400,
            detail="Diffusion model not configured on this server. Start it with "
                   "NETRA_NMT_DIFFUSION_LOCAL_DIR=<export dir> or "
                   "--diffusion-local-dir (see scripts/export_diffusion_checkpoint.py).",
        )
    return t


@app.post("/api/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest) -> TranslateResponse:
    if req.direction not in DIRECTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown direction {req.direction!r}.")
    text = req.text.strip()
    if not text:
        return TranslateResponse(translation="", direction=req.direction)
    if req.model_type == "diffusion":
        out = _diffusion_translator().translate(
            text, direction=req.direction, steps=req.decode_steps,
        )
    else:
        out = _translator().translate(
            text, direction=req.direction, mode=req.mode,
            beam_size=req.beam_size, max_new_tokens=req.max_new_tokens,
        )
    return TranslateResponse(translation=out, direction=req.direction)


@app.get("/api/health")
def health() -> dict:
    t = _state["translator"]
    dt = _state["diffusion_translator"]
    return {"status": "ok" if t is not None else "loading",
            "device": str(t.device) if t is not None else None,
            "diffusion_available": dt is not None}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (_STATIC / "index.html").read_text(encoding="utf-8")


# Serve brand assets (logo, wordmark) and any other static files.
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


def main(argv=None) -> None:
    """Console entry point: serve the FastAPI app with uvicorn."""
    p = argparse.ArgumentParser(prog="netra-web", description="Serve the netra-nmt web app.")
    p.add_argument("--repo-id", type=str, default=None)
    p.add_argument("--local-dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--enable-diffusion", action="store_true", default=False,
                   help="Also load the experimental diffusion model from its published "
                        "default repo (Darayut/netra-nmt-small-diffusion). Off by default "
                        "since it's a second ~90M model's worth of extra memory/startup "
                        "time that most deployments won't use.")
    p.add_argument("--diffusion-repo-id", type=str, default=None,
                   help="Enable the diffusion model from a specific Hub repo "
                        "(implies --enable-diffusion).")
    p.add_argument("--diffusion-local-dir", type=str, default=None,
                   help="Enable the diffusion model from a local export instead "
                        "(see scripts/export_diffusion_checkpoint.py; implies --enable-diffusion).")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    if args.repo_id:
        os.environ["NETRA_NMT_REPO_ID"] = args.repo_id
    if args.local_dir:
        os.environ["NETRA_NMT_LOCAL_DIR"] = str(args.local_dir)
    if args.device:
        os.environ["NETRA_NMT_DEVICE"] = args.device
    if args.enable_diffusion:
        os.environ["NETRA_NMT_ENABLE_DIFFUSION"] = "1"
    if args.diffusion_repo_id:
        os.environ["NETRA_NMT_DIFFUSION_REPO_ID"] = args.diffusion_repo_id
    if args.diffusion_local_dir:
        os.environ["NETRA_NMT_DIFFUSION_LOCAL_DIR"] = str(args.diffusion_local_dir)

    import uvicorn

    uvicorn.run("netra_nmt.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
