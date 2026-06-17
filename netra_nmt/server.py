"""
netra_nmt.server — FastAPI backend + web UI for the netra-nmt translator.

Exposes a small REST API and serves a two-pane translation website
(source on the left, output on the right, with an EN⇄KM swap button).

Run with the console script::

    netra-web                          # http://127.0.0.1:8000
    netra-web --port 8080 --device cpu
    netra-web --local-dir export       # load weights from a local export dir

REST API
--------
    POST /api/translate
        {"text": "Hello", "direction": "en2km", "mode": "greedy"}
      → {"translation": "...", "direction": "en2km"}

    GET /api/health  →  {"status": "ok", "device": "cpu"}

Configuration is read from the environment at startup
(``NETRA_NMT_REPO_ID`` / ``NETRA_NMT_LOCAL_DIR`` / ``NETRA_NMT_DEVICE``).
"""

from __future__ import annotations

import argparse
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import DEFAULT_DIRECTION, DIRECTIONS
from .translator import NetraTranslator

_STATIC = Path(__file__).resolve().parent / "static"

# Loaded once on startup (see lifespan).
_state: dict = {"translator": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["translator"] = NetraTranslator(
        repo_id=os.environ.get("NETRA_NMT_REPO_ID") or None,
        local_dir=os.environ.get("NETRA_NMT_LOCAL_DIR") or None,
        device=os.environ.get("NETRA_NMT_DEVICE") or None,
    )
    yield
    _state["translator"] = None


app = FastAPI(title="netra-nmt", version="0.1.0", lifespan=lifespan)


class TranslateRequest(BaseModel):
    text: str
    direction: str = DEFAULT_DIRECTION
    mode: str = Field(default="greedy", pattern="^(greedy|beam|sample)$")
    beam_size: int = Field(default=5, ge=1, le=10)
    max_new_tokens: int = Field(default=128, ge=1, le=256)


class TranslateResponse(BaseModel):
    translation: str
    direction: str


def _translator() -> NetraTranslator:
    t = _state["translator"]
    if t is None:  # pragma: no cover - only if called before startup
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return t


@app.post("/api/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest) -> TranslateResponse:
    if req.direction not in DIRECTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown direction {req.direction!r}.")
    text = req.text.strip()
    if not text:
        return TranslateResponse(translation="", direction=req.direction)
    out = _translator().translate(
        text, direction=req.direction, mode=req.mode,
        beam_size=req.beam_size, max_new_tokens=req.max_new_tokens,
    )
    return TranslateResponse(translation=out, direction=req.direction)


@app.get("/api/health")
def health() -> dict:
    t = _state["translator"]
    return {"status": "ok" if t is not None else "loading",
            "device": str(t.device) if t is not None else None}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (_STATIC / "index.html").read_text(encoding="utf-8")


def main(argv=None) -> None:
    """Console entry point: serve the FastAPI app with uvicorn."""
    p = argparse.ArgumentParser(prog="netra-web", description="Serve the netra-nmt web app.")
    p.add_argument("--repo-id", type=str, default=None)
    p.add_argument("--local-dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    if args.repo_id:
        os.environ["NETRA_NMT_REPO_ID"] = args.repo_id
    if args.local_dir:
        os.environ["NETRA_NMT_LOCAL_DIR"] = str(args.local_dir)
    if args.device:
        os.environ["NETRA_NMT_DEVICE"] = args.device

    import uvicorn

    uvicorn.run("netra_nmt.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
