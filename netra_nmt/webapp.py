"""
netra_nmt.webapp — Gradio web demo for the netra-nmt translator.

Launch with the console script::

    netra-web

or programmatically::

    from netra_nmt.webapp import build_demo
    build_demo().launch()
"""

from __future__ import annotations

import argparse

from .config import DEFAULT_DIRECTION, DIRECTIONS
from .translator import NetraTranslator

_DIRECTION_LABELS = {
    "en2km": "English → Khmer",
    "km2en": "Khmer → English",
}


def build_demo(translator: NetraTranslator | None = None):
    import gradio as gr

    translator = translator or NetraTranslator()

    label_to_dir = {v: k for k, v in _DIRECTION_LABELS.items()}

    def _run(text: str, direction_label: str, mode: str, beam_size: int):
        if not text or not text.strip():
            return ""
        direction = label_to_dir.get(direction_label, DEFAULT_DIRECTION)
        return translator.translate(
            text, direction=direction, mode=mode, beam_size=int(beam_size)
        )

    demo = gr.Interface(
        fn=_run,
        inputs=[
            gr.Textbox(lines=4, label="Source text", placeholder="Type text to translate…"),
            gr.Dropdown(
                choices=[_DIRECTION_LABELS[d] for d in sorted(DIRECTIONS)],
                value=_DIRECTION_LABELS[DEFAULT_DIRECTION],
                label="Direction",
            ),
            gr.Dropdown(choices=["greedy", "beam", "sample"], value="greedy", label="Decoding"),
            gr.Slider(1, 10, value=5, step=1, label="Beam size (beam mode only)"),
        ],
        outputs=gr.Textbox(label="Translation"),
        title="netra-nmt — English ↔ Khmer Translation",
        description="A compact from-scratch NMT model for English↔Khmer.",
    )
    return demo


def main(argv=None):
    p = argparse.ArgumentParser(prog="netra-web", description="Launch the netra-nmt web demo.")
    p.add_argument("--repo-id", type=str, default=None)
    p.add_argument("--local-dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    args = p.parse_args(argv)

    translator = NetraTranslator(
        repo_id=args.repo_id, local_dir=args.local_dir, device=args.device
    )
    build_demo(translator).launch(
        server_name=args.host, server_port=args.port, share=args.share
    )


if __name__ == "__main__":
    main()
