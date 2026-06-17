"""
netra_nmt.webapp — Streamlit web demo for the netra-nmt translator.

Launch with the console script (wraps ``streamlit run``)::

    netra-web                         # http://127.0.0.1:8501
    netra-web --port 8600 --device cpu
    netra-web --local-dir export      # load weights from a local export dir

or directly with Streamlit (pass the installed module's file path)::

    streamlit run "$(python -c 'import netra_nmt.webapp as w; print(w.__file__)')"

Configuration is passed to the Streamlit script through environment variables
(``NETRA_NMT_REPO_ID`` / ``NETRA_NMT_LOCAL_DIR`` / ``NETRA_NMT_DEVICE``).
"""

from __future__ import annotations

import argparse
import os
import sys

# Absolute imports: Streamlit executes this file as a top-level script (no
# package parent), so relative imports would fail. The package is installed,
# so `netra_nmt.*` resolves whether run as a module or by file path.
from netra_nmt.config import DEFAULT_DIRECTION, DIRECTIONS
from netra_nmt.translator import NetraTranslator

_DIRECTION_LABELS = {
    "en2km": "English → Khmer",
    "km2en": "Khmer → English",
}


def _load_translator() -> NetraTranslator:
    import streamlit as st

    @st.cache_resource(show_spinner="Loading netra-nmt model…")
    def _build():
        return NetraTranslator(
            repo_id=os.environ.get("NETRA_NMT_REPO_ID") or None,
            local_dir=os.environ.get("NETRA_NMT_LOCAL_DIR") or None,
            device=os.environ.get("NETRA_NMT_DEVICE") or None,
        )

    return _build()


def render() -> None:
    """Render the Streamlit UI. Executed on every script run by Streamlit."""
    import streamlit as st

    st.set_page_config(page_title="netra-nmt", page_icon="🇰🇭", layout="centered")
    st.title("netra-nmt — English ↔ Khmer Translation")
    st.caption("A compact from-scratch NMT model for English↔Khmer.")

    label_to_dir = {v: k for k, v in _DIRECTION_LABELS.items()}

    with st.sidebar:
        st.header("Settings")
        direction_label = st.selectbox(
            "Direction",
            [_DIRECTION_LABELS[d] for d in sorted(DIRECTIONS)],
            index=sorted(DIRECTIONS).index(DEFAULT_DIRECTION),
        )
        mode = st.selectbox("Decoding", ["greedy", "beam", "sample"], index=0)
        beam_size = st.slider("Beam size", 1, 10, 5, disabled=(mode != "beam"))
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1,
                                disabled=(mode != "sample"))
        top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05, disabled=(mode != "sample"))
        max_new_tokens = st.slider("Max output tokens", 16, 256, 128, 8)

    direction = label_to_dir[direction_label]
    translator = _load_translator()

    text = st.text_area("Source text", height=160, placeholder="Type text to translate…")

    if st.button("Translate", type="primary") or text:
        if text.strip():
            with st.spinner("Translating…"):
                result = translator.translate(
                    text, direction=direction, mode=mode, beam_size=int(beam_size),
                    temperature=float(temperature), top_p=float(top_p),
                    max_new_tokens=int(max_new_tokens),
                )
            st.text_area("Translation", value=result, height=160)


def main(argv=None) -> None:
    """Console entry point: launch ``streamlit run`` on this module."""
    p = argparse.ArgumentParser(prog="netra-web", description="Launch the netra-nmt web demo.")
    p.add_argument("--repo-id", type=str, default=None)
    p.add_argument("--local-dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8501)
    args = p.parse_args(argv)

    # Pass configuration to the Streamlit-run process via the environment.
    if args.repo_id:
        os.environ["NETRA_NMT_REPO_ID"] = args.repo_id
    if args.local_dir:
        os.environ["NETRA_NMT_LOCAL_DIR"] = args.local_dir
    if args.device:
        os.environ["NETRA_NMT_DEVICE"] = args.device

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit", "run", os.path.abspath(__file__),
        "--server.address", args.host,
        "--server.port", str(args.port),
    ]
    sys.exit(stcli.main())


# Streamlit runs this file as "__main__"; the entry point imports it as a module.
if __name__ == "__main__":
    render()
