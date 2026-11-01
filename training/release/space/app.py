"""Netra-NMT — Hugging Face Space.

Source: github.com/netra-ai-lab/Netra-NMT training/release/space/.
Wraps netra_nmt's NetraTranslator (compact 77M-parameter English<->Khmer
transformer) in a Gradio UI, exposing a `/predict` API endpoint so external
callers (e.g. the website's playground) can get translations programmatically,
alongside the human-facing translator UI.
"""

import gradio as gr

from netra_nmt import NetraTranslator

_translator = None


def _get_translator() -> NetraTranslator:
    global _translator
    if _translator is None:
        _translator = NetraTranslator(device="cpu")
    return _translator


def predict(text: str, direction: str, mode: str, beam_size: int) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    translator = _get_translator()
    return translator.translate(
        text,
        direction=direction,
        mode=mode,
        beam_size=int(beam_size),
    )


theme = gr.themes.Soft(primary_hue="orange", secondary_hue="stone")

with gr.Blocks(title="Netra-NMT — English <-> Khmer Translation", theme=theme) as demo:
    gr.Markdown(
        """
        # Netra-NMT
        ### A 77M-parameter English <-> Khmer translation model
        Type a sentence, pick a direction, and translate — runs on CPU.
        """
    )

    direction_radio = gr.Radio(
        choices=[("English -> Khmer", "en2km"), ("Khmer -> English", "km2en")],
        value="en2km",
        label="Direction",
    )

    with gr.Accordion("Decoding settings", open=False):
        with gr.Row():
            mode_radio = gr.Radio(["greedy", "beam"], value="greedy", label="Decoding mode")
            beam_slider = gr.Slider(1, 5, value=5, step=1, label="Beam size", info="Only used in beam mode")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_text = gr.Textbox(label="Source", lines=10, placeholder="Type text to translate…")
            translate_btn = gr.Button("Translate", variant="primary", size="lg")
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Translation", lines=10, interactive=False)

    translate_btn.click(
        fn=predict,
        inputs=[input_text, direction_radio, mode_radio, beam_slider],
        outputs=[output_text],
    )
    input_text.submit(
        fn=predict,
        inputs=[input_text, direction_radio, mode_radio, beam_slider],
        outputs=[output_text],
    )

    gr.Markdown(
        """
        ---
        <div style="text-align: center; font-size: 0.9em; color: #666; padding: 20px 0;">
        Netra Lab | Trained from scratch on 220M tokens of English-Khmer parallel text |
        Runs on ONNX Runtime (CPU) with <a href="https://pypi.org/project/netra-nmt/" target="_blank">netra-nmt</a> |
        <a href="https://huggingface.co/Netra-Lab/Netra-NMT" target="_blank">View model card</a> |
        <a href="https://github.com/netra-ai-lab/Netra-NMT" target="_blank">GitHub</a>
        </div>
        """
    )

if __name__ == "__main__":
    demo.queue().launch()
