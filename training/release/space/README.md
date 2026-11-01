---
title: Netra NMT
emoji: 👁️
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
license: mit
models:
- Netra-Lab/Netra-NMT
---

# Netra NMT

A 77M-parameter English↔Khmer translation model, running on ONNX Runtime on this Space's CPU.
See the [model card](https://huggingface.co/Netra-Lab/Netra-NMT) and the
[GitHub repo](https://github.com/netra-ai-lab/Netra-NMT).

The `/predict` API takes `text`, `direction` (`en2km` or `km2en`), `mode` (`greedy` or `beam`)
and `beam_size`, and returns the translation.
