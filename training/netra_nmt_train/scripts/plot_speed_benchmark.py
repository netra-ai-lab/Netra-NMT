"""
plot_speed_benchmark.py — render the inference-speed benchmark in Netra brand
style (matching scripts/benchmark_metrics.ipynb + assets/inference_benchmark.png),
saved to assets/speed_benchmark.png.

Reads results/speed_hw.json (produced by scripts/benchmark_speed_hw.py) and
draws, with a left-side vertical legend:
  * a CPU row and a GPU row, each with a Greedy and a Beam-5 tok/s panel, and
  * a full-width "Model size" bar chart (fp16, MB).
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[3]


# ── Brand palette / helpers (identical to benchmark_metrics.ipynb) ──────────
def _warm_gray(level):
    r = min(1.0, level + 0.03); g = min(1.0, level - 0.01); b = min(1.0, level - 0.04)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

C_PRIMARY = "#e75f1b"
C_BG = "#ffffff"; C_PANEL = "#ffffff"; C_TEXT = "#1c1c1c"; C_MUTED = "#888888"

MODEL_COLORS = {
    "Netra-NMT-77M (Ours)": C_PRIMARY,
    "NLLB-350M (EN-KM)":    _warm_gray(0.55),
    "NLLB-600M":            _warm_gray(0.66),
}
MODEL_ORDER = list(MODEL_COLORS)

# Total parameters (M) and fp16 weight size (MB = params * 2 bytes) — a fair,
# constant-precision size comparison (Netra ships fp16; the NLLB downloads are
# fp32 and correspondingly ~2x larger on disk).
PARAMS_M = {"Netra-NMT-77M (Ours)": 77.2, "NLLB-350M (EN-KM)": 350.5, "NLLB-600M": 615.1}
SIZE_MB = {m: p * 2.0 for m, p in PARAMS_M.items()}

DISPLAY = {
    "Netra-NMT-77M (Ours)": "Netra-NMT (Ours) — 77M",
    "NLLB-350M (EN-KM)":    "NLLB-350M — 351M",
    "NLLB-600M":            "NLLB-600M — 615M",
}
SHORT = {"Netra-NMT-77M (Ours)": "Netra-NMT (Ours)",
         "NLLB-350M (EN-KM)": "NLLB-350M", "NLLB-600M": "NLLB-600M"}

_FONTS = ["Segoe UI", "DejaVu Sans", "Noto Sans Khmer"]
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = _FONTS

DEVICES = [("cpu", "CPU · i7-13700KF"), ("cuda", "GPU · RTX 3060")]
MODES = [("greedy", "Greedy (tok/s)"), ("beam", "Beam-5 (tok/s)")]


def _spine_clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e0e0e0")
    ax.spines["bottom"].set_color("#e0e0e0")


def main():
    data = json.loads((REPO_ROOT / "results" / "speed_hw.json").read_text())
    idx = {(r["model"], r["device"]): r for r in data if "error" not in r}
    devices = [d for d in DEVICES if any(k[1] == d[0] for k in idx)]

    ndev = len(devices)
    fig = plt.figure(figsize=(13, 4.6 * ndev + 3.4), facecolor=C_BG)
    gs = fig.add_gridspec(
        ndev + 1, 2, height_ratios=[1] * ndev + [0.8],
        top=0.88, bottom=0.06, left=0.26, right=0.97, hspace=0.60, wspace=0.17,
    )
    fig.suptitle("Netra-NMT Inference Benchmark",
                 fontsize=18, fontweight="bold", color=C_TEXT, x=0.615, y=0.965)

    # ── Left-side vertical legend: hardware header + model dots ─────────────
    fig.text(0.035, 0.74, "Intel i7-13700KF\nNVIDIA RTX 3060",
             fontsize=12, fontweight="bold", color=C_TEXT, va="top")
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=MODEL_COLORS[m], markersize=14,
                      label=DISPLAY[m], markeredgewidth=0) for m in MODEL_ORDER]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.028, 0.5),
               ncol=1, frameon=False, fontsize=12, labelcolor=C_TEXT,
               handletextpad=0.6, labelspacing=1.4, borderpad=0.0)

    # ── Speed panels (device rows × mode cols) ─────────────────────────────
    for ri, (dev, dev_label) in enumerate(devices):
        for ci, (mode, mode_label) in enumerate(MODES):
            ax = fig.add_subplot(gs[ri, ci])
            present = [m for m in MODEL_ORDER if (m, dev) in idx]
            vals = [idx[(m, dev)][f"{mode}_tok_s"] for m in present]
            bars = ax.bar(range(len(present)), vals, width=0.62,
                          color=[MODEL_COLORS[m] for m in present], linewidth=0)
            y_max = max(vals)
            ax.set_ylim(0, y_max * 1.20)
            for b, m in zip(bars, present):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + y_max * 0.03,
                        f"{b.get_height():.1f}", ha="center", va="bottom",
                        fontsize=11, fontweight="bold", color=MODEL_COLORS[m])
            ax.set_facecolor(C_PANEL)
            ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=9, colors=C_MUTED, length=0, pad=3)
            ax.yaxis.grid(True, color="#efefef", linewidth=0.7, zorder=0)
            ax.set_axisbelow(True)
            _spine_clean(ax)
            ax.set_xlabel(mode_label, fontsize=12.5, fontweight="bold",
                          color=C_TEXT, labelpad=14)
            ax.text(0.5, -0.235, "Higher Better", transform=ax.transAxes,
                    ha="center", va="top", fontsize=9.5, color=C_MUTED, style="italic")
            if ci == 0:
                ax.set_ylabel(dev_label, fontsize=12.5, fontweight="bold",
                              color=C_TEXT, labelpad=12)

    # ── Model-size bar chart (full width, horizontal) ──────────────────────
    ax_sz = fig.add_subplot(gs[ndev, :])
    order = MODEL_ORDER[::-1]                       # first model on top
    ypos = list(range(len(order)))
    sizes = [SIZE_MB[m] for m in order]
    ax_sz.barh(ypos, sizes, height=0.6,
               color=[MODEL_COLORS[m] for m in order], linewidth=0)
    xmax = max(sizes)
    for y, m in zip(ypos, order):
        ax_sz.text(SIZE_MB[m] + xmax * 0.012, y, f"{SIZE_MB[m]:.0f} MB",
                   va="center", ha="left", fontsize=11, fontweight="bold",
                   color=MODEL_COLORS[m])
    ax_sz.set_xlim(0, xmax * 1.18)
    ax_sz.set_yticks(ypos)
    ax_sz.set_yticklabels([SHORT[m] for m in order], fontsize=10.5, color=C_TEXT)
    ax_sz.set_facecolor(C_PANEL)
    ax_sz.tick_params(axis="x", labelsize=9, colors=C_MUTED, length=0, pad=3)
    ax_sz.tick_params(axis="y", length=0, pad=6)
    ax_sz.xaxis.grid(True, color="#efefef", linewidth=0.7, zorder=0)
    ax_sz.set_axisbelow(True)
    _spine_clean(ax_sz)
    ax_sz.set_xlabel("Model size (fp16, MB)", fontsize=12.5, fontweight="bold",
                     color=C_TEXT, labelpad=14)
    ax_sz.text(0.5, -0.42, "Smaller is lighter — Netra is ~4.5× / 8× smaller than the NLLB baselines",
               transform=ax_sz.transAxes, ha="center", va="top",
               fontsize=9.5, color=C_MUTED, style="italic")

    out = REPO_ROOT / "assets" / "speed_benchmark.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=C_BG)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
