"""Draw the annotator's data-flow diagram."""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = sys.argv[1]
INK, MUTE, FAINT = "#222222", "#6b6b6b", "#a8a8a8"
MONO = "DejaVu Sans Mono"
SRC = {"npz": "#2c6fbb", "log": "#b8860b", "proj": "#5c5c5c"}
ARR = {"thumbnail": "#444444", "feature_heatmap": "#c0392b", "tissue_mask": "#1b7f3b",
       "gm_mask": "#6a1b9a", "wm_mask": "#00838f", "_log.pkl": SRC["log"]}
PANEL = {"1": "#6a1b9a", "2": "#1b7f3b", "3": "#c0392b", "4": "#00838f"}

fig, ax = plt.subplots(figsize=(17, 10.6))
ax.set_xlim(0, 17); ax.set_ylim(0, 10.6); ax.axis("off")
fig.patch.set_facecolor("white")

def box(x, y, w, h, fc="white", ec=MUTE, lw=1.2, r=0.10, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=1, linestyle=ls))

def txt(x, y, s, size=10, w="normal", c=INK, ha="left", style="normal", f=None):
    ax.text(x, y, s, fontsize=size, fontweight=w, color=c, ha=ha, va="center",
            style=style, family=f, zorder=5)

def chip(x, y, label, colour, size=8.3):
    w = 0.095 * len(label) + 0.30
    box(x, y - 0.145, w, 0.29, fc=colour + "1f", ec=colour, lw=0.9, r=0.145)
    txt(x + w / 2, y, label, size=size, c=colour, ha="center", w="bold", f=MONO)
    return w

# ---------------------------------------------------------------- title ----
txt(0.35, 10.22, "SANA ROI Annotator", size=19, w="bold")
txt(0.35, 9.82, "what is loaded, and what each function uses it for", size=11, c=MUTE)

# ------------------------------------------------------- loaded at start ----
box(0.35, 1.45, 4.25, 7.85, fc="#fafafa", ec="#d8d8d8")
txt(0.60, 9.05, "LOADED AT START", size=9.5, w="bold", c=MUTE)

box(0.60, 6.35, 3.75, 2.45, fc="white", ec=SRC["npz"], lw=1.6)
txt(0.80, 8.50, "<slide>.npz", size=11.5, w="bold", c=SRC["npz"], f=MONO)
txt(0.80, 8.21, "per-slide archive from the pipeline", size=8.4, c=MUTE, style="italic")
y = 7.87
for name in ("thumbnail", "feature_heatmap", "tissue_mask", "gm_mask", "wm_mask"):
    chip(0.85, y, name, ARR[name]); y -= 0.34

box(0.60, 5.53, 3.75, 0.62, fc="#f7f7f7", ec="#dddddd", ls=(0, (3, 2)))
txt(0.80, 5.96, "also in the archive, not loaded", size=8, c=FAINT, style="italic")
txt(0.80, 5.71, "cells   ·   gmwm_contours", size=8.2, c=FAINT, f=MONO)

box(0.60, 4.02, 3.75, 1.18, fc="white", ec=SRC["log"], lw=1.6)
txt(0.80, 4.92, "<slide>_log.pkl", size=11.5, w="bold", c=SRC["log"], f=MONO)
txt(0.80, 4.63, "scan parameters - used only by (4)", size=8.4, c=MUTE, style="italic")
txt(0.80, 4.31, "mpp · ds · thumbnail_level · ds_thumbnail", size=7.4, c=SRC["log"], f=MONO)

box(0.60, 1.70, 3.75, 1.95, fc="white", ec=SRC["proj"], lw=1.6)
txt(0.80, 3.39, "<project>/", size=11.5, w="bold", c=SRC["proj"], f=MONO)
txt(0.80, 3.11, "the slide drive is never written to", size=8.4, c=MUTE, style="italic")
txt(0.80, 2.81, "project.json", size=8.5, c=SRC["proj"], f=MONO)
txt(1.00, 2.57, "user name  ·  slide paths", size=8, c=MUTE)
txt(0.80, 2.31, "annotations/<slide>.geojson", size=8.5, c=SRC["proj"], f=MONO)
txt(1.00, 2.09, "ROIs, stamped with the author  — from (2)", size=8, c=MUTE)

# ------------------------------------------------------------- panels ------
def panel(x, y, w, h, num, title, colour, uses, steps, out):
    box(x, y, w, h, fc="white", ec=colour, lw=1.7)
    box(x, y + h - 0.52, w, 0.52, fc=colour, ec=colour, r=0.10)
    txt(x + 0.24, y + h - 0.26, num, size=13, w="bold", c="white")
    txt(x + 0.64, y + h - 0.26, title, size=12, w="bold", c="white")

    # chips wrap to a second row rather than running past the panel edge
    cx, cy = x + 0.25, y + h - 0.88
    for name in uses:
        cw = 0.095 * len(name) + 0.30
        if cx + cw > x + w - 0.20:
            cx, cy = x + 0.25, cy - 0.38
        cx += chip(cx, cy, name, ARR[name]) + 0.12

    ty = cy - 0.46
    for s in steps:
        txt(x + 0.28, ty, s, size=9.1, c=INK)
        ty -= 0.315
    txt(x + 0.28, y + 0.30, out, size=9.1, c=colour, w="bold")

panel(5.05, 5.55, 5.70, 3.75, "1", "Draw contours", PANEL["1"],
      ["tissue_mask", "gm_mask", "wm_mask"],
      ["assert  gm | wm == tissue",
       "edge = tissue & ~erode(tissue)",
       "gm_csf = edge & gm            purple",
       "wm_csf = edge & wm            black",
       "gm_wm  = GM rim & ~edge       green",
       "grown to 5 px wide (dilate ×2); cached until masks change"],
      "→  three boundary masks, painted onto the frame")

panel(11.05, 5.55, 5.60, 3.75, "2", "GM annotate", PANEL["2"],
      ["tissue_mask", "wm_mask", "gm_mask"],
      ["cursor confined to gm_mask",
       "one click → shortest CSF↔WM chord by ray cast",
       "step ± half-width, re-cast at each shoulder",
       "snap the 4 corners onto the GM contour",
       "fit a cubic along each boundary arc"],
      "→  4 GeoJSON curves:  CSF · GM · L · R")

panel(5.05, 1.45, 5.70, 3.85, "3", "Feature heatmap", PANEL["3"],
      ["feature_heatmap", "tissue_mask", "thumbnail"],
      ["channel 0 density   /   channel 1 size",
       "min-max window from tissue pixels only",
       "Reds lookup table, alpha 0.5",
       "blended inside the tissue only"],
      "→  tinted frame + histogram")

panel(11.05, 1.45, 5.60, 3.85, "4", "Run NEUSEG + GMM", PANEL["4"],
      ["feature_heatmap", "tissue_mask", "thumbnail", "_log.pkl"],
      ["worker thread, ~20 s, 4 progress stages",
       "run_gmm → GM posterior per tissue pixel",
       "post_process → new gm / wm masks",
       "scatter: density × size, coloured by posterior",
       "hover ⇄ crosshair on the slide"],
      "→  session only; the archive is untouched")

# ------------------------------------------------- render chain footer ------
box(0.35, 0.30, 16.30, 0.80, fc="#f4f4f4", ec="#dddddd")
txt(0.60, 0.70, "RENDER CHAIN", size=9, w="bold", c=MUTE)
txt(0.60, 0.45, "each stage re-runs the ones after it", size=8, c=FAINT, style="italic")
sx = 3.05
for i, (label, colour) in enumerate([("thumbnail", "#444444"), ("+ heatmap (3)", PANEL["3"]),
                                     ("+ contours (1)", PANEL["1"]), ("+ zoom", MUTE),
                                     ("+ ROIs & cursor (2)", PANEL["2"])]):
    txt(sx, 0.70, label, size=9.4, c=colour, w="bold")
    sx += 0.108 * len(label) + 0.30
    if i < 4:
        txt(sx - 0.20, 0.70, "→", size=10, c="#c0c0c0")

fig.savefig(OUT, dpi=190, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
