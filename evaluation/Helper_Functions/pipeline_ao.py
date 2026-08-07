"""
Shared-threshold %AO for manual + NEUSEG ROIs via Noah's CLI pipeline
(pdnl_extract -> pdnl_process -> pdnl_aggregate), run once per slide.

Both sources' ROIs go into ONE geojson with distinct names, so pdnl_extract tiles both,
pdnl_process computes ONE triangular threshold over all chunks, and pdnl_aggregate reports
per-ROI %AO off that shared positive map. Also reports manual-vs-NEUSEG ROI overlap + IoU,
and (verbose) draws the WSI thumbnail with ROI outlines + the used tiles as red boxes.

Manual/NEUSEG geojsons are normalized by a class-based single-ROI parser (single_roi):
classes are bucketed CSF/R/GM/L and the ROI count is decided by the per-class counts, not by
the (unreliable) 'name' property. See single_roi for the 3-way logic + status strings.

The heavy per-slide scratch (chunk PNGs) is written to a temp directory that is deleted before
returning -- only the small result dict is kept (pass work_dir=... to keep it for inspection).
"""

import os
import sys
import json
import glob
import pickle
import shutil
import tempfile
import subprocess

import numpy as np
import pandas as pd
import openslide
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon
from shapely.ops import unary_union

from Helper_Functions.annotation_parsing import single_roi, build_combined_geojson

BIN = os.path.dirname(sys.executable)   # neuseg env bin (holds pdnl_extract / pdnl_process / pdnl_aggregate)


def _roi_polygons(out):
    """Read the ROI polygons pdnl_extract wrote (out/<name>.geojson, class 'ROI'), grouped by source.
    Returns {'manual': [(name, Polygon), ...], 'neuseg': [...]} (buffer(0) repairs self-intersections)."""
    polys = {"manual": [], "neuseg": []}
    for f in glob.glob(os.path.join(out, "*.geojson")):
        if os.path.basename(f) == "combined.geojson": continue
        d = json.load(open(f)); fts = d["features"] if isinstance(d, dict) and "features" in d else d
        for ft in fts:
            if (ft["properties"].get("classification") or {}).get("name") == "ROI":
                full_name = ft["properties"].get("name")
                src = full_name.split("_")[0] if isinstance(full_name, str) else None
                if src in polys:
                    polys[src].append((full_name, Polygon(ft["geometry"]["coordinates"][0]).buffer(0)))
    return polys


def _stitch_roi(out, roi_name, chunk_size):
    """Stitch the chunks overlapping one ROI back into two canvases (over the ROI's tile bbox):
      frame_c : RGB tiles (white where no tile),  pos_c : DAB-positive pixels INSIDE the ROI (binary).
    Returns (frame_c, pos_c, i0, j0) with (i0,j0) the tile-grid origin, or None if the ROI has no tiles."""
    cdir, dab, S = os.path.join(out, "chunks"), os.path.join(out, "dab"), chunk_size
    entries = []
    for cd in os.listdir(cdir):
        mp = os.path.join(cdir, cd, f"mask_{roi_name}.png")
        if not (os.path.isdir(os.path.join(cdir, cd)) and os.path.isfile(mp)):
            continue
        mroi = np.array(Image.open(mp).convert("L")) > 0
        if mroi.any():
            i, j = map(int, cd.split("_"))
            entries.append((i, j, mroi))
    if not entries:
        return None
    i0, i1 = min(e[0] for e in entries), max(e[0] for e in entries)
    j0, j1 = min(e[1] for e in entries), max(e[1] for e in entries)
    frame_c = np.full(((j1 - j0 + 1) * S, (i1 - i0 + 1) * S, 3), 255, np.uint8)
    pos_c = np.zeros(((j1 - j0 + 1) * S, (i1 - i0 + 1) * S), np.uint8)
    for i, j, mroi in entries:
        cd = f"{i}_{j}"
        fr = np.array(Image.open(os.path.join(cdir, cd, "frame.png")).convert("RGB"))
        pos = np.array(Image.open(os.path.join(dab, cd, "pos.png")).convert("L")) > 0
        h = min(fr.shape[0], pos.shape[0], mroi.shape[0]); w = min(fr.shape[1], pos.shape[1], mroi.shape[1])
        y, x = (j - j0) * S, (i - i0) * S
        frame_c[y:y + h, x:x + w] = fr[:h, :w]
        pos_c[y:y + h, x:x + w] = (pos[:h, :w] & mroi[:h, :w]).astype(np.uint8) * 255
    return frame_c, pos_c, i0, j0


def _plot_pipeline(svs_path, out, polys, level, chunk_size, overlap, iou, threshold, slide_name, ao):
    """Col 0: WSI overview (thumbnail + ROIs + red tiles) over DAB histogram; one column per ROI =
    stitched tiles+ROI (top) and DAB-positive pixels inside the ROI (bottom, with its %AO)."""
    srccol = {"manual": "orange", "neuseg": "green"}
    roi_list = [(src, name, poly) for src in ("manual", "neuseg") for (name, poly) in polys[src]]
    slide = openslide.OpenSlide(svs_path)
    W0, H0 = slide.level_dimensions[0]
    dsL = slide.level_downsamples[level]
    S, cdir = chunk_size, os.path.join(out, "chunks")
    ncols = 1 + len(roi_list)
    fig, axes = plt.subplots(2, ncols, figsize=(5.5 * ncols, 9), squeeze=False)

    # (0,0) WSI thumbnail + ROI outlines + used tiles (red boxes)
    thumb = slide.get_thumbnail((1600, max(1, round(1600 * H0 / W0)))); tscale = W0 / thumb.size[0]
    ax = axes[0, 0]; ax.imshow(thumb)
    tiles = [d for d in os.listdir(cdir) if os.path.isdir(os.path.join(cdir, d))]
    for k, cd in enumerate(tiles):
        i, j = map(int, cd.split("_"))
        ax.add_patch(Rectangle((i * S * dsL / tscale, j * S * dsL / tscale), S * dsL / tscale, S * dsL / tscale,
                               ec="red", fc="none", lw=0.8, label="tiles" if k == 0 else None))
    seen = set()
    for src, name, poly in roi_list:
        for g in ([poly] if poly.geom_type == "Polygon" else list(poly.geoms)):
            if g.is_empty: continue
            xy = np.array(g.exterior.coords) / tscale
            ax.plot(xy[:, 0], xy[:, 1], color=srccol[src], lw=1.6, label=src if src not in seen else None)
        seen.add(src)
    iou_str = f"{iou:.3f}" if iou is not None else "n/a"
    ax.set_title(f"{slide_name}\noverlap={overlap}  IoU={iou_str}  tiles={len(tiles)}", fontsize=8)
    ax.axis("off"); ax.legend(loc="lower right", fontsize=7)

    # (1,0) DAB histogram (all tiles) + shared threshold
    ax = axes[1, 0]
    vals = [np.load(f).ravel() for f in glob.glob(os.path.join(out, "dab", "*", "stain.npy"))]
    if vals:
        ax.hist(np.concatenate(vals), bins=64, range=(0, 255), color="gray")
    ax.axvline(threshold, color="red", lw=1.5, label=f"threshold = {threshold}")
    ax.set_yscale("log"); ax.set_xlabel("DAB stain (0-255)"); ax.set_ylabel("pixels (log)")
    ax.set_title("DAB histogram (all tiles)"); ax.legend(fontsize=8)

    # one column per ROI: stitched tiles+ROI (top), DAB-positive inside ROI (bottom)
    ao_pct = dict(zip(ao["ROI"], ao["ao_pct"]))
    for col, (src, name, poly) in enumerate(roi_list, start=1):
        st = _stitch_roi(out, name, chunk_size)
        if st is None:
            axes[0, col].axis("off"); axes[1, col].axis("off"); continue
        frame_c, pos_c, i0, j0 = st
        ax = axes[0, col]; ax.imshow(frame_c)
        for g in ([poly] if poly.geom_type == "Polygon" else list(poly.geoms)):
            if g.is_empty: continue
            xy = np.array(g.exterior.coords) / dsL - np.array([i0 * S, j0 * S])   # level-0 -> tile-canvas
            ax.plot(xy[:, 0], xy[:, 1], color=srccol[src], lw=1.5)
        ax.set_title(f"{name}: tiles + ROI", fontsize=9); ax.axis("off")
        ax = axes[1, col]; ax.imshow(pos_c, cmap="gray")
        ax.set_title(f"DAB-positive in ROI  (%AO={ao_pct.get(name, float('nan')):.2f})", fontsize=9); ax.axis("off")

    plt.tight_layout(); plt.show()
    slide.close()


def evaluate_slide_ao_pipeline(curr_slide_name, eval_cohort_df, level=0, stain="H-DAB",
                               chunk_size=1024, verbose=False, work_dir=None):
    """Shared-threshold %AO for manual + NEUSEG ROIs via pdnl_extract -> process -> aggregate (one pass).
    Returns dict(ao_df, threshold, overlap, iou, pairwise_overlap, status). verbose -> thumbnail + ROIs + used-tiles figure.

    work_dir=None -> scratch (chunk PNGs) goes to an auto-deleted temp dir; only the result dict is kept.
    Pass work_dir=<path> to keep the per-slide scratch under <path>/<slide> for inspection.
    """
    row = eval_cohort_df.set_index("Filename").loc[curr_slide_name]

    tmp = work_dir is None
    out = tempfile.mkdtemp(prefix="pdnl_ao_") if tmp else os.path.join(work_dir, curr_slide_name)
    if not tmp and os.path.exists(out):
        shutil.rmtree(out)
    try:
        # Create a single geojson combining both sources' ROIs, with distinct names
        combo, status = build_combined_geojson(row, os.path.join(out, "combined.geojson"))
        if combo is None:                                # a source was not usable -> skip
            print(f"[pipeline] {curr_slide_name}: skipped -> {status}")
            return dict(ao_df=None, threshold=None, overlap=None, iou=None, status=status)
        if any(v != "ok" for v in status.values()):      # surface non-fatal warnings
            print(f"[pipeline] {curr_slide_name}: {status}")

        def run(cmd):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-1500:], r.stderr[-1500:]); raise RuntimeError(f"{os.path.basename(cmd[0])} failed")

        # [1] pdnl_extract: tile the WSI over the combined ROI (no -c: ROI built from the 4 segments)
        run([f"{BIN}/pdnl_extract", "local", "-s", row["WSI_path"], "-a", combo,
             "-o", out, "--level", str(level), "--chunk_size", str(chunk_size)])
        # [2] pdnl_process: ONE triangular threshold over all chunks (smoothing/normalize_bg = defaults False)
        run([f"{BIN}/pdnl_process", "-i", f"{out}/chunks", "-o", f"{out}/dab", "-s", stain])
        # [3] pdnl_aggregate: per-ROI %AO off the shared positive map
        run([f"{BIN}/pdnl_aggregate", "-i", f"{out}/chunks", "-o", out, "--output_name", "test", "-p", f"{out}/dab"])

        # per-ROI %AO table
        ao = pd.read_csv(os.path.join(out, "ao.csv"))
        ao["ao_pct"] = 100 * ao["AO"]

        # [4] manual-vs-NEUSEG ROI overlap + IoU (pairwise per ROI, not unioned across all ROIs)
        polys = _roi_polygons(out)
        pairwise_overlap = []
        overlap = None
        iou = None
        if polys["manual"] and polys["neuseg"]:
            for manual_name, manual_poly in polys["manual"]:
                for neuseg_name, neuseg_poly in polys["neuseg"]:
                    inter = manual_poly.intersection(neuseg_poly).area
                    uni = manual_poly.union(neuseg_poly).area
                    overlap_bool = bool(manual_poly.intersects(neuseg_poly))
                    pairwise_overlap.append({
                        "manual_roi": manual_name,
                        "neuseg_roi": neuseg_name,
                        "overlap": overlap_bool,
                        "iou": (inter / uni if uni else 0.0),
                        "intersection": inter,
                        "union": uni,
                    })
            overlap = any(p["overlap"] for p in pairwise_overlap)
            iou = max((p["iou"] for p in pairwise_overlap), default=0.0)

        # shared threshold (all chunks share it) -- read from any dab chunk's log
        logs = sorted(glob.glob(os.path.join(out, "dab", "*", "log.pkl")))
        threshold = pickle.load(open(logs[0], "rb")).get("threshold") if logs else None

        # [5] verbose: WSI thumbnail + ROI outlines + used tiles (red boxes)
        if verbose:
            _plot_pipeline(row["WSI_path"], out, polys, level, chunk_size,
                           overlap, iou, threshold, curr_slide_name, ao)

        return dict(ao_df=ao[["Name", "ROI", "Area", "Positive", "AO", "ao_pct"]],
                    threshold=threshold, overlap=overlap, iou=iou,
                    pairwise_overlap=pd.DataFrame(pairwise_overlap) if pairwise_overlap else None,
                    status=status)
    finally:
        if tmp:
            shutil.rmtree(out, ignore_errors=True)
