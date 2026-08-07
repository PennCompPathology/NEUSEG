import os
import json
import numpy as np
import openslide
import pdnl_sana as sana

from Helper_Functions.annotation_parsing import parse_rois
import pdnl_sana.image
import matplotlib.pyplot as plt


# ================================================ Shared helpers ================================================
def _show_overlay(thumb_img, scale, layers, title, figsize_w=12):
    """
    Overlay boundary layers on a thumbnail image.

    thumb_img : HxWx3 array (or PIL image) to draw on
    scale     : full-res -> thumbnail divisor applied to every contour
    layers    : list of (polys, close, kwargs); polys are full-res (N,2) arrays,
                `close` connects last->first, kwargs passed to ax.plot (may hold 'label')
    """
    img = np.asarray(thumb_img)
    h, w = img.shape[:2]
    _, ax = plt.subplots(figsize=(figsize_w, figsize_w * h / w))
    ax.imshow(img)
    for polys, close, kw in layers:
        kw = dict(kw)
        label = kw.pop("label", None)
        first = True
        for p in polys:
            p = np.asarray(p, dtype=float) / scale
            if len(p) == 0:
                continue
            if close:
                p = np.vstack([p, p[:1]])
            ax.plot(p[:, 0], p[:, 1], label=label if first else None, **kw)
            first = False
    ax.set_title(title)
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.show()


def _roi_layer(annot, key, **kw):
    """Collect one boundary type across all ROIs into a plot layer (open curves)."""
    return ([c[key] for c in annot.values() if key in c], False, kw)


# ================================================ NEUSEG Related ================================================
NEUSEG_REQUIRED_FILES = {"annotations.geojson", "gm_mask.npy", "wm_mask.npy",
                  "tissue_mask.npy", "thumbnail.png"}

NEUSEG_FILE_NAMES = {
    "Thumbnail":   "thumbnail.png",
    "GM Mask":     "gm_mask.npy",
    "WM Mask":     "wm_mask.npy",
    "Tissue Mask": "tissue_mask.npy",
    "Annotation":  "annotations.geojson",
}



def get_neuseg_results(curr_slide_name, neuseg_dir, svs_path,
                       file_names=NEUSEG_FILE_NAMES, verbose=False):
    """
    Get NEUSEG GM-WM / GM-CSF contours & semi-auto annotation for one slide,
    in FULL-RES pixel coords.

    Parameters
    ----------
    curr_slide_name : str   svs filename without extension (used for labels only)
    neuseg_dir : str        NEUSEG result dir for this slide (holds masks + geojson)
    svs_path : str          full-res .svs path for this slide
    file_names : dict       mask/thumbnail/annotation filenames
    verbose : bool          print diagnostics and show an overlay figure
                            (thumbnail + mask contours + NEUSEG ROI annotation)

    Returns
    -------
    dict with keys:
      slide_name, neuseg_dir, svs_path,
      full_res (W0,H0), thumb (Wt,Ht),
      mpp, scale (full-res / thumbnail; masks share the thumbnail resolution),
      contour_gm_wm  : list of (N,2) full-res arrays (WM outline)
      contour_gm_csf : list of (M,2) full-res arrays (tissue outline)
      annot       : {roi_name: {'gm_wm', 'gm_csf', 'left', 'right'}} (N,2) full-res
    """

    # [1] thumbnail -> full-res scale (from the SVS pyramid)
    #     masks and thumbnail share one resolution, so a single scale covers both
    slide = openslide.OpenSlide(svs_path)
    W0, H0 = slide.level_dimensions[0]
    mpp = float(slide.properties.get('aperio.MPP',
                slide.properties.get('openslide.mpp-x', 'nan')))
    slide.close()

    slide_tb = sana.image.Frame(os.path.join(neuseg_dir, file_names["Thumbnail"]))
    Wt, Ht = slide_tb.size()
    sx, sy = W0 / Wt, H0 / Ht
    assert abs(sx - sy) / sx < 0.02, f"non-uniform scale sx={sx:.3f} sy={sy:.3f}"
    scale = sx

    if verbose:
        print(f"full-res={W0}x{H0}  thumb={Wt}x{Ht}  scale={scale:.4f}  mpp={mpp:.4f}")

    # [2] mask-derived contours (masks are at the thumbnail resolution) --> full-reso
    def to_fullres(xy):                       # thumbnail/mask-space -> full-res
            return np.asarray(xy, dtype=float) * scale
    
    wm_arr     = np.load(os.path.join(neuseg_dir, file_names["WM Mask"])).astype(np.uint8)
    tissue_arr = np.load(os.path.join(neuseg_dir, file_names["Tissue Mask"])).astype(np.uint8)

    Hm, Wm = wm_arr.shape[:2]
    assert (Wm, Hm) == (Wt, Ht), \
        f"mask {Wm}x{Hm} != thumbnail {Wt}x{Ht}; masks assumed at thumbnail resolution"

    wm_polys,     _ = sana.image.Frame(wm_arr).to_polygons()       # GM-WM boundary
    tissue_polys, _ = sana.image.Frame(tissue_arr).to_polygons()   # GM-CSF boundary

    contour_gm_wm  = [to_fullres(p) for p in wm_polys]                 # Upsample to full-reso
    contour_gm_csf = [to_fullres(p) for p in tissue_polys]             # Upsample to full-reso

    if verbose:
        print(f"mask contours -> GM-WM: {len(contour_gm_wm)} polys | GM-CSF: {len(contour_gm_csf)} polys")

    # [3] saved annotations.geojson -> full-res
    with open(os.path.join(neuseg_dir, file_names["Annotation"])) as f:
        feats = json.load(f)

    if isinstance(feats, dict) and "features" in feats:   # FeatureCollection safety
        feats = feats["features"]

    # Parse the ROIs
    annot, status = parse_rois(feats, "neuseg", scale=scale)

    if verbose:
        for roi, c in annot.items():
            print(f"geojson ROI '{roi}':",
                  "GM-WM", None if "gm_wm"  not in c else c["gm_wm"].shape,
                  "| GM-CSF", None if "gm_csf" not in c else c["gm_csf"].shape)
        if status != "ok":
            print(f"[neuseg] {status}")

        # overlay: thumbnail (SVS) + mask contours + NEUSEG ROI annotation
        walls = ([c["left"]  for c in annot.values() if "left"  in c] +
                 [c["right"] for c in annot.values() if "right" in c], False,
                 dict(color="gray", lw=2.0, ls=":", label="ROI L/R walls"))
        layers = [
            (contour_gm_wm,  True, dict(color="green",  lw=2.0, label="mask GM-WM")),
            (contour_gm_csf, True, dict(color="purple", lw=2.0, label="mask GM-CSF")),
            _roi_layer(annot, "gm_wm",  color="orange", lw=3.0, label="NEUSEG ROI GM-WM"),
            _roi_layer(annot, "gm_csf", color="cyan",   lw=3.0, label="NEUSEG ROI GM-CSF"),
            walls,
        ]
        _show_overlay(slide_tb.img, scale, layers, f"[NEUSEG Semi-Auto ROI Annotation] {curr_slide_name}")

    return {
        "slide_name": curr_slide_name,
        "neuseg_dir": neuseg_dir,
        "svs_path":   svs_path,
        "full_res":   (W0, H0),
        "thumb":      (Wt, Ht),
        "mpp":        mpp,
        "scale":      scale,
        "contour_gm_wm":  contour_gm_wm,
        "contour_gm_csf": contour_gm_csf,
        "annot":       annot,
    }


# ================================================ Manual Annotation Related ================================================


def _svs_thumbnail(svs_path, downsample=32):
    """Return (RGB thumbnail array, full-res/thumb scale, (W0, H0)) from an SVS."""
    slide = openslide.OpenSlide(svs_path)
    W0, H0 = slide.level_dimensions[0]
    lvl = slide.get_best_level_for_downsample(downsample)
    tw, th = slide.level_dimensions[lvl]
    thumb = np.asarray(slide.read_region((0, 0), lvl, (tw, th)).convert("RGB"))
    slide.close()
    return thumb, W0 / tw, (W0, H0)


def get_manual_results(curr_slide_name, geojson_path, neuseg_res=None, verbose=False):
    """
    Load manual QuPath ROI boundaries for one slide, in FULL-RES pixel coords.
    Manual annotations are already full resolution, so NO scaling is applied.

    Manual geojson classes: 'CSF_GM' (GM-CSF), 'GM_WM' (GM-WM), 'L'/'R' (walls),
    grouped by ROI name.

    Parameters
    ----------
    curr_slide_name : str   svs filename without extension (labels only)
    geojson_path : str      manual .geojson path for this slide
    neuseg_res : dict or None  get_neuseg_results output for the SAME slide; if given
                            (and verbose), its SVS thumbnail + NEUSEG contours are
                            drawn under the manual annotation (no recompute)
    verbose : bool          print per-ROI diagnostics, and (if neuseg_res given)
                            show the manual-vs-NEUSEG overlay figure

    Returns
    -------
    dict: slide_name, geojson_path, annot {roi: {'gm_wm','gm_csf','left','right'}}
    """
    with open(geojson_path) as f:
        feats = json.load(f)
    if isinstance(feats, dict) and "features" in feats:   # FeatureCollection
        feats = feats["features"]
    annot, status = parse_rois(feats, "manual")   # class-based parse (manual remap internal)

    if verbose:
        if len(feats) == 0:
            print(f"[manual] WARNING: '{curr_slide_name}' has 0 features (empty annotation)")
        for roi, c in annot.items():
            missing = [k for k in ("gm_wm", "gm_csf", "left", "right") if k not in c]
            print(f"manual ROI '{roi}':",
                  "GM-WM", None if "gm_wm" not in c else c["gm_wm"].shape,
                  "| GM-CSF", None if "gm_csf" not in c else c["gm_csf"].shape,
                  ("| missing: " + ",".join(missing)) if missing else "")
        if status != "ok":
            print(f"[manual] {status}")

        # overlay: SVS thumbnail + NEUSEG contours (green/purple) + manual annot
        # (orange/cyan). Reuses neuseg_res -- nothing recomputed.
        if neuseg_res is not None and annot:
            thumb, disp_scale, _ = _svs_thumbnail(neuseg_res["svs_path"])
            walls = ([c["left"]  for c in annot.values() if "left"  in c] +
                     [c["right"] for c in annot.values() if "right" in c], False,
                     dict(color="gray", lw=2.0, ls=":", label="manual L/R walls"))
            _show_overlay(thumb, disp_scale, [
                (neuseg_res["contour_gm_wm"],  True, dict(color="green",  lw=2.0, label="NEUSEG contour GM-WM")),
                (neuseg_res["contour_gm_csf"], True, dict(color="purple", lw=2.0, label="NEUSEG contour GM-CSF")),
                _roi_layer(annot, "gm_wm",  color="orange", lw=3.0, label="manual GM-WM"),
                _roi_layer(annot, "gm_csf", color="cyan",   lw=3.0, label="manual GM-CSF"),
                walls,
            ], f"[Manual ROI Annotation] {curr_slide_name}")

    return {
        "slide_name":   curr_slide_name,
        "geojson_path": geojson_path,
        "annot":        annot,
    }
