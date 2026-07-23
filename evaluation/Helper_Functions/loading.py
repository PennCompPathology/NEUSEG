import os
import glob
import json
import numpy as np
import openslide
import pdnl_sana as sana
import pdnl_sana.image
import matplotlib.pyplot as plt


# ================================================ Shared helpers ================================================
def _feature_xy(geo):
    """GeoJSON geometry -> (N,2) float array (matches read_annotations)."""
    t = geo["type"]
    if t == "MultiPolygon":
        x, y = [], []
        for coords in geo["coordinates"]:
            x += [c[0] for c in coords[0]];  y += [c[1] for c in coords[0]]
    elif t == "Polygon":
        x = [c[0] for c in geo["coordinates"][0]];  y = [c[1] for c in geo["coordinates"][0]]
    elif t in ("MultiPoint", "LineString"):
        x = [c[0] for c in geo["coordinates"]];     y = [c[1] for c in geo["coordinates"]]
    else:
        x, y = [], []
    return np.column_stack([x, y]).astype(float) if len(x) else np.empty((0, 2))


def _find_svs(wsi_dir, slide_name):
    """Return the single .svs path for `slide_name` under wsi_dir/<class>/."""
    matches = glob.glob(os.path.join(wsi_dir, "*", slide_name + ".svs"))
    assert len(matches) == 1, f"expected 1 svs for '{slide_name}', found {len(matches)}: {matches}"
    return matches[0]


def _svs_thumbnail(svs_path, downsample=32):
    """Return (RGB thumbnail array, full-res/thumb scale, (W0, H0)) from an SVS."""
    slide = openslide.OpenSlide(svs_path)
    W0, H0 = slide.level_dimensions[0]
    lvl = slide.get_best_level_for_downsample(downsample)
    tw, th = slide.level_dimensions[lvl]
    thumb = np.asarray(slide.read_region((0, 0), lvl, (tw, th)).convert("RGB"))
    slide.close()
    return thumb, W0 / tw, (W0, H0)


def _parse_boundaries(feats, class_map, transform=None, default_roi=""):
    """
    Group geojson features into {roi_name: {key: (N,2) array}}.

    class_map : geojson classification name -> output key
    transform : optional fn applied to each (N,2) array (e.g. thumbnail->full-res)
    Returns (annot, skipped) where skipped lists unmapped/duplicate features.
    """
    annot, skipped = {}, []
    for ft in feats:
        props = ft.get("properties", {})
        cname = props.get("classification", {}).get("name", "")
        if cname not in class_map:
            skipped.append(cname or "<blank>")
            continue
        roi = props.get("name", "") or default_roi
        key = class_map[cname]
        xy = _feature_xy(ft["geometry"])
        if transform is not None:
            xy = transform(xy)
        roi_d = annot.setdefault(roi, {})
        if key in roi_d:                         # duplicate class within same ROI
            skipped.append(f"{roi}:{cname}(dup)")
            continue
        roi_d[key] = xy
    return annot, skipped


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
REQUIRED_FILES = {"annotations.geojson", "gm_mask.npy", "wm_mask.npy",
                  "tissue_mask.npy", "thumbnail.png"}

FILE_NAMES = {
    "Thumbnail":   "thumbnail.png",
    "GM Mask":     "gm_mask.npy",
    "WM Mask":     "wm_mask.npy",
    "Tissue Mask": "tissue_mask.npy",
    "Annotation":  "annotations.geojson",
}

NEUSEG_CLASS_MAP = {"GM": "gm_wm", "CSF": "gm_csf", "L": "left", "R": "right"}


def find_neuseg_dir(neuseg_annot_dir, slide_name, required=REQUIRED_FILES):
    """
    Walk neuseg_annot_dir and return the deepest subdirectory that both
    (a) contains all `required` files, and (b) is tied to `slide_name`
    (its basename == slide_name, or slide_name appears in the path).
    Returns the absolute path, or raises FileNotFoundError.
    """
    matches = []
    for dirpath, _, filenames in os.walk(neuseg_annot_dir):
        if not required.issubset(filenames):
            continue
        if os.path.basename(dirpath) == slide_name or slide_name in dirpath:
            matches.append(dirpath)

    if not matches:
        raise FileNotFoundError(
            f"No dir under {neuseg_annot_dir} has all {required} for '{slide_name}'")

    # prefer an exact basename match; then take the deepest path
    exact = [m for m in matches if os.path.basename(m) == slide_name]
    pool = exact or matches
    return max(pool, key=lambda p: p.count(os.sep))


def get_neuseg_results(curr_slide_name, neuseg_annot_dir, wsi_dir,
                       file_names=FILE_NAMES, verbose=False):
    """
    Get NEUSEG GM-WM / GM-CSF contours & semi-auto annotation for one slide,
    in FULL-RES pixel coords.

    Parameters
    ----------
    curr_slide_name : str   svs filename without extension
    neuseg_annot_dir : str  root dir of NEUSEG results (searched recursively)
    wsi_dir : str           root dir holding the .svs files (in class subfolders)
    file_names : dict       mask/thumbnail/annotation filenames
    verbose : bool          print diagnostics and show an overlay figure
                            (thumbnail + mask contours + NEUSEG ROI annotation)

    Returns
    -------
    dict with keys:
      slide_name, neuseg_dir, svs_path,
      full_res (W0,H0), thumb (Wt,Ht),
      mpp, scale (full-res / thumbnail; masks share the thumbnail resolution),
      mask_gm_wm  : list of (N,2) full-res arrays (WM outline)
      mask_gm_csf : list of (M,2) full-res arrays (tissue outline)
      annot       : {roi_name: {'gm_wm', 'gm_csf', 'left', 'right'}} (N,2) full-res
    """
    # [0] locate NEUSEG dir + matching full-res SVS
    neuseg_dir = find_neuseg_dir(neuseg_annot_dir, curr_slide_name)
    svs_path = _find_svs(wsi_dir, curr_slide_name)

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

    def to_fullres(xy):                       # thumbnail/mask-space -> full-res
        return np.asarray(xy, dtype=float) * scale

    # [2] mask-derived contours (masks are at the thumbnail resolution)
    wm_arr     = np.load(os.path.join(neuseg_dir, file_names["WM Mask"])).astype(np.uint8)
    tissue_arr = np.load(os.path.join(neuseg_dir, file_names["Tissue Mask"])).astype(np.uint8)

    Hm, Wm = wm_arr.shape[:2]
    assert (Wm, Hm) == (Wt, Ht), \
        f"mask {Wm}x{Hm} != thumbnail {Wt}x{Ht}; masks assumed at thumbnail resolution"

    wm_polys,     _ = sana.image.Frame(wm_arr).to_polygons()       # GM-WM boundary
    tissue_polys, _ = sana.image.Frame(tissue_arr).to_polygons()   # GM-CSF boundary
    mask_gm_wm  = [to_fullres(p) for p in wm_polys]
    mask_gm_csf = [to_fullres(p) for p in tissue_polys]
    if verbose:
        print(f"mask contours -> GM-WM: {len(mask_gm_wm)} polys | GM-CSF: {len(mask_gm_csf)} polys")

    # [3] saved annotations.geojson -> full-res
    with open(os.path.join(neuseg_dir, file_names["Annotation"])) as f:
        feats = json.load(f)
    if isinstance(feats, dict) and "features" in feats:   # FeatureCollection safety
        feats = feats["features"]
    annot, skipped = _parse_boundaries(feats, NEUSEG_CLASS_MAP, transform=to_fullres)

    if verbose:
        for roi, c in annot.items():
            print(f"geojson ROI '{roi}':",
                  "GM-WM", None if "gm_wm"  not in c else c["gm_wm"].shape,
                  "| GM-CSF", None if "gm_csf" not in c else c["gm_csf"].shape)
        if skipped:
            print(f"[neuseg] skipped/unmapped features: {skipped}")

        # overlay: thumbnail (SVS) + mask contours + NEUSEG ROI annotation
        walls = ([c["left"]  for c in annot.values() if "left"  in c] +
                 [c["right"] for c in annot.values() if "right" in c], False,
                 dict(color="gray", lw=2.0, ls=":", label="ROI L/R walls"))
        layers = [
            (mask_gm_wm,  True, dict(color="green",  lw=3.0, label="mask GM-WM")),
            (mask_gm_csf, True, dict(color="purple", lw=3.0, label="mask GM-CSF")),
            _roi_layer(annot, "gm_wm",  color="orange", lw=3.0, label="NEUSEG ROI GM-WM"),
            _roi_layer(annot, "gm_csf", color="cyan",   lw=3.0, label="NEUSEG ROI GM-CSF"),
            walls,
        ]
        _show_overlay(slide_tb.img, scale, layers, curr_slide_name)

    return {
        "slide_name": curr_slide_name,
        "neuseg_dir": neuseg_dir,
        "svs_path":   svs_path,
        "full_res":   (W0, H0),
        "thumb":      (Wt, Ht),
        "mpp":        mpp,
        "scale":      scale,
        "mask_gm_wm":  mask_gm_wm,
        "mask_gm_csf": mask_gm_csf,
        "annot":       annot,
    }


# ================================================ Manual Annotation Related ================================================
# manual QuPath class names -> our keys (note: differ from NEUSEG's CSF/GM)
MANUAL_CLASS_MAP = {"GM_WM": "gm_wm", "CSF_GM": "gm_csf", "L": "left", "R": "right"}


def find_manual_geojson(manual_annot_dir, slide_name):
    """
    Walk manual_annot_dir and return the .geojson file for `slide_name`
    (one geojson per slide). A file matches when its basename (without the
    .geojson extension) equals `slide_name`, or contains it. Returns the
    absolute path, or raises FileNotFoundError.
    """
    matches = []
    for dirpath, _, filenames in os.walk(manual_annot_dir):
        for fn in filenames:
            if not fn.lower().endswith(".geojson"):
                continue
            stem = os.path.splitext(fn)[0]
            if stem == slide_name or slide_name in stem:
                matches.append(os.path.join(dirpath, fn))

    if not matches:
        raise FileNotFoundError(
            f"No .geojson under {manual_annot_dir} matches slide '{slide_name}'")

    # prefer an exact stem match; else the shortest basename (closest match)
    exact = [m for m in matches if os.path.splitext(os.path.basename(m))[0] == slide_name]
    pool = exact or matches
    return min(pool, key=lambda p: len(os.path.basename(p)))


def get_manual_results(curr_slide_name, manual_annot_dir, wsi_dir=None,
                       class_map=MANUAL_CLASS_MAP, verbose=False):
    """
    Load manual QuPath ROI boundaries for one slide, in FULL-RES pixel coords.
    Manual annotations are already full resolution, so NO scaling is applied.

    Manual geojson classes: 'CSF_GM' (GM-CSF boundary), 'GM_WM' (GM-WM boundary),
    'L'/'R' (side walls). Boundaries are grouped by their ROI name.

    Parameters
    ----------
    curr_slide_name : str   svs filename without extension
    manual_annot_dir : str  root dir of manual annotations (searched recursively)
    wsi_dir : str or None   root dir holding the .svs files; required only to draw
                            the verbose overlay figure (manual has no thumbnail.png)
    class_map : dict        QuPath class name -> output key
    verbose : bool          print per-ROI diagnostics, flag empty/incomplete files,
                            and (if wsi_dir given) show an overlay figure

    Returns
    -------
    dict with keys:
      slide_name, geojson_path,
      annot : {roi_name: {'gm_wm', 'gm_csf', 'left', 'right'}} (N,2) full-res arrays
    """
    geojson_path = find_manual_geojson(manual_annot_dir, curr_slide_name)
    with open(geojson_path) as f:
        feats = json.load(f)
    if isinstance(feats, dict) and "features" in feats:   # FeatureCollection
        feats = feats["features"]
    annot, skipped = _parse_boundaries(feats, class_map, default_roi="ROI")  # no scaling

    if verbose:
        if len(feats) == 0:
            print(f"[manual] WARNING: '{curr_slide_name}' has 0 features (empty annotation)")
        for roi, c in annot.items():
            missing = [k for k in ("gm_wm", "gm_csf", "left", "right") if k not in c]
            print(f"manual ROI '{roi}':",
                  "GM-WM", None if "gm_wm" not in c else c["gm_wm"].shape,
                  "| GM-CSF", None if "gm_csf" not in c else c["gm_csf"].shape,
                  ("| missing: " + ",".join(missing)) if missing else "")
        if skipped:
            print(f"[manual] skipped/unmapped features: {skipped}")

        # overlay: SVS thumbnail + manual ROI annotation (manual coords are full-res)
        if wsi_dir is not None and annot:
            thumb, disp_scale, _ = _svs_thumbnail(_find_svs(wsi_dir, curr_slide_name))
            walls = ([c["left"]  for c in annot.values() if "left"  in c] +
                     [c["right"] for c in annot.values() if "right" in c], False,
                     dict(color="gray", lw=2.0, ls=":", label="manual L/R walls"))
            layers = [
                _roi_layer(annot, "gm_wm",  color="orange", lw=3.0, label="manual GM-WM"),
                _roi_layer(annot, "gm_csf", color="cyan",   lw=3.0, label="manual GM-CSF"),
                walls,
            ]
            _show_overlay(thumb, disp_scale, layers, f"{curr_slide_name} (manual)")

    return {
        "slide_name":   curr_slide_name,
        "geojson_path": geojson_path,
        "annot":        annot,
    }
