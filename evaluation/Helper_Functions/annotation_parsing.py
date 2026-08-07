"""
Class-based geojson ROI parsing, shared by the %AO pipeline and the annotation-to-contour analysis.

single_roi   : bucket a geojson's boundary edges by class (CSF/R/GM/L, after an optional remap) and
               decide the ROI count from the per-class counts -- NOT the unreliable 'name' property.
               Returns (rois, status): each ROI is a list of 4 edge-features named f'{source}_{i}';
               status is 'ok' | 'warn: ...' (rois still returned) | 'error: ...' (rois is None).
parse_rois   : single_roi reshaped to {roi_name: {'gm_wm','gm_csf','left','right': (N,2) array}} --
               the shape get_neuseg_results / get_manual_results / annotation_to_contour_distances use.
build_combined_geojson : both sources' ROIs -> ONE level-0 geojson for pdnl_extract (%AO pipeline).
"""

import os
import json

import numpy as np
import openslide

# edge class (CSF/R/GM/L, after remap) -> boundary key used by the distance analysis
_CLASS_TO_KEY = {"CSF": "gm_csf", "R": "right", "GM": "gm_wm", "L": "left"}
# manual QuPath class names -> the CSF/R/GM/L buckets (NEUSEG classes are already CSF/R/GM/L)
_MANUAL_REMAP = {"CSF_GM": "CSF", "GM_WM": "GM"}


def single_roi(features, source, remap=None, scale=None):
    """Bucket edges by (remapped) class CSF/R/GM/L and decide by count A=CSF, B=GM, C=L, D=R.
    Returns (rois, status):
      rois   : list of ROIs (each = list of 4 edge-features named f'{source}_{i}'), or None on error.
      status : 'ok' | 'warn: ...' (rois still returned) | 'error: ...' (rois is None).
    3-way logic:
      A=B=C=D=1   -> ONE ROI. If the 4 edges share a name -> 'ok'; else still 1 ROI, status 'warn: ...'
                     (recovers slides whose single ROI was split across names).
      A=B=C=D=k>1 -> group the k*4 edges by name; exactly k complete {CSF,R,GM,L} groups -> k ROIs 'ok';
                     otherwise rois=None, status 'error: ...' (names don't partition into k ROIs).
      else        -> rois=None, status 'error: class counts disagree ...'.
    """
    order = ("CSF", "R", "GM", "L")
    buckets = {c: [] for c in order}
    for ft in features:
        cl = (ft.get("properties") or {}).get("classification")
        nm = cl.get("name") if cl else None
        if remap: nm = remap.get(nm, nm)                # CSF_GM->CSF, GM_WM->GM (manual); L/R unchanged
        if nm in buckets: buckets[nm].append(ft)

    counts = {c: len(buckets[c]) for c in order}
    if len(set(counts.values())) != 1:                  # A,B,C,D not all equal -> malformed
        return None, (f"error: class counts disagree "
                      f"(CSF={counts['CSF']}, GM={counts['GM']}, L={counts['L']}, R={counts['R']})")
    k = counts["CSF"]
    if k == 0:
        return None, "error: no CSF/R/GM/L edges found"

    def emit(pick, i):                                  # normalize class, scale coords, name f'{source}_{i}'
        edges = []
        for c in order:
            ft = pick[c]
            if scale is not None: ft["geometry"]["coordinates"] = scale(ft["geometry"]["coordinates"])
            ft.setdefault("properties", {}).setdefault("classification", {})["name"] = c
            ft["properties"]["name"] = f"{source}_{i}"
            edges.append(ft)
        return edges

    if k == 1:                                          # case 1: single ROI (names checked, not required)
        names = {(buckets[c][0].get("properties") or {}).get("name") for c in order}
        roi = emit({c: buckets[c][0] for c in order}, 0)
        if len(names) == 1:
            return [roi], "ok"
        return [roi], f"warn: single ROI, edges span names {sorted(map(str, names))}"

    # case 2: k>1 -> pair the k*4 edges by their original name
    groups = {}
    for c in order:
        for ft in buckets[c]:
            onm = (ft.get("properties") or {}).get("name")
            g = groups.setdefault(onm, {})
            if c in g:
                return None, f"error: k={k} but class {c} repeats under name {onm!r} (cannot pair)"
            g[c] = ft
    if len(groups) != k or any(set(g) != set(order) for g in groups.values()):
        summary = ", ".join(f"{n!r}:{sorted(g)}" for n, g in groups.items())
        return None, f"error: k={k} but names do not form {k} complete ROIs: {{{summary}}}"
    return [emit(groups[onm], i) for i, onm in enumerate(groups)], "ok"


def parse_rois(features, source, scale=1.0):
    """single_roi -> {roi_name: {'gm_wm','gm_csf','left','right': (N,2) float array}}, plus status.
    source='manual' applies the CSF_GM/GM_WM class remap; `scale` (thumbnail->level-0 factor, used by
    NEUSEG) multiplies every coordinate. Returns ({}, status) when single_roi errors (status says why).
    Drop-in replacement for the old _parse_boundaries (name-based) parser."""
    remap = _MANUAL_REMAP if source == "manual" else None
    scale_fn = None
    if scale != 1.0:
        s = scale
        def scale_fn(c):
            return [scale_fn(x) for x in c] if isinstance(c[0], (list, tuple)) else [c[0] * s, c[1] * s]

    rois, status = single_roi(features, source, remap=remap, scale=scale_fn)

    if rois is None:
        return {}, status
    
    annot = {
        roi[0]["properties"]["name"]: {
            _CLASS_TO_KEY[ft["properties"]["classification"]["name"]]:
                np.asarray(ft["geometry"]["coordinates"], float).reshape(-1, 2)
            for ft in roi
        }
        for roi in rois
    }
    return annot, status


def build_combined_geojson(row, dst):
    """ONE geojson holding both sources' ROIs at level-0 (classes CSF/R/GM/L, names 'manual_i'/'neuseg_i').
    manual: rename CSF_GM->CSF, GM_WM->GM (coords already level-0).
    neuseg: scale thumbnail coords -> level-0 by W0/Wt (classes already CSF/R/GM/L).
    Returns (path, status_dict) on success, or (None, status_dict) if a source is unusable.
    status_dict = {'manual': <status>, 'neuseg': <status>}."""

    # helper to get the features list from a geojson dict (or return the input if not a dict)
    def feats(d): return d["features"] if isinstance(d, dict) and "features" in d else d

    # manual ROI(s) -> name='manual_i', renamed classes (class-based parse)
    m, m_status = single_roi(feats(json.load(open(row["manual_path"]))),  # Load the manual ROI geojson
                             "manual", remap=_MANUAL_REMAP)

    # neuseg ROI(s) -> name='neuseg_i', coords scaled thumbnail -> level 0
    W0 = openslide.OpenSlide(row["WSI_path"]).level_dimensions[0][0]
    Wt = np.load(os.path.join(row["neuseg_path"], "wm_mask.npy")).shape[1]
    s = W0 / Wt   # scale factor from thumbnail to level-0
    scale = lambda c: [scale(x) for x in c] if isinstance(c[0], (list, tuple)) else [c[0]*s, c[1]*s]
    n, n_status = single_roi(feats(json.load(open(os.path.join(row["neuseg_path"], "annotations.geojson")))),  # Load the NEUSEG ROI geojson
                             "neuseg", scale=scale)

    status = {"manual": m_status, "neuseg": n_status}
    if m is None or n is None:
        return None, status

    # Save the combined geojson (flatten every ROI's 4 edges into one feature list)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    features = [ft for roi in m for ft in roi] + [ft for roi in n for ft in roi]
    json.dump({"type": "FeatureCollection", "features": features}, open(dst, "w"))
    return dst, status
