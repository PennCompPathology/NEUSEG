"""
Annotation-to-contour distance metric (Option 2: shapely, continuous distance).

Given an annotation (a set of ROI boundary curves) and a set of segmentation
*contours* (mask outlines as polygons), this measures, for each ROI boundary,
how far the annotation curve sits from the matching contour, in microns.

Method (per ROI, per boundary type, e.g. 'gm_wm', 'gm_csf'):
  1. Sample the annotation curve UNIFORMLY every `step_um` microns along its
     length (arc-length sampling). This normalises uneven vertex spacing, so a
     coarse ~20-vertex curve still yields a dense, evenly spaced set of query
     points to measure from.
  2. For each sample point, compute the EXACT perpendicular distance to the
     contour, treating the contour as a continuous poly-LINE (shapely). Unlike a
     KD-tree over the contour's vertices, this measures to the nearest location
     anywhere ALONG the contour edges -- so it does not depend on how densely the
     contour happens to be vertex-sampled (no quantisation error).
  3. Convert pixel distances -> microns (x mpp) and summarise (mean, etc.).

The function is deliberately generic: it takes any `annot` dict of the shape
    {roi_name: {'gm_wm': (N,2) array, 'gm_csf': (M,2) array, ...}}
so the SAME call works for NEUSEG's semi-auto annotation and for the manual
annotation -- both get_neuseg_results(...)['annot'] and
get_manual_results(...)['annot'] already return this shape.

All coordinates are full-resolution (level-0) pixels; `mpp` is level-0 microns
per pixel. Requires shapely >= 2.0 (for the vectorised `shapely.*` functions).
"""

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection



# ============================================================================
# Contour packaging: list of polygon-vertex arrays -> one shapely geometry
# ============================================================================
def _contour_to_geometry(polys):
    """
    Turn a contour (a list of (K,2) polygon-vertex arrays, full-res pixels) into
    a single shapely geometry we can measure distances to.

    Each polygon is the OUTLINE of a mask region, so we represent it as a
    LineString of its vertices -- NOT a filled Polygon. We want the distance to
    the boundary curve; a filled Polygon would report distance 0 for any point
    sitting *inside* the region, which is not what we mean by "distance to the
    contour".

    We also close each ring (append the first vertex if the polygon is not
    already closed) so the outline includes the final edge back to the start.

    Returns a single shapely geometry (the union of all outline pieces), or None
    if there is nothing usable.
    """
    if not polys:                                 # None or empty list
        return None

    lines = []
    for p in polys:
        p = np.asarray(p, dtype=float)
        if p.shape[0] < 2:                        # need >= 2 points for a segment
            continue
        if not np.array_equal(p[0], p[-1]):       # close the ring if open
            p = np.vstack([p, p[:1]])
        lines.append(LineString(p))

    if not lines:
        return None

    # union_all merges the separate outline pieces into ONE geometry, so a single
    # distance query considers all of them at once (e.g. a region made of several
    # disconnected islands, or a ring with holes).
    return shapely.union_all(lines)


# ============================================================================
# Uniform arc-length sampling of an annotation curve
# ============================================================================
def _sample_polyline(xy, step_px):
    """
    Sample points every `step_px` pixels ALONG a polyline (arc-length spacing).

    `xy`      : (N,2) array of full-res vertices for one annotation boundary.
    `step_px` : spacing between samples, in pixels (= 1 micron expressed in px).

    Returns an (M,2) array of sampled (x, y) points, or an empty (0,2) array for
    a degenerate curve (fewer than 2 vertices, or zero total length).

    Sampling is by distance ALONG the connected segments, so the result does not
    depend on where the original vertices fall: a straight 2-vertex segment and a
    wiggly 20-vertex curve are both sampled uniformly at `step_px` spacing.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.shape[0] < 2:
        return np.empty((0, 2), dtype=float)

    line = LineString(xy)
    if line.length == 0:                          # all vertices coincide
        return np.empty((0, 2), dtype=float)

    # n_steps segments -> n_steps + 1 sample points, including both endpoints.
    # ceil() guarantees the spacing is <= step_px (never coarser than 1 micron).
    n_steps = max(1, int(np.ceil(line.length / step_px)))
    dists = np.linspace(0.0, line.length, n_steps + 1)

    # Vectorised interpolation along the line (shapely 2.0): distance -> Point.
    pts = shapely.line_interpolate_point(line, dists)
    return shapely.get_coordinates(pts)


# ============================================================================
# Verbose validation plot (one close-up per ROI x boundary)
# ============================================================================
# Colours match the loading.py verbose overlays so the two views stay consistent:
#   gm_wm  -> contour green,  annotation orange
#   gm_csf -> contour purple, annotation cyan
_VERBOSE_COLORS = {
    "gm_wm":  {"contour": "green",  "annot": "orange"},
    "gm_csf": {"contour": "purple", "annot": "cyan"},
}


def _plot_distance_validation(roi_name, key, curve, samples, geom,
                              contour_polys, step_um, mean_um):
    """
    Close-up figure to visually validate the distance calculation for one
    (ROI, boundary). It draws, in the same colours as the other verbose overlays:
      - the contour outline(s)                 (green / purple)
      - the annotation curve                   (orange / cyan)
      - the 1-micron arc-length sample points  (dots; the spacing is in the legend)
      - a hairline from EACH sample to its nearest point on the contour
        (these hairlines ARE the distances being averaged -- their individual
         lengths are not labelled, per request)
    """
    colors = _VERBOSE_COLORS.get(key, {"contour": "green", "annot": "orange"})

    # For each sample, shapely.shortest_line gives the 2-point segment from the
    # sample to the nearest location on the contour -- i.e. exactly the distance
    # we measured. get_coordinates stacks them as (2*M, 2); reshape -> (M, 2, 2).
    nn_lines = shapely.shortest_line(shapely.points(samples), geom)
    nn_segs = shapely.get_coordinates(nn_lines).reshape(-1, 2, 2)

    curve = np.asarray(curve, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 8))

    # (1) nearest-distance hairlines first, so the curves draw on top of them.
    ax.add_collection(LineCollection(nn_segs, colors="0.35", linewidths=0.5,
                                     alpha=0.6, label="nearest distance"))

    # (2) contour outline(s) in the contour colour (label only the first piece).
    first = True
    for poly in (contour_polys or []):
        poly = np.asarray(poly, dtype=float)
        if poly.shape[0] < 2:
            continue
        ax.plot(poly[:, 0], poly[:, 1], color=colors["contour"], lw=2.0,
                label="contour" if first else None)
        first = False

    # (3) annotation curve in the annotation colour.
    ax.plot(curve[:, 0], curve[:, 1], color=colors["annot"], lw=1.5,
            label="annotation")

    # (4) the sample points; the sampling step (microns) is shown in the legend.
    ax.scatter(samples[:, 0], samples[:, 1], s=8, color=colors["annot"],
               edgecolors="k", linewidths=0.2, zorder=3,
               label=f"samples (every {step_um:g} um, n={len(samples)})")

    # Close-up: crop to the annotation's extent (+5% padding) so the boundary
    # fills the view instead of the whole slide.
    pad = 0.05 * max(np.ptp(curve[:, 0]), np.ptp(curve[:, 1]), 1.0)
    ax.set_xlim(curve[:, 0].min() - pad, curve[:, 0].max() + pad)
    ax.set_ylim(curve[:, 1].min() - pad, curve[:, 1].max() + pad)

    ax.set_aspect("equal")     # keep perpendicular distances visually undistorted
    ax.invert_yaxis()          # image convention: y increases downward
    ax.set_title(f"{roi_name} - {key}  (mean {mean_um:.2f} um)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Public API
# ============================================================================
def annotation_to_contour_distances(annot, neuseg_res,
                                    step_um=1.0, keys=("gm_wm", "gm_csf"),
                                    verbose=False):
    """
    Annotation-to-contour distance for every ROI, in microns.

    Reuses a get_neuseg_results(...) output (its contours + mpp), so nothing is
    recomputed. The SAME call works for NEUSEG or manual annotation -- only `annot`
    changes (get_neuseg_results(...)['annot'] or get_manual_results(...)['annot']).

    Parameters
    ----------
    annot : dict        {roi: {'gm_wm': (N,2), 'gm_csf': (M,2), ...}} full-res pixels
    neuseg_res : dict   get_neuseg_results output; uses its contour_gm_wm /
                        contour_gm_csf (full-res polygons) and mpp
    step_um : float     arc-length sampling spacing along the annotation, in microns
    keys : tuple        boundary types to evaluate ('left'/'right' walls skipped)
    verbose : bool      show one close-up validation figure per (roi, boundary):
                        contour, annotation, 1-micron samples, and a hairline from
                        each sample to its nearest contour point

    Returns
    -------
    pandas.DataFrame, one row per (roi, boundary):
        roi, boundary, n_samples, mean_um, median_um, max_um, dists_um
    `dists_um` (raw per-sample array) is kept so ROIs can be re-aggregated later,
    e.g. the macro-mean across ROIs is df.groupby('boundary')['mean_um'].mean().
    """
    # Reuse the already-computed NEUSEG contours + mpp (no disk I/O, no recompute).
    contours = {"gm_wm": neuseg_res["contour_gm_wm"], "gm_csf": neuseg_res["contour_gm_csf"]}
    mpp = neuseg_res["mpp"]

    # 1 micron expressed in full-res pixels -> the annotation sampling step.
    step_px = step_um / mpp

    # Build each contour geometry ONCE and reuse it across all ROIs. (Contours are
    # per-slide, not per-ROI, so there is no need to rebuild them inside the loop.)
    contour_geoms = {k: _contour_to_geometry(contours.get(k)) for k in keys}

    rows = []
    for roi_name, boundaries in annot.items():
        for key in keys:
            curve = boundaries.get(key)          # annotation curve for this boundary
            geom = contour_geoms.get(key)        # matching contour geometry

            # Skip if: this ROI has no such boundary, there is no contour to
            # measure against, or the curve is too short to sample.
            if curve is None or geom is None or np.asarray(curve).shape[0] < 2:
                continue

            samples = _sample_polyline(curve, step_px)   # (M,2), one pt per micron
            if samples.shape[0] == 0:
                continue

            # EXACT distance from each sampled point to the contour LINE, in px.
            # shapely.points(samples) -> array of M Point geometries;
            # shapely.distance broadcasts the single contour geometry against all
            # of them, returning an (M,) array of nearest-distance values.
            d_px = shapely.distance(shapely.points(samples), geom)
            d_um = d_px * mpp                            # pixels -> microns
            mean_um = float(np.mean(d_um))

            # Optional visual validation for this (ROI, boundary).
            if verbose:
                _plot_distance_validation(roi_name, key, curve, samples, geom,
                                          contours.get(key), step_um, mean_um)

            rows.append({
                "roi":       roi_name,
                "boundary":  key,
                "n_samples": int(d_um.size),
                "mean_um":   mean_um,
                "median_um": float(np.median(d_um)),
                "max_um":    float(np.max(d_um)),
                "dists_um":  d_um,                       # raw, for later re-agg
            })

    return pd.DataFrame(
        rows,
        columns=["roi", "boundary", "n_samples",
                 "mean_um", "median_um", "max_um", "dists_um"],
    )
