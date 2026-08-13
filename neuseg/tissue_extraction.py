#!/usr/bin/env python3
"""Tissue mask extraction from a WSI thumbnail, thresholded off the background peak."""

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.morphology import remove_small_holes
from skimage import measure

import pdnl_sana as sana
import pdnl_sana.image
import pdnl_sana.filter


def background_threshold(img, bins=None, smooth_div=100, peak_prom=0.05, valley_prom=0.005):
    """Find the grey level separating blank slide from tissue, anchored on the background peak.

    img         : 2D float array to threshold, i.e. the blurred `1 - gray`
    bins        : histogram bins, default Rice rule 2 * N**(1/3)
    smooth_div  : histogram smoothed with sigma = bins / smooth_div (~1% of the range),
                  which suppresses the RGB->gray quantization comb
    peak_prom   : min peak prominence as a fraction of the tallest bin
    valley_prom : min valley prominence, same units

    Returns (threshold, bg_peak, centers, hist, rule).  `rule` is 'valley' or
    'plateau'; `threshold` is None only if the histogram never stops descending.
    """
    # --- (1) Smoothed histogram of the image ---
    if bins is None:
        bins = int(np.ceil(2 * img.size ** (1 / 3)))

    hist, edges = np.histogram(img, bins=bins, range=(img.min(), img.max()))
    hist = gaussian_filter1d(hist.astype(float), bins / smooth_div)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # --- (2) Anchor on the background peak = leftmost prominent peak ---
    # Background is blank glass, so it is always the leftmost mode of `1 - gray`.
    # Pad a zero bin onto each end first --> To detect the background peak robustly
    hp = np.concatenate(([0.0], hist, [0.0]))
    peaks,   _ = find_peaks( hp, prominence=peak_prom   * hist.max())
    valleys, _ = find_peaks(-hp, prominence=valley_prom * hist.max())
    peaks, valleys = peaks - 1, valleys - 1 # Index -1 due to padding

    i_bg = peaks[0] if len(peaks) else int(np.argmax(hist))

    # --- (3) Threshold at the first valley right of the anchor, else at the plateau ---
    right = valleys[valleys > i_bg]
    if len(right):
        i_th, rule = int(right[0]), 'valley'
    else:
        d = np.diff(hist)                       # d[k] = hist[k+1] - hist[k]
        k = i_bg + 1                            # descending, so d is negative here
        while k < len(d) and d[k] <= d[k - 1]:  # walk while it is still steepening
            k += 1
        i_th, rule = (k if k < len(d) else None), 'plateau'

    thresh = centers[i_th] if i_th is not None else None
    return thresh, centers[i_bg], centers, hist, rule


def get_tissue_mask(tb):
    """Build a tissue mask from a WSI thumbnail.

    tb : sana.image.Frame thumbnail, RGB uint8

    Returns (tissue_mask, stages).  `tissue_mask` is a short sana Frame, 1 where
    tissue.  `stages` is a dict of per-step intermediates for tissue_mask_debug.
    """
    # --- (1) RGB --> inverted grayscale ---
    # Tissue is DARKER than the slide background, so invert: high = tissue.
    GRAY_SIGMA = 1.0  # blurring amount
    gray_inv = 1.0 - rgb2gray(tb.img.astype(np.float32) / 255.0).astype(np.float32)
    wsi_gray = gaussian(gray_inv, sigma=GRAY_SIGMA)  # blur to stabilize thresholding

    # --- (2) Threshold just right of the background peak ---
    thresh, bg_peak, hist_x, hist_y, rule = background_threshold(wsi_gray)
    if thresh is None:
        raise ValueError('no background/tissue threshold found in the histogram')

    # --- (3) Initial mask + smoothing (closing -> opening) ---
    tissue_mask_bool = (wsi_gray > thresh).astype(np.uint8)
    tissue_mask = sana.image.frame_like(tb, tissue_mask_bool)
    tissue_mask.to_short()
    tissue_mask.apply_morphology_filter(sana.filter.MorphologyFilter('closing', 'ellipse', 2))
    tissue_mask.apply_morphology_filter(sana.filter.MorphologyFilter('opening', 'ellipse', 15))
    mask_smooth = (tissue_mask.img.squeeze() > 0)  # back to np ndarray

    # --- (4) Connected-component gating: drop small blobs touching the WSI edge ---
    mask = mask_smooth.astype(np.uint8)
    Hh, Ww = mask.shape
    labels = measure.label(mask, connectivity=2)

    if labels.max() == 0:  # no tissue found, return empty mask
        kept = np.zeros_like(mask, dtype=np.uint8)
    else:
        props = measure.regionprops(labels)

        def touches_border(p):  # does this region touch the border of the image?
            minr, minc, maxr, maxc = p.bbox  # maxr/maxc exclusive
            return (minr == 0) or (minc == 0) or (maxr == Hh) or (maxc == Ww)

        areas = np.array([p.area for p in props])
        lbls  = np.array([p.label for p in props])
        touch = np.array([touches_border(p) for p in props])

        # Keep everything off the border, the largest blob, and any border-touching
        # blob at least 5% of the WSI -- so only small edge debris is dropped.
        area_thresh = 0.05 * Hh * Ww
        big_border_labels = set(lbls[(touch) & (areas >= area_thresh)])
        largest_label = lbls[np.argmax(areas)]
        keep_labels = set(lbls[~touch]) | {largest_label} | big_border_labels

        kept = np.isin(labels, list(keep_labels)).astype(np.uint8)

    # --- (5) Fill small internal holes only ---
    SMALL_HOLE_FRAC = 0.001  # 0.1% of tissue area
    kept_bool = kept.astype(bool)
    tissue_area_px = int(kept_bool.sum())
    min_hole_area_px = max(1, int(round(SMALL_HOLE_FRAC * tissue_area_px)))
    kept_filled = remove_small_holes(kept_bool, area_threshold=min_hole_area_px, connectivity=2)

    # --- (6) Finalize (SANA frame) ---
    tissue_mask = sana.image.frame_like(tb, kept_filled.astype(np.uint8))
    tissue_mask.to_short()

    stages = {
        'gray_sigma': GRAY_SIGMA,
        'wsi_gray': wsi_gray,
        'thresh': thresh, 'bg_peak': bg_peak, 'rule': rule,
        'hist_x': hist_x, 'hist_y': hist_y,
        'mask_threshold': tissue_mask_bool,          # (3) straight after thresholding
        'mask_smooth': mask_smooth,                  # (4) after closing -> opening
        'mask_gated': kept_bool,                     # (5) after border gating
        'mask_filled': kept_filled,                  # (6) after hole filling
        'min_hole_area_px': min_hole_area_px,
        'mask_final': tissue_mask.img.squeeze() > 0,  # (7) final
    }
    return tissue_mask, stages


def _panel(ax, base, title, overlays=(), contours=(), cmap=None, vmin=None, vmax=None):
    """Draw one debug panel: an image with mask overlays and contour outlines.

    ax       : matplotlib Axes to draw into
    base     : 2D/3D array shown as the background image
    title    : panel title
    overlays : iterable of (bool mask, color, alpha, label)
    contours : iterable of (bool mask, color, linewidth, label)

    Returns None.  Overlays are resampled with the image, so features only a few
    pixels across wash out at small panel sizes; contours are stroked in figure
    space and stay crisp at any dpi -- use them for small features such as holes.
    """
    ax.imshow(base, cmap=cmap, vmin=vmin, vmax=vmax)
    handles = []
    for m, color, alpha, label in overlays:
        m = np.asarray(m, dtype=bool)
        ax.imshow(np.ma.masked_where(~m, m), cmap=ListedColormap([color]), alpha=alpha)
        if label:
            handles.append(Patch(facecolor=color, alpha=alpha, label=label))
    for m, color, lw, label in contours:
        m = np.asarray(m, dtype=bool)
        if m.any():
            ax.contour(m.astype(float), levels=[0.5], colors=[color], linewidths=lw)
        if label:
            handles.append(Line2D([], [], color=color, lw=max(lw, 1.2), label=label))
    if handles:
        ax.legend(handles=handles, loc='lower left', fontsize=7, framealpha=0.8)
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def tissue_mask_debug(WSI_file_name, tb, stages, debug_path):
    """Write a per-stage debug figure for one slide's tissue mask.

    WSI_file_name : slide file name, used for the figure title and output name
    tb            : the same thumbnail passed to get_tissue_mask
    stages        : the `stages` dict returned by get_tissue_mask
    debug_path    : directory to write into, created if missing

    Returns the path of the written PNG, `<debug_path>/<slide>_debug.png`.
    """
    os.makedirs(debug_path, exist_ok=True)

    TISSUE_C, BG_C, HOLE_C = 'red', '#1f77ff', '#00e5ff'

    # --- (1) Display-only derived values ---
    # Only the holes that step (5) actually filled in.
    holes_filled = stages['mask_filled'] & ~stages['mask_gated']
    pct = lambda m: 100 * np.asarray(m, dtype=bool).mean()

    # Contrast stretch for panel (1): a few near-black specks otherwise set the
    # white point far above the tissue mode and the panel renders nearly black.
    gray_vmax = float(np.percentile(stages['wsi_gray'], 99.5))

    # --- (2) Image panels ---
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    _panel(axes[0, 0], tb.img, 'Thumbnail')
    _panel(axes[0, 1], stages['wsi_gray'],
           f'(1) Inverted grayscale (1 - gray), sigma={stages["gray_sigma"]}\n'
           f'displayed over [0, {gray_vmax:.3f}] (99.5th pct)',
           cmap='gray', vmin=0.0, vmax=gray_vmax)

    # --- (3) Histogram panel, with the anchor and whichever rule set the cut ---
    ax = axes[0, 2]
    ax.plot(stages['hist_x'], stages['hist_y'], color='0.35', lw=1)
    ax.axvline(stages['bg_peak'], color=BG_C, lw=1.5,
               label=f'background peak = {stages["bg_peak"]:.4f}')
    ax.axvline(stages['thresh'], color=TISSUE_C, lw=2,
               label=f'{stages["rule"]} threshold = {stages["thresh"]:.4f}')
    ax.set_yscale('log')
    ax.set_xlabel('1 - gray')
    ax.set_ylabel('pixels (log)')
    ax.legend(fontsize=7)
    ax.set_title(f'(2) Smoothed histogram  ({len(stages["hist_x"])} bins)', fontsize=9)

    # --- (4) Mask panels, one per pipeline stage ---
    _panel(axes[0, 3], tb.img,
           f'(3) Mask right after thresholding  -  {pct(stages["mask_threshold"]):.1f}%',
           overlays=[(stages['mask_threshold'], TISSUE_C, 0.4, None)])
    _panel(axes[1, 0], tb.img,
           f'(4) After smoothing: closing(2) -> opening(15)  -  {pct(stages["mask_smooth"]):.1f}%',
           overlays=[(stages['mask_smooth'], TISSUE_C, 0.4, None)])
    _panel(axes[1, 1], tb.img,
           f'(5) After connected-component / border gating  -  {pct(stages["mask_gated"]):.1f}%',
           overlays=[(stages['mask_gated'], TISSUE_C, 0.4, None)])
    _panel(axes[1, 2], tb.img,
           f'(6) Small internal holes filled (<={stages["min_hole_area_px"]} px)',
           overlays=[(stages['mask_gated'], TISSUE_C, 0.4, 'mask before fill'),
                     (holes_filled, HOLE_C, 1.0, None)],
           contours=[(holes_filled, HOLE_C, 1.0,
                      f'filled holes ({int(holes_filled.sum())} px, {pct(holes_filled):.2f}%)')])
    _panel(axes[1, 3], tb.img,
           f'(7) Final tissue mask  -  {pct(stages["mask_final"]):.1f}%',
           overlays=[(stages['mask_final'], TISSUE_C, 0.4, None)])

    # --- (5) Save ---
    fig.suptitle(WSI_file_name, fontsize=12)
    fig.tight_layout()

    out_name = os.path.splitext(os.path.basename(WSI_file_name))[0] + '_debug.png'
    out_path = os.path.join(debug_path, out_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
