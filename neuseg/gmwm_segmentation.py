#!/usr/bin/env python3
"""GM/WM segmentation of cortex from the NEUSEG soma feature heatmaps."""

import os

import numpy as np
import maxflow
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba, ListedColormap
from matplotlib.patches import Ellipse, Patch
from skimage.transform import resize
from scipy.ndimage import label, binary_dilation
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

import pdnl_sana.image
import pdnl_sana.slide

GM_C, WM_C = '#7b3fa0', '#2ca02c'   # purple / green


def _finish(fig, output_directory, name, dpi=130):
    """Write `fig` into output_directory under `name`, or show it if no directory."""
    if output_directory is None:
        plt.show()
    else:
        fig.savefig(os.path.join(output_directory, name), dpi=dpi, bbox_inches='tight')
        plt.close(fig)


def resample_mask_to_features(tissue_mask, shape):
    """Resample a thumbnail-resolution tissue mask onto the feature heatmap grid.

    tissue_mask : sana Frame tissue mask at thumbnail resolution, values 0/1
    shape       : (n_rows, n_cols) of the feature heatmap to match

    Returns a bool array of `shape`, True where tissue.
    """
    # --- (1) Drop the trailing channel axis the sana Frame carries ---
    tissue_mask_thumb = tissue_mask.img.squeeze()

    # --- (2) Resample onto the feature grid ---
    # The heatmap is the thumbnail divided by --ds_thumbnail, so the two only
    # already agree when that is 1.  order=0 / no anti-aliasing: it is a binary
    # image, not an intensity image, so the values must stay discrete.
    return resize(tissue_mask_thumb, shape, order=0,
                  preserve_range=True, anti_aliasing=False).astype(bool)

def drop_outliers(soma_density, soma_size, factor=1.5):
    """Drop empty and outlying pixels so the GMM is fit on the bulk of the data.

    soma_density, soma_size : 1D tissue-pixel arrays, same length
    factor                  : Tukey fence width, in IQRs

    Returns (inlier_mask, density_inliers, size_inliers), where `inlier_mask` is
    a bool array over the input arrays.
    """
    def mask_iqr(x):
        """Tukey inlier mask: True where x is within `factor` IQRs of the quartiles."""
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        return (x >= q1 - factor*iqr) & (x <= q3 + factor*iqr)

    # --- (1) Drop pixels with no cells at all: they carry no signal ---
    # nonzero - 1D np bool array / Shape: full tissue-pixel length
    # True  = keep, pixel has signal / False = both features are 0
    nonzero = ~((soma_density == 0) & (soma_size == 0))

    # --- (2) Tukey fences on the survivors, both features must agree ---
    inliers = np.zeros_like(nonzero) # All False first
    # inliers - 1D np bool array  / Shape: full tissue-pixel length
    # True = pixel has signal AND both features are not ouliers / False = otherwise
    inliers[nonzero] = mask_iqr(soma_density[nonzero]) & mask_iqr(soma_size[nonzero])

    return inliers, soma_density[inliers], soma_size[inliers]

DENSITY_CMAP, SIZE_CMAP = 'viridis', 'magma'
DROPPED_C = '#00e5ff'   # cyan, for pixels drop_outliers removed

def viz_gmm_result(tb, tissue_mask_features, soma_density, soma_size, inliers,
                   X_scaled_tissue, labels, gmm, gm_label,
                   logger=None, input_slide=None, n_scatter=20000, output_directory=None):
    """Eight-panel diagnostic of the GM/WM fit, from raw cells through to the labels.

    tb                   : sana Frame thumbnail, RGB uint8
    tissue_mask_features : (h, w) bool tissue mask on the feature grid
    soma_density, soma_size : (h, w) feature maps, raw units
    inliers              : (N,) bool, which tissue pixels survived drop_outliers
    X_scaled_tissue      : (N, 2) standardized [density, size] for every tissue pixel
    labels               : (N,) fitted component per tissue pixel
    gmm                  : the fitted GaussianMixture
    gm_label             : which component is GM; the other is WM
    logger               : pdnl_sana logger, needed to open the slide below
    input_slide          : path to the WSI; with it the cell overlay is drawn,
                           without it that one panel is left empty
    n_scatter            : tissue pixels to subsample into the scatter panels
    output_directory     : cells.npy is read from here and gmm_result.png written
                           here; the figure is shown instead if None

    Returns None.
    """
    # --- (0) Cells for the overlay panel; nothing else in the pipeline needs them ---
    # They are in slide (level 0) coordinates, so the slide is opened purely for
    # the downsample that puts them on the thumbnail.
    cells, cells_ds = None, None
    cells_f = os.path.join(output_directory, 'cells.npy') if output_directory else None
    if input_slide is not None and cells_f is not None and os.path.exists(cells_f):
        cells = np.load(cells_f)
        loader = pdnl_sana.slide.Loader(logger, input_slide)
        try:
            cells_ds = loader.converter.ds[loader.thumbnail_level]
        finally:
            loader.close()

    # --- (1) Scatter the 1D per-tissue-pixel arrays back onto the feature grid ---
    # labels/inliers hold one entry per TISSUE pixel, so they are written back
    # through the same mask that flattened the image in the first place.
    label_map = np.zeros(tissue_mask_features.shape, dtype=np.uint8)  # 0 none, 1 GM, 2 WM
    label_map[tissue_mask_features] = np.where(labels == gm_label, 1, 2)
    dropped_map = np.zeros(tissue_mask_features.shape, dtype=bool)
    dropped_map[tissue_mask_features] = ~inliers

    label_thumb = resize(label_map, tb.img.shape[:2], order=0,
                         preserve_range=True, anti_aliasing=False).astype(np.uint8)

    # --- (2) Colour scales taken from the INLIERS, so outliers do not set the range ---
    d_tissue, s_tissue = soma_density[tissue_mask_features], soma_size[tissue_mask_features]
    d_lo, d_hi = d_tissue[inliers].min(), d_tissue[inliers].max()
    s_lo, s_hi = s_tissue[inliers].min(), s_tissue[inliers].max()

    rng = np.random.default_rng(0)
    idx = rng.choice(len(labels), size=min(n_scatter, len(labels)), replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(27, 13))

    def feature_panel(ax, fmap, lo, hi, cmap, title):
        """Feature map over tissue only, with the drop_outliers pixels marked."""
        m = np.ma.masked_where(~tissue_mask_features, fmap)
        im = ax.imshow(m, cmap=cmap, vmin=lo, vmax=hi, interpolation='nearest')
        ax.imshow(np.ma.masked_where(~dropped_map, dropped_map),
                  cmap=ListedColormap([DROPPED_C]), interpolation='nearest')
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        ax.legend(handles=[Patch(facecolor=DROPPED_C,
                                 label=f'dropped  {100*dropped_map.sum()/tissue_mask_features.sum():.1f}%')],
                  loc='lower left', fontsize=8, framealpha=0.9)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    # ---------------- Row 1 ----------------
    axes[0,0].imshow(tb.img)
    axes[0,0].set_title('Thumbnail', fontsize=10)
    axes[0,0].axis('off')

    # (1,2) segmented cells over the thumbnail, as run_features plots them
    ax = axes[0,1]
    ax.imshow(tb.img)
    if cells is not None and cells_ds is not None:
        ds_cells = 4                                  # subsample, as in run_features
        ax.plot(cells[::ds_cells,0]/cells_ds, cells[::ds_cells,1]/cells_ds,
                '*', markersize=1, color='red')
        ax.set_title(f'({ds_cells}x) subsampled cell centres  --  {len(cells)} cells', fontsize=10)
    else:
        ax.set_title('cells not supplied', fontsize=10)
    ax.axis('off')

    feature_panel(axes[0,2], soma_density, d_lo, d_hi, DENSITY_CMAP,
                  f'Soma density  --  inlier range [{d_lo:.3g}, {d_hi:.3g}]')
    feature_panel(axes[0,3], soma_size, s_lo, s_hi, SIZE_CMAP,
                  f'Soma size  --  inlier range [{s_lo:.3g}, {s_hi:.3g}]')

    # ---------------- Row 2 ----------------
    # (2,1) the raw GMM segmentation, one opaque layer so classes do not blend
    ax = axes[1,0]
    ax.imshow(tb.img)
    overlay = np.zeros((*label_thumb.shape, 4))
    overlay[label_thumb == 1] = to_rgba(GM_C, 0.85)
    overlay[label_thumb == 2] = to_rgba(WM_C, 0.85)
    ax.imshow(overlay, interpolation='nearest')
    n_tissue = int((label_map > 0).sum())
    pct = lambda v: 100 * (label_map == v).sum() / n_tissue
    ax.legend(handles=[Patch(facecolor=GM_C, label=f'GM  {pct(1):.1f}% of tissue'),
                       Patch(facecolor=WM_C, label=f'WM  {pct(2):.1f}% of tissue')],
              loc='lower left', fontsize=8, framealpha=0.9)
    ax.set_title(f'GM/WM straight out of the GMM  --  {n_tissue} tissue px', fontsize=10)
    ax.axis('off')

    # Shared limits for the three scatter panels: clip to the bulk, since a few
    # extreme outliers otherwise squeeze both components into one corner.
    lo, hi = np.percentile(X_scaled_tissue, [0.5, 99.5], axis=0)
    pad = 0.35 * (hi - lo)
    xlim = (lo[0] - pad[0], hi[0] + pad[0])
    ylim = (lo[1] - pad[1], hi[1] + pad[1])

    def scatter_axes(ax, title):
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel('soma density (standardized)')
        ax.set_ylabel('soma size (standardized)')
        ax.set_title(title, fontsize=10)

    # (2,2) coloured by fitted component, with the model's own contours
    ax = axes[1,1]
    for comp, color, name in [(gm_label, GM_C, 'GM'), (1 - gm_label, WM_C, 'WM')]:
        sel = idx[labels[idx] == comp]
        ax.scatter(X_scaled_tissue[sel,0], X_scaled_tissue[sel,1],
                   s=2, alpha=0.25, color=color, linewidths=0, label=f'{name} pixels')
        mu, cov = gmm.means_[comp], gmm.covariances_[comp]
        ax.plot(mu[0], mu[1], 'x', color=color, ms=14, mew=2.5)
        # eigh returns ascending eigenvalues, so the last is the major axis;
        # an n-std ellipse is the 1-std one scaled by n.
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vecs[1,-1], vecs[0,-1]))
        for n_std in (1, 2, 3):
            ax.add_patch(Ellipse(mu, 2*n_std*np.sqrt(vals[-1]), 2*n_std*np.sqrt(vals[0]),
                                 angle=angle, fc='none', ec=color, lw=1.0,
                                 alpha=1.0 - 0.2*(n_std - 1)))
    ax.legend(fontsize=8, markerscale=4, loc='upper right')
    scatter_axes(ax, f'2D GMM  --  x = mean, ellipses = 1/2/3 std  ({len(idx)} px shown)')

    # (2,3) and (2,4) the same points, coloured by raw feature value on the row-1
    # scales -- so a panel here reads as the spatial map above it, in feature space
    for col, (vals_1d, lo_v, hi_v, cmap, name) in enumerate(
            [(d_tissue, d_lo, d_hi, DENSITY_CMAP, 'soma density'),
             (s_tissue, s_lo, s_hi, SIZE_CMAP, 'soma size')], start=2):
        ax = axes[1,col]
        sc = ax.scatter(X_scaled_tissue[idx,0], X_scaled_tissue[idx,1], c=vals_1d[idx],
                        cmap=cmap, vmin=lo_v, vmax=hi_v, s=2, alpha=0.5, linewidths=0)
        fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        scatter_axes(ax, f'coloured by {name} (row 1 scale)')

    fig.tight_layout()
    _finish(fig, output_directory, 'gmm_result.png')

def run_gmm(tb, tissue_mask, features, logger=None, debug=False,
            output_directory=None, **kwargs):
    """Segment grey vs white matter from the soma feature heatmaps.

    tb          : sana Frame thumbnail, RGB uint8
    tissue_mask : sana Frame tissue mask at thumbnail resolution
    features    : (H, W, 3) feature heatmap -- soma density, size, intensity
    logger      : optional pdnl_sana logger for progress output
    debug       : draw the eight-panel figure through viz_gmm_result
    output_directory : where that figure goes; shown instead if None
    kwargs      : input_slide -- for the debug figure's cell overlay only

    Returns (gm_prob, tissue_mask_features): the (H, W) GM posterior, zero outside
    the tissue, and the (H, W) bool tissue mask it was scored on.  Both feed
    post_process, which turns them into the final masks.
    """
    # --- (1) Split the heatmap into the features the GMM is fit on ---
    soma_density, soma_size = features[:,:,0], features[:,:,1]

    # --- (2) Put the tissue mask on the same grid as the features ---
    tissue_mask_features = resample_mask_to_features(tissue_mask, soma_density.shape)

    # --- (3) Keep only the tissue pixels ---
    soma_density_tissue = soma_density[tissue_mask_features]
    soma_size_tissue = soma_size[tissue_mask_features]

    # --- (4) Drop empty and outlying pixels before fitting ---
    inliers, soma_density_inliers, soma_size_inliers = drop_outliers(soma_density_tissue, soma_size_tissue)

    # --- (5) Standardize the features ---
    # The GMM is fit on the inliers, so the scaler is fit on them.
    X_tissue = np.stack([soma_density_tissue, soma_size_tissue], axis=1)
    X_inliers = np.stack([soma_density_inliers, soma_size_inliers], axis=1)
    scaler = StandardScaler()
    X_scaled_inliers = scaler.fit_transform(X_inliers)

    # --- (6) Fit the 2D GMM and work out which component is GM ---
    # No means_init: the components are placed by EM alone, then identified below.
    # n_init=5: with a single k-means start EM lands in a poor local optimum on
    # roughly half of all seeds, splitting the data along soma size instead of the
    # density gradient.  Restarting and keeping the best log-likelihood removes
    # that coin flip, which otherwise shows up as sensitivity to --window_size.
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0, n_init=5)
    gmm.fit(X_scaled_inliers)
    if logger is not None and not gmm.converged_:
        logger.warning(f"GMM did not converge in {gmm.n_iter_} iterations")

    # GM has larger somas AND lower density than WM.  When the two features
    # disagree on which component is which, trust the better separated one.
    # NOTE: Agree case is absorbed because when the features agree, the two rules are the same answer
    means = gmm.means_                              # columns: [density, size]
    if abs(means[0,1] - means[1,1]) >= abs(means[0,0] - means[1,0]):
        gm_label = int(np.argmax(means[:,1]))       # larger soma size
    else:
        gm_label = int(np.argmin(means[:,0]))       # lower soma density
    if logger is not None:
        # the means are what chose gm_label, so log them together: an inverted
        # slide is diagnosable from this line alone
        logger.info(f"GMM: GM is component {gm_label}, means (standardized) "
                    f"GM [density {means[gm_label,0]:+.2f}, size {means[gm_label,1]:+.2f}] "
                    f"WM [density {means[1-gm_label,0]:+.2f}, size {means[1-gm_label,1]:+.2f}]")

    # --- (7) Score every tissue pixel, not just the inliers ---
    # Same scaler as the fit, so the outliers land where the model expects them.
    X_scaled_tissue = scaler.transform(X_tissue)
    labels = gmm.predict(X_scaled_tissue)           # (N,)   component per pixel
    probs = gmm.predict_proba(X_scaled_tissue)      # (N, 2) posterior per pixel

    # --- (8) Reconstruct the segmentation and show it ---
    if debug:
        viz_gmm_result(tb, tissue_mask_features, soma_density, soma_size, inliers,
                       X_scaled_tissue, labels, gmm, gm_label,
                       logger=logger, input_slide=kwargs.get('input_slide'),
                       output_directory=output_directory)

    # Scatter the per-tissue-pixel GM posterior back onto the grid.  post_process
    # takes it from here: the CRF needs the posterior itself, and the hard
    # segmentation this figure just drew is recoverable as gm_prob > 0.5.
    gm_prob = np.zeros(tissue_mask_features.shape, dtype=float)
    gm_prob[tissue_mask_features] = probs[:, gm_label]
    
    return gm_prob, tissue_mask_features

def crf_potts(gm_prob, tissue_mask, beta=None, connectivity=8, eps=1e-6):
    """Spatially regularize the per-pixel GM posteriors with a Potts-model graph cut.

    The GMM scores every pixel on its own, so its labels are noisy wherever the
    two components overlap.  Minimizing  sum -log p(label) + beta * (# disagreeing
    neighbour pairs)  buys spatial coherence back.  Being a two-label Potts energy
    with non-negative pairwise costs, the graph cut finds the global optimum.

    gm_prob      : (h, w) float, GM posterior; only tissue pixels are read
    tissue_mask  : (h, w) bool, which pixels become graph nodes
    beta         : smoothness strength; None uses the p_target default below
    connectivity : 4 or 8 neighbours
    eps          : probability floor, so a confident pixel cannot cost -inf

    Returns (gm_mask, n_flipped): bool (h, w), and how many pixels the cut moved.
    """
    # --- (1) Smoothness strength, in the -log p units of the unary costs ---
    # The default is the cost of overruling a p_target posterior, divided over the
    # neighbours pulling on each pixel (more of them at 8-connectivity).
    if beta is None:
        p_target = 0.75
        k = 1.0 if connectivity == 4 else 1.5
        beta = abs(np.log((1.0 - p_target) / p_target)) / k

    # --- (2) Unary cost of each label: the negative log posterior ---
    d_gm = -np.log(np.clip(gm_prob, eps, 1.0))
    d_wm = -np.log(np.clip(1.0 - gm_prob, eps, 1.0))

    # --- (3) One graph node per tissue pixel ---
    # idx carries each tissue pixel's node id and -1 elsewhere, so the edge
    # building below can tell in one comparison whether a neighbour is a node.
    idx = np.full(tissue_mask.shape, -1, dtype=np.int32)
    n_nodes = int(tissue_mask.sum())
    idx[tissue_mask] = np.arange(n_nodes, dtype=np.int32)
    graph = maxflow.Graph[float](n_nodes, connectivity * n_nodes)
    graph.add_nodes(n_nodes)

    # SOURCE side is GM, SINK side is WM.  A node kept on the source side pays
    # its sink capacity, so the two costs go in swapped.
    graph.add_grid_tedges(idx[tissue_mask], d_wm[tissue_mask], d_gm[tissue_mask])

    # --- (4) Potts edges between neighbouring tissue pixels ---
    # Only half the offsets: add_edges is already bidirectional, so adding the
    # mirrored ones would double every capacity.  Dividing by the step length
    # makes a diagonal cut cost the same per unit of boundary as an axial one.
    offsets = [(0,1), (1,0)] if connectivity == 4 else [(0,1), (1,0), (1,1), (1,-1)]
    h, w = tissue_mask.shape
    for dy, dx in offsets:
        src_y, dst_y = slice(0, h - dy), slice(dy, h)
        src_x, dst_x = ((slice(0, w - dx), slice(dx, w)) if dx >= 0 else
                        (slice(-dx, w), slice(0, w + dx)))
        u, v = idx[src_y, src_x], idx[dst_y, dst_x]
        both = (u >= 0) & (v >= 0)              # drop pairs that leave the tissue
        cap = np.full(int(both.sum()), beta / np.hypot(dy, dx))
        graph.add_edges(u[both], v[both], cap, cap)

    # --- (5) Solve, and read the partition back onto the grid ---
    graph.maxflow()
    gm_mask = np.zeros(tissue_mask.shape, dtype=bool)
    gm_mask[tissue_mask] = ~graph.get_grid_segments(idx[tissue_mask])   # sink = WM
    n_flipped = int((gm_mask != (gm_prob > 0.5))[tissue_mask].sum())
    return gm_mask, n_flipped

CHANGE_C = '#ff1744'   # red, for pixels a post-processing step moved
def viz_post_process(gm_raw, gm_crf, gm_final, tissue, output_directory=None):
    """Three panels walking through post_process, marking what each step changed.

    gm_raw   : (h, w) bool GM straight off the GMM posterior
    gm_crf   : (h, w) bool GM after the graph cut
    gm_final : (h, w) bool GM after island pruning
    tissue   : (h, w) bool tissue mask all three share
    output_directory : post_process.png written here, or the figure is shown

    Returns None.
    """
    n_tissue = max(1, int(tissue.sum()))
    disc = lambda r: np.ones((2*r + 1,) * 2, dtype=bool)

    # fill=True marks the changed pixels themselves; fill=False rings them instead,
    # so a block of relabelled pixels still shows which class it ended up in.  Both
    # are dilated well past their true extent because the panels are drawn at about
    # a third of mask resolution and the CRF moves scattered single pixels -- read
    # the legend, not the area, for how much actually changed.
    FILL_R, RING_R = 6, 3
    steps = [('GMM posterior > 0.5', gm_raw,   None,   False),
             ('after CRF',           gm_crf,   gm_raw, True),
             ('after island prune',  gm_final, gm_crf, False)]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    for ax, (name, gm, prev, fill) in zip(axes, steps):
        # --- (1) Flat GM/WM colour, non-tissue left white ---
        img = np.ones((*tissue.shape, 4))
        img[tissue & gm] = to_rgba(GM_C)
        img[tissue & ~gm] = to_rgba(WM_C)
        ax.imshow(img, interpolation='nearest')
        handles = [Patch(facecolor=GM_C, label=f'GM  {100*(gm & tissue).sum()/n_tissue:.1f}%'),
                   Patch(facecolor=WM_C, label=f'WM  {100*(~gm & tissue).sum()/n_tissue:.1f}%')]

        # --- (2) Mark what this step moved, relative to the panel on its left ---
        if prev is not None:
            changed = (gm != prev) & tissue
            grown = binary_dilation(changed, disc(FILL_R if fill else RING_R))
            marked = grown if fill else (grown & ~changed)
            ax.imshow(np.ma.masked_where(~marked, marked),
                      cmap=ListedColormap([CHANGE_C]), interpolation='nearest')
            handles.append(Patch(facecolor=CHANGE_C,
                                 label=f'changed  {int(changed.sum())} px '
                                       f'({100*changed.sum()/n_tissue:.3f}%)'))

        ax.legend(handles=handles, loc='lower left', fontsize=8, framealpha=0.9)
        ax.set_title(name, fontsize=11)
        ax.axis('off')

    fig.tight_layout()
    _finish(fig, output_directory, 'post_process.png')

def post_process(gm_prob, tissue_mask, mpp, beta=8, connectivity=8, min_island_mm2=1.0,
                 logger=None, debug=False, output_directory=None):
    """Turn the per-pixel GM posteriors into clean masks: CRF, then islands.

    gm_prob      : (h, w) float GM posterior from the GMM, 0 outside the tissue
    tissue_mask  : (h, w) bool tissue mask on the same grid
    mpp          : microns per pixel of that grid
    beta         : CRF smoothness strength -- the boundary tension.  It sets the
                   thinnest structure that survives, 2.414*beta/|log((1-p)/p)| px,
                   so confident tissue is held while uncertain slivers are absorbed.
    connectivity : CRF neighbourhood, 4 or 8
    min_island_mm2 : components smaller than this are removed.  A fixed area, not a
                   fraction of the section: anatomy does not scale with how much
                   tissue landed on the slide.  1 mm2 is about 1 mm across, less
                   than the cortex is thick, so nothing that small can be a real
                   GM or WM territory.
    logger       : optional pdnl_sana logger for progress output
    debug        : draw the per-step figure through viz_post_process
    output_directory : where that figure goes; shown instead if None

    Returns (gm_mask, wm_mask), bool arrays that partition the tissue.
    """
    # --- (1) Thresholds in pixels of this grid ---
    tissue = tissue_mask.astype(bool)
    tissue_area_px = int(tissue.sum())
    min_island_px = max(1, int(round(min_island_mm2 * 1e6 / mpp**2)))

    # --- (2) Spatially regularize the posteriors into hard labels ---
    gm_crf, n_flipped = crf_potts(gm_prob, tissue, beta=beta, connectivity=connectivity)

    # --- (3) Drop connected components too small to be real anatomy ---
    def prune_islands(mask):
        """Clear components of `mask` under min_island_px; returns (mask, n, px)."""
        cc, n = label(mask)
        if n == 0:
            return mask, 0, 0
        counts = np.bincount(cc.ravel())
        counts[0] = 0                                   # background is not an island
        small = np.where((counts > 0) & (counts < min_island_px))[0]
        if small.size == 0:
            return mask, 0, 0
        out = mask.copy()
        out[np.isin(cc, small)] = False
        return out, int(small.size), int(counts[small].sum())

    # WM first, then GM against the updated complement, so no tissue pixel is
    # left unlabelled: whatever is pruned from one class is absorbed by the other.
    wm, n_wm, px_wm = prune_islands(tissue & ~gm_crf)
    gm, n_gm, px_gm = prune_islands(tissue & ~wm)
    wm = tissue & ~gm

    # --- (4) Report ---
    if logger is not None:
        logger.info(f"post-processing: CRF beta {beta:g}, island floor "
                    f"{min_island_mm2:g} mm2 = {min_island_px} px "
                    f"(tissue {tissue_area_px} px = {tissue_area_px*mpp**2/1e6:.1f} mm2)")
        pct = lambda px: 100 * px / max(1, tissue_area_px)
        logger.info(f"post-processing: CRF flipped {n_flipped} px ({pct(n_flipped):.3f}% of tissue)")
        logger.info(f"post-processing: pruned {n_wm} WM islands ({px_wm} px) and "
                    f"{n_gm} GM islands ({px_gm} px), {pct(px_wm + px_gm):.3f}% of tissue")

    # --- (5) Debug figure, one panel per step above ---
    if debug:
        viz_post_process((gm_prob > 0.5) & tissue, gm_crf, gm, tissue,
                         output_directory=output_directory)

    return gm, wm


# render_contours: the three boundary lines, then the light fills beneath them
GMWM_C, GMBG_C, WMBG_C = '#00c800', '#dc00ff', '#000000'
GM_FILL_C, WM_FILL_C, FILL_ALPHA = '#d9b3e6', '#bff2bf', 0.25

def render_contours(tb, gm_mask, wm_mask, tissue_mask, line_radius_px=3,
                    show_fills=True, title=None, debug=False, output_directory=None):
    """Trace the GM/WM boundaries as polygons in thumbnail pixel coordinates.

    tb               : sana Frame thumbnail, RGB uint8; only used for the figure
    gm_mask, wm_mask : (h, w) bool final masks, at the thumbnail resolution
    tissue_mask      : (h, w) bool tissue mask; its complement is the background
    line_radius_px   : figure only -- lines are grown to 2r+1 px to stay visible
    show_fills       : figure only -- tint the GM and WM regions under the lines
    title            : optional figure title
    debug            : draw GMWM_contour.png as well as returning the polygons
    output_directory : where that figure goes; shown instead if None

    Returns {'shape': [h, w], 'gm_wm': [...], 'gm_csf': [...]}, each boundary a list
    of [[x, y], ...] rings in thumbnail pixels, ready to hand to json.dump.
    """
    # --- (1) Make the three regions exclusive and tissue-bounded ---
    tissue = tissue_mask.astype(bool)
    gm = gm_mask.astype(bool) & tissue
    wm = wm_mask.astype(bool) & tissue & ~gm         # GM wins any overlap
    bg = ~tissue

    # --- (2) Trace the two boundaries the downstream analysis names ---
    # Same derivation as evaluation/Helper_Functions/loading.py: the WM outline is
    # the GM-WM boundary, the tissue outline is the GM-CSF boundary.  to_polygons
    # returns (bodies, holes) and the holes are boundaries too -- a WM island inside
    # GM, an enclosed CSF space inside the tissue -- so both go in.
    def rings(mask):
        bodies, holes = pdnl_sana.image.Frame(mask.astype(np.uint8)).to_polygons()
        return [np.asarray(ring).tolist() for ring in bodies + holes]

    contours = {'shape': list(tissue.shape),
                'gm_wm': rings(wm),
                'gm_csf': rings(tissue)}

    if not debug:
        return contours

    # --- (3) One-pixel boundaries for the figure, each drawn on its own side ---
    # Dilating the NEIGHBOUR and intersecting keeps every line inside the region it
    # belongs to, so a GM-WM line cannot also be claimed as a GM-background line.
    se = np.ones((3, 3), dtype=bool)
    gm_wm = gm & binary_dilation(wm, structure=se)
    gm_bg = gm & binary_dilation(bg, structure=se) & ~gm_wm    # GM-WM takes priority
    wm_bg = wm & binary_dilation(bg, structure=se)

    # --- (4) Collapse the three disjoint sets into one label map, then thicken ---
    # Growing them one label at a time preserves the priority: dilating all three
    # at once would let a later line overwrite an earlier one where they overlap.
    lbl = np.zeros(tissue.shape, dtype=np.uint8)     # 0 none, 1 GM-WM, 2 GM-BG, 3 WM-BG
    lbl[gm_wm], lbl[gm_bg], lbl[wm_bg] = 1, 2, 3
    se_thick = np.ones((2*line_radius_px + 1,) * 2, dtype=bool)
    lbl_thick = np.zeros_like(lbl)
    for k in (1, 2, 3):
        grown = binary_dilation(lbl == k, structure=se_thick)
        lbl_thick[(lbl_thick == 0) & grown] = k

    # --- (5) Draw: thumbnail, then fills, then lines on top ---
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(tb.img)

    fill_handles = []
    if show_fills:
        for region, color, name in ((gm, GM_FILL_C, 'GM'), (wm, WM_FILL_C, 'WM')):
            fill = np.zeros((*tissue.shape, 4))
            fill[region] = to_rgba(color, FILL_ALPHA)
            ax.imshow(fill, interpolation='nearest')
            fill_handles.append(Patch(facecolor=color, alpha=FILL_ALPHA, label=f'{name} (fill)'))

    overlay = np.zeros((*tissue.shape, 4))
    line_handles = []
    for k, color, name in ((1, GMWM_C, 'GM-WM'), (2, GMBG_C, 'GM-background'),
                           (3, WMBG_C, 'WM-background')):
        overlay[lbl_thick == k] = to_rgba(color, 0.98)
        line_handles.append(Patch(facecolor=color, label=name))
    ax.imshow(overlay, interpolation='nearest')

    ax.legend(handles=line_handles + fill_handles, loc='lower right',
              fontsize=8, framealpha=0.9)
    if title is not None:
        ax.set_title(title, fontsize=11)
    ax.axis('off')
    fig.tight_layout()
    _finish(fig, output_directory, 'GMWM_contour.png', dpi=300)

    return contours
