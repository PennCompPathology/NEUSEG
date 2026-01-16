import os
# force all BLAS libraries to single‑thread per process
os.environ["OPENBLAS_NUM_THREADS"]      = "1"
os.environ["OMP_NUM_THREADS"]           = "1"
os.environ["MKL_NUM_THREADS"]           = "1"

# import argparse
# import time
# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib.patches import Ellipse
# from skimage.transform import resize
# from scipy.stats import norm
# from scipy.ndimage import gaussian_filter, distance_transform_edt, label
# import matplotlib.patches as mpatches
# from scipy.ndimage import binary_dilation
# from sklearn.preprocessing import StandardScaler
# from sklearn.mixture import GaussianMixture
# import math
# import maxflow
# from matplotlib.colors import ListedColormap
# from matplotlib.patches import Patch
# from matplotlib.colors import to_rgba
# from diptest import diptest
# from typing import Optional

# import json
# import datetime

# import warnings
# warnings.filterwarnings("ignore")

# import sys
# # sys.path.insert(0, '/home/hsroh/Research/sana/src')
# import pdnl_sana as sana
# import pdnl_sana.logging
# import pdnl_sana.slide
# import pdnl_sana.filter
# import pdnl_sana.process
# import pdnl_sana.quantify

import sys
import warnings
import argparse
import time
import math
import json
import datetime
from typing import Optional

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, to_rgba

from skimage.transform import resize

from scipy.stats import norm
from scipy.ndimage import (
    gaussian_filter,
    distance_transform_edt,
    label,
    binary_dilation,
)

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

import maxflow
from diptest import diptest

import pdnl_sana as sana
import pdnl_sana.logging
import pdnl_sana.slide
import pdnl_sana.filter
import pdnl_sana.process
import pdnl_sana.quantify

warnings.filterwarnings("ignore")


################################################################## Helper Functions ##################################################################
def mask_iqr(x, factor=1.5):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return (x >= q1 - factor*iqr) & (x <= q3 + factor*iqr)

def get_soma_feature_1d_gmm_stats(soma_density,
                                  soma_size,
                                  n_components: int = 2,
                                  random_state: int = 0,
                                  ) -> dict:
    """
    Fit 1D GMMs to soma_density and soma_size and return per-component
    (mean, std) for each feature. No plotting.

    Returns
    -------
    gmm_results : dict
        {
          'soma_density': [(mean_c0, std_c0), (mean_c1, std_c1), ...],
          'soma_size'   : [(mean_c0, std_c0), (mean_c1, std_c1), ...],
        }
    """
    gmm_results = {}
    for data, key in [(soma_density, 'soma_density'), (soma_size, 'soma_size')]:
        data_1d = np.asarray(data).reshape(-1, 1)
        gmm = GaussianMixture(n_components=n_components, random_state=random_state).fit(data_1d)

        comp_stats = []
        for comp in range(n_components):
            mean = float(gmm.means_[comp, 0])

            # Robustly get variance for 1D across covariance types
            if gmm.covariance_type == 'full':
                var = float(gmm.covariances_[comp, 0, 0])
            elif gmm.covariance_type == 'diag':
                var = float(gmm.covariances_[comp, 0])
            elif gmm.covariance_type == 'tied':
                var = float(gmm.covariances_[0, 0])
            elif gmm.covariance_type == 'spherical':
                var = float(gmm.covariances_[comp])
            else:
                var = float(np.asarray(gmm.covariances_).reshape(-1)[0])

            std = float(np.sqrt(var))
            comp_stats.append((mean, std))

        gmm_results[key] = comp_stats

    return gmm_results

def fit_2d_gmm(X_scaled, density_GM_mean, size_GM_mean, density_WM_mean, size_WM_mean, use_init_centroids=True, random_state=0):
    if use_init_centroids:
        means_init = np.array([
            [density_GM_mean, size_GM_mean],    # GM centroid
            [density_WM_mean, size_WM_mean]     # WM centroid
        ])
        gmm_2d = GaussianMixture(n_components=2, means_init=means_init, covariance_type='full', random_state=random_state)
    else:
        gmm_2d = GaussianMixture(n_components=2, covariance_type='full', random_state=random_state)
    gmm_2d.fit(X_scaled)
    labels = gmm_2d.predict(X_scaled)
    probs = gmm_2d.predict_proba(X_scaled)
    return gmm_2d, labels, probs

def _normalize_minmax(arr, mask):
    """
    Min-max normalize arr into [0,1] over the region 'mask' (True=valid).
    Keeps values outside mask unchanged (unused anyway).
    """
    out = arr.astype(np.float64).copy()
    vals = out[mask]
    vmin, vmax = np.min(vals), np.max(vals)
    if vmax > vmin:
        out[mask] = (vals - vmin) / (vmax - vmin)
    else:
        out[mask] = 0.0  # degenerate case: constant feature inside mask
    return out

def _unique_neighbor_offsets(connectivity):
    """
    Half-neighborhood offsets to avoid double-adding undirected edges.
    4-connected: right, down
    8-connected: right, down, down-right, down-left
    """
    if connectivity == 4:
        return [(0, 1), (1, 0)]
    elif connectivity == 8:
        return [(0, 1), (1, 0), (1, 1), (1, -1)]
    else:
        raise ValueError("connectivity must be 4 or 8")
    
def _edge_length(dy, dx):
    """Euclidean distance between two pixel centers at offset (dy, dx)."""
    return math.hypot(dy, dx)

def build_unary_from_probs(gm_prob_map, wm_prob_map, tissue_mask, eps=1e-6):
    """
    Unary costs D_i(0), D_i(1) = -log p(label | x), clipped to [eps, 1].
    Label 0 := GM, Label 1 := WM  (consistent with gm_prob_map, wm_prob_map inputs)
    """
    # Clip probabilities to avoid −∞ costs
    p0 = np.clip(gm_prob_map, eps, 1.0)
    p1 = np.clip(wm_prob_map, eps, 1.0)

    # Convert probabilities to unary costs
    D0 = -np.log(p0)
    D1 = -np.log(p1)

    # Create a node index map for graph construction
    # Node indexing restricted to tissue pixels
    H, W = tissue_mask.shape
    idx_map = -np.ones((H, W), dtype=np.int32)  # -1 = “no node”
    y_all, x_all = np.where(tissue_mask)        # coords of tissue pixels
    idx_map[y_all, x_all] = np.arange(len(y_all), dtype=np.int32) # gives the compact node ID (0..N−1) only for tissue pixels; non-tissue stays −1.

    return D0, D1, idx_map, (y_all, x_all)

def build_edges_and_weights(
    tissue_mask,
    feature_map=None,
    beta=20.0,
    sigma=0.1,
    connectivity=4,
    contrast_sensitive=False,
    idx_map=None,
):
    """
    Builds the pairwise graph (edge list) for a grid-CRF/Potts model over your tissue pixels only.
        
        beta (float): smoothness strength.
        sigma (float): contrast scale in normalized feature units.
        idx_map (H×W int): maps each tissue pixel → compact node id

    Build (u, v, w) lists for edges, using Potts weights:
      w_ij = beta * exp( - (F_i - F_j)^2 / (2 sigma^2) ) / dist(i,j)   (contrast-sensitive)
      w_ij = beta / dist(i,j)                                          (constant Potts)
    """
    H, W = tissue_mask.shape
    
    if idx_map is None:
        raise ValueError("idx_map is required")

    if contrast_sensitive:
        if feature_map is None:
            raise ValueError("feature_map must be provided for contrast-sensitive Potts")
        # Normalize feature to [0,1] inside tissue so sigma is interpretable
        F = _normalize_minmax(feature_map.astype(np.float64), tissue_mask)
    else:
        F = None  # not used

    offsets = _unique_neighbor_offsets(connectivity)
    U, V, Wts = [], [], []

    y_all, x_all = np.where(tissue_mask)
    for y, x in zip(y_all, x_all): # Iterates over all tissue pixels
        u = idx_map[y, x] # fetches its node id
        for dy, dx in offsets: # For each neighbor offset
            # Get the neighbor corrdinate
            y2, x2 = y + dy, x + dx
            # only use this neighbor if it’s valid
            if 0 <= y2 < H and 0 <= x2 < W and tissue_mask[y2, x2]: 
                v = idx_map[y2, x2]
                dist = _edge_length(dy, dx)
                if contrast_sensitive:
                    diff = F[y, x] - F[y2, x2]
                    w = beta * math.exp(- (diff * diff) / (2.0 * sigma * sigma)) / dist
                else:
                    w = beta / dist # same weight for all neighbors at the same geometric distance.
                if w > 0:
                    U.append(u)
                    V.append(v)
                    Wts.append(w)

    return np.array(U, dtype=np.int32), np.array(V, dtype=np.int32), np.array(Wts, dtype=np.float64)

def crf_potts_graphcut(
    gm_prob_map,
    wm_prob_map,
    tissue_mask,
    feature_map_for_pairwise=None,   # e.g., grayscale, soma_density, etc.
    beta=20.0,
    sigma=0.1,
    connectivity=4,
    contrast_sensitive=False,
    eps=1e-6,
    initial_labels_binary=None,      # optional: node-order {0=GM, 1=WM}
    return_flip_map=True,
):
    """
    Minimize the binary CRF energy:
      E(y;x) = sum_i D_i(y_i) + sum_(i,j) w_ij * 1{y_i != y_j}
    with a graph-cut (exact for binary Potts with w_ij >= 0).

    Returns:
      label_map    : uint8 HxW, {0=background, 1=GM, 2=WM}
      labels_binary: uint8 length-N node labels {0=GM, 1=WM}
      energy       : float, total energy of returned labeling
      n_flipped    : int or None, # nodes flipped vs. initial_labels_binary
      frac_flipped : float or None, fraction flipped
      flip_map     : uint8 HxW or None, 1 where flipped (inside tissue), else 0
    """
    # --- Unary costs from posteriors (negative log-likelihoods) ---
    D0, D1, idx_map, (y_all, x_all) = build_unary_from_probs(
        gm_prob_map, wm_prob_map, tissue_mask, eps=eps
    )
    N = len(y_all)

    # --- Graph construction ---
    # Pre-allocate with ~N nodes and ~4N or ~8N edges as a hint
    g = maxflow.Graph[float](N, N * (4 if connectivity == 4 else 8))
    nodes = g.add_nodes(N)

    # t-links (terminal edges) encode unary costs.
    # IMPORTANT MAPPING:
    #  - We'll interpret SOURCE-side as label 0 (GM), SINK-side as label 1 (WM).
    #  - For PyMaxflow: add_tedges(nodes, cap_source, cap_sink)
    #    If a node ends on SOURCE side, the cut pays cap_sink
    #    If a node ends on SINK side,   the cut pays cap_source
    #  - To pay D0 when on SOURCE side and D1 when on SINK side, we pass (D1, D0).
    # g.add_tedges(nodes, D1[y_all, x_all], D0[y_all, x_all])
    for n, (yy, xx) in enumerate(zip(y_all, x_all)):
        g.add_tedge(n, float(D1[yy, xx]), float(D0[yy, xx]))

    # --- Pairwise Potts edges (n-links) ---
    U, V, Wts = build_edges_and_weights(
        tissue_mask=tissue_mask,
        feature_map=feature_map_for_pairwise,
        beta=beta,
        sigma=sigma,
        connectivity=connectivity,
        contrast_sensitive=contrast_sensitive,
        idx_map=idx_map,
    )
    # Symmetric capacities encode the Potts penalty w_ij * 1{y_i != y_j}
    for u, v, w in zip(U, V, Wts):
        g.add_edge(u, v, w, w)

    # --- Solve max-flow / min-cut (global optimum for this energy) ---
    _ = g.maxflow()

    # Read back the partition: 0=SOURCE set (GM), 1=SINK set (WM)
    labels_binary = np.fromiter((g.get_segment(n) for n in range(N)), dtype=np.uint8, count=N)

    # Map to an image-sized label map: {0=outside, 1=GM, 2=WM}
    label_map = np.zeros_like(tissue_mask, dtype=np.uint8)
    label_map[y_all, x_all] = labels_binary + 1

    # --- Energy (explicit check; useful for debugging/tuning) ---
    unary_energy = float(
        np.sum(D0[y_all, x_all] * (labels_binary == 0)) +
        np.sum(D1[y_all, x_all] * (labels_binary == 1))
    )
    pairwise_energy = 0.0
    for u, v, w in zip(U, V, Wts):
        if labels_binary[u] != labels_binary[v]:
            pairwise_energy += w
    energy = unary_energy + float(pairwise_energy)

    # --- Flips vs initial labels (optional diagnostic) ---
    n_flipped = None
    frac_flipped = None
    flip_map = None
    if initial_labels_binary is not None:
        if len(initial_labels_binary) != N:
            raise ValueError("initial_labels_binary must have length equal to #tissue nodes.")
        flip_mask_nodes = (labels_binary != initial_labels_binary)
        n_flipped = int(flip_mask_nodes.sum())
        frac_flipped = n_flipped / float(N)
        if return_flip_map:
            flip_map = np.zeros_like(tissue_mask, dtype=np.uint8)
            flip_map[y_all, x_all] = flip_mask_nodes.astype(np.uint8)

    return label_map, labels_binary, energy, n_flipped, frac_flipped, flip_map

def upsample_map(map2d, target_shape, *,
                 order=1,
                 anti_aliasing=True):
    """
    Upsample any 2D array to target_shape.
    
    Parameters
    ----------
    map2d : np.ndarray
      Input 2D array (float, int, or bool).
    target_shape : tuple of int (n_rows, n_cols)
      Desired output shape.
    order : int
      Interpolation order: 0=nearest, 1=bilinear, etc.
    anti_aliasing : bool
      Whether to apply anti-alias filter (only relevant if order>0).
    
    Returns
    -------
    out : np.ndarray
      Resized array, cast back to the original dtype (bool→bool, int→int, float→float).
    """
    # always work in float
    floated = map2d.astype(float)
    resized = resize(
        floated,
        target_shape,
        order=order,
        preserve_range=True,
        anti_aliasing=anti_aliasing
    )
    # cast back
    if map2d.dtype == bool:
        return resized.astype(bool)
    else:
        return resized.astype(map2d.dtype)

def compute_bic_1vs2_no_init(X, covariance_type='full', random_state=0):
    """
    Fit 1- and 2-component GMMs without initial centroids,
    compute their BIC scores, and decide if 1 or 2 clusters is more likely.
    """
    # 1-component GMM
    gmm1 = GaussianMixture(n_components=1, covariance_type=covariance_type, random_state=random_state)
    gmm1.fit(X)
    bic1 = gmm1.bic(X)

    # 2-component GMM
    gmm2 = GaussianMixture(n_components=2, covariance_type=covariance_type, random_state=random_state)
    gmm2.fit(X)
    bic2 = gmm2.bic(X)

    # Compare — negative delta means 2 comps is better
    delta_bic = bic2 - bic1
    if delta_bic < -10:
        verdict = "likely 2 clusters"
    elif delta_bic > 10:
        verdict = "likely 1 cluster"
    else:
        verdict = "ambiguous — needs QC"

    return {
        "bic_1comp": bic1,
        "bic_2comp": bic2,
        "delta_bic_2minus1": delta_bic,
        "verdict": verdict,
        "gmm_1comp": gmm1,
        "gmm_2comp": gmm2
    }
    
def save_slide_dashboard(
    *,
    # --- images / maps ---
    tb_img,                        # thumbnail RGB (H×W×3)
    slide_name: str,
    tissue_mask_img,               # tissue mask (for display) (H×W) or (H×W×3)
    binary_mask: np.ndarray,       # boolean mask for valid pixels
    soma_density_np: np.ndarray,   # H×W
    soma_size_np: np.ndarray,      # H×W
    soma_density_tissue_inliner: np.ndarray,  # 1D array of inlier density values (for vmax)
    soma_size_tissue_inliner: np.ndarray,     # 1D array of inlier size values (for vmax)
    gm_prob_map_thumb: np.ndarray, # thumbnail-res probability map for GM (H×W)
    wm_prob_map_thumb: np.ndarray, # thumbnail-res probability map for WM (H×W)
    label_map_thumb_rgb: np.ndarray,          # colored GM/WM (H×W×3)
    label_map_thumb_cleaned_rgb: np.ndarray,  # post-processed colored GM/WM (H×W×3)
    flip_map: np.ndarray,          # H×W, 0/1 map of flips
    tissue_idx: np.ndarray,        # H×W boolean mask (same used to build CRF nodes)
    # --- features / models ---
    X_scaled_inliner: np.ndarray,  # (N_inliers, 2)
    X_scaled_full: np.ndarray,     # (N_all, 2) (same order as used for probs, if you overlay flips)
    gmm_2d: GaussianMixture,
    gm_label: int,
    wm_label: int,
    flip_mask_nodes: Optional[np.ndarray] = None,  # (N_all,) bool; aligns with X_scaled_full rows
    # --- QC inputs ---
    any_unimodal: bool = False,
    verdict_2d: str = "likely 2 clusters",
    collapse_suspected: bool = False,
    entropy_flag: bool = False,
    dip_density: float = 0.0,
    pval_density: float = 1.0,
    dip_size: float = 0.0,
    pval_size: float = 1.0,
    # --- output / style ---
    out_path: str = "slide_dashboard.png",
    axis_pad_frac: float = 0.30,
    dpi: int = 300,
) -> str:
    """
    Build the 3×4 dashboard figure and save to `out_path` (returns the path).

    Notes:
    - `flip_mask_nodes` must align with `X_scaled_full` row order if provided.
    - QC flags are summarized on the right side of the figure.
    """
    # Precompute masked heatmaps
    masked_soma_density = np.where(binary_mask, soma_density_np, np.nan)
    masked_soma_size    = np.where(binary_mask, soma_size_np, np.nan)

    # Figure + grid
    fig, axs = plt.subplots(3, 4, figsize=(20, 12), constrained_layout=False)

    # -------------------------------- [0,0] Thumbnail --------------------------------
    axs[0, 0].imshow(tb_img)
    axs[0, 0].set_title(slide_name)
    axs[0, 0].set_axis_off()

    # -------------------------------- [1,0] Tissue Mask --------------------------------
    axs[1, 0].imshow(tissue_mask_img)
    axs[1, 0].set_title("Tissue Mask")
    axs[1, 0].set_axis_off()

    # -------------------------------- [0,1] Soma Density Heatmap --------------------------------
    im00 = axs[0, 1].imshow(
        masked_soma_density, cmap='coolwarm',
        vmax=float(np.nanmax(soma_density_tissue_inliner))
    )
    axs[0, 1].set_title("Soma Density (Masked + Inlier)")
    axs[0, 1].set_axis_off()
    # Optional colorbar:
    # plt.colorbar(im00, ax=axs[0, 1], shrink=0.9)

    # -------------------------------- [1,1] Soma Size Heatmap --------------------------------
    im01 = axs[1, 1].imshow(
        masked_soma_size, cmap='coolwarm',
        vmax=float(np.nanmax(soma_size_tissue_inliner))
    )
    axs[1, 1].set_title("Soma Size (Masked + Inlier)")
    axs[1, 1].set_axis_off()
    # Optional colorbar:
    # plt.colorbar(im01, ax=axs[1, 1], shrink=0.9)

    # -------------------------------- [0,2] GM Probability Map --------------------------------
    axs[0, 2].imshow(tb_img, alpha=0.4)
    im_gm = axs[0, 2].imshow(gm_prob_map_thumb, cmap='Blues', alpha=0.6, vmin=0, vmax=1)
    axs[0, 2].set_title("GM Probability Map")
    axs[0, 2].set_axis_off()
    plt.colorbar(im_gm, ax=axs[0, 2], fraction=0.04, label='P(GM)')

    # -------------------------------- [1,2] WM Probability Map --------------------------------
    axs[1, 2].imshow(tb_img, alpha=0.4)
    im_wm = axs[1, 2].imshow(wm_prob_map_thumb, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
    axs[1, 2].set_title("WM Probability Map")
    axs[1, 2].set_axis_off()
    plt.colorbar(im_wm, ax=axs[1, 2], fraction=0.04, label='P(WM)')

    # -------------------------------- [2,0] & [2,1] 1D Hist + 2-comp GMM --------------------------------
    features = [X_scaled_inliner[:, 0], X_scaled_inliner[:, 1]]
    feature_names = ['soma_density', 'soma_size']
    plot_names = ['Soma Density', 'Soma Size']
    colors = ['red', 'blue']
    linestyles = ['dotted', 'dotted']
    gmm_results = {}

    for i, (data, feature_key, name) in enumerate(zip(features, feature_names, plot_names)):
        # Histogram
        axs[2, i].hist(
            data, bins=50, color='gray', alpha=0.7, edgecolor='k', density=True, label='Histogram'
        )
        # Fit 1D GMM
        data_reshaped = np.asarray(data).reshape(-1, 1)
        gmm_1d = GaussianMixture(n_components=2, random_state=0).fit(data_reshaped)
        x = np.linspace(float(np.min(data)), float(np.max(data)), 1000)
        gmm_density = np.exp(gmm_1d.score_samples(x.reshape(-1, 1)))
        axs[2, i].plot(x, gmm_density, color='black', lw=2, label='GMM Total')

        # Per-component PDFs
        comp_stats = []
        for comp in range(2):
            mean = float(gmm_1d.means_[comp, 0])
            std = float(np.sqrt(gmm_1d.covariances_[comp, 0, 0]))  # ok for 'full' in 1D
            weight = float(gmm_1d.weights_[comp])
            pdf = weight * norm.pdf(x, mean, std)
            axs[2, i].plot(x, pdf, color=colors[comp], lw=2, linestyle=linestyles[comp],
                           label=f'Component {comp+1}')
            comp_stats.append((mean, std))
        gmm_results[feature_key] = comp_stats

        axs[2, i].set_title(f'{name} Histogram (Scaled + Inlier)')
        axs[2, i].set_xlabel(name)
        axs[2, i].set_ylabel('Density')
        axs[2, i].legend()

    # -------------------------------- [2,2] 2D GMM scatter + ellipses + flips --------------------------------
    ax = axs[2, 2]
    point_colors   = {gm_label: '#A0C4FF', wm_label: '#FFADAD'}   # pastel
    contour_colors = {gm_label: '#1f77b4', wm_label: '#d62728'}   # darker
    labels_text    = {gm_label: 'GM', wm_label: 'WM'}

    z = gmm_2d.predict(X_scaled_full)

    # pastel scatter by cluster
    for k in [gm_label, wm_label]:
        mask = (z == k)
        if np.any(mask):
            ax.scatter(
                X_scaled_full[mask, 0], X_scaled_full[mask, 1],
                s=6, alpha=0.7, c=point_colors[k], edgecolors='none',
                label=f'{labels_text[k]} points', zorder=1
            )

    # means + ellipses
    for k in [gm_label, wm_label]:
        mean = gmm_2d.means_[k]
        cov  = gmm_2d.covariances_[k]
        if cov.ndim == 1:
            cov = np.diag(cov)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

        ax.scatter(mean[0], mean[1], marker='x', color=contour_colors[k],
                   s=120, linewidths=3, label=f'{labels_text[k]} mean', zorder=3)

        for nsig in range(1, 5):
            width, height = 2 * nsig * np.sqrt(vals[0]), 2 * nsig * np.sqrt(vals[1])
            ell = Ellipse(
                xy=mean, width=width, height=height, angle=angle,
                edgecolor=contour_colors[k], facecolor='none',
                linewidth=2.2,
                alpha=0.9 if nsig == 1 else 0.6 if nsig == 2 else 0.45 if nsig == 3 else 0.35,
                label=f'{labels_text[k]}: {nsig}$\\sigma$' if nsig == 1 else None,
                zorder=2
            )
            ax.add_patch(ell)

    # flips overlay (lime)
    if flip_mask_nodes is not None:
        flip_mask = np.asarray(flip_mask_nodes).astype(bool)
        if flip_mask.shape[0] == X_scaled_full.shape[0] and np.any(flip_mask):
            ax.scatter(
                X_scaled_full[flip_mask, 0], X_scaled_full[flip_mask, 1],
                s=6, c='#7FFF00', edgecolors='none', alpha=1.0,
                label='Flipped (CRF ≠ GMM)', zorder=5
            )

    # axis limits from inliers
    X_for_limits = X_scaled_inliner if X_scaled_inliner is not None else X_scaled_full
    x_min, x_max = np.min(X_for_limits[:, 0]), np.max(X_for_limits[:, 0])
    y_min, y_max = np.min(X_for_limits[:, 1]), np.max(X_for_limits[:, 1])
    dx, dy = x_max - x_min, y_max - y_min
    pad_x = max(dx * axis_pad_frac, 1e-6)
    pad_y = max(dy * axis_pad_frac, 1e-6)
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_xlabel('Soma Density (scaled)')
    ax.set_ylabel('Soma Size (scaled)')
    ax.set_title('GMM Clusters (CRF Flipped)')
    # de-dup legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    # -------------------------------- [0,3] CRF GM/WM (raw) --------------------------------
    axs[0, 3].imshow(label_map_thumb_rgb)
    axs[0, 3].set_title('GMM Seg (CRF) [GM: Blue / WM: Red]')
    axs[0, 3].set_axis_off()

    # -------------------------------- [1,3] CRF GM/WM (cleaned) --------------------------------
    axs[1, 3].imshow(label_map_thumb_cleaned_rgb)
    axs[1, 3].set_title('GMM Seg (CRF + Cleaned) (Thumbnail-Res)')
    axs[1, 3].set_axis_off()

    # -------------------------------- [2,3] CRF flip overlay --------------------------------
    lime_rgba = to_rgba('lime', alpha=0.85)
    cmap_flips = ListedColormap([(0, 0, 0, 0.0), lime_rgba])

    # background
    axs[2, 3].imshow(soma_density_np, cmap='gray')  # grayscale background
    axs[2, 3].imshow(flip_map.astype(int), cmap=cmap_flips, interpolation='nearest')
    axs[2, 3].legend(handles=[Patch(facecolor=lime_rgba, edgecolor='none', label='Flipped')],
                     loc='lower right')
    axs[2, 3].set_title("CRF: flipped pixels (Soma Density)")
    axs[2, 3].set_axis_off()

    # -------------------------------- QC text panel on the right --------------------------------
    # 1) Domain-knowledge alignment (GM: larger size & lower density)
    gmm_means = gmm_2d.means_
    domain_pass = (
        (gmm_means[0,1] > gmm_means[1,1] and gmm_means[0,0] < gmm_means[1,0]) or
        (gmm_means[0,1] < gmm_means[1,1] and gmm_means[0,0] > gmm_means[1,0])
    )
    domain_msg = "Aligns with Domain Knowledge: ✔ PASS " if domain_pass else "The GMM clusters do not align with Domain Knowledge: ✘ FAIL"

    # 2–5) Other QC tests → convert to PASS=True / FAIL=False
    qc_pass = {
        "Domain knowledge alignment": domain_pass,                                  # (1)
        "Multimodality in each feature": not bool(any_unimodal),                   # (2)
        "Class separability (2D)": (verdict_2d == "likely 2 clusters"),            # (3)
        "Class imbalance": not bool(collapse_suspected),                           # (4)
        "Model fit diagnostics": not bool(entropy_flag),                           # (5)
    }

    n_tests  = len(qc_pass)              # -> 5
    n_passed = sum(qc_pass.values())     # count of True
    n_failed = n_tests - n_passed
    slide_qc_fail = (n_failed > 0)

    # Reserve right margin for text
    fig.tight_layout(rect=[0, 0, 0.86, 1])

    # Header + summary
    x_pos = 0.86
    qc_y  = 0.95
    fig.text(x_pos, qc_y,
            f"[QC summary: {n_passed}/{n_tests} passed, {n_failed}/{n_tests} failed.]",
            fontsize=10, fontweight='bold', va='top', ha='left')

    # Details
    qc_y -= 0.05
    fig.text(x_pos, qc_y, "Details:",
            fontsize=10, fontweight='bold', va='top', ha='left')

    max_name = max(len(name) for name in qc_pass)
    for name, passed in qc_pass.items():
        qc_y -= 0.04
        mark  = "✔" if passed else "✘"
        label = f" - {name:<{max_name}} : {mark} {'PASS' if passed else 'FAIL'}"
        fig.text(
            x_pos, qc_y, label,
            family='monospace',            # <-- key change
            color=('green' if passed else 'red'),
            fontsize=9, fontweight='bold', va='top', ha='left'
        )

    # Final decision
    qc_y -= 0.06

    # Decide outcome from qc_pass dict you built above (5 tests)
    all_pass = all(qc_pass.values())
    only_unimodal_failed = (not qc_pass["Multimodality in each feature"]) and all(
        v for k, v in qc_pass.items() if k != "Multimodality in each feature"
    )

    save_name_prefix = ''
    if all_pass:
        final_msg = "FINAL QC: ✔ PASS"
        final_color = "green"
        save_name_prefix = '[PASSED]_'
    elif only_unimodal_failed:
        # Triangle warning; keep it short and specific
        final_msg = "FINAL QC: ▲ WARNING — unimodality detected"
        final_color = "#E69F00"  # orange
        save_name_prefix = '[WARNING]_'
    else:
        final_msg = "FINAL QC: ✘ FAIL (review needed)"
        final_color = "red"
        save_name_prefix = '[REVIEW]_'

    fig.text(
        x_pos, qc_y, final_msg,
        color=final_color, fontsize=10, fontweight='bold',
        va='top', ha='left'
    )

    # Stats
    qc_y -= 0.06
    fig.text(
        x_pos, qc_y,
        f"Soma Density: dip={dip_density:.4f}, p-value={pval_density:.4f}\n"
        f"Soma Size:    dip={dip_size:.4f}, p-value={pval_size:.4f}",
        color='black', fontsize=10, fontweight='bold', va='top', ha='left'
    )

    # -------------------------------- Save & close --------------------------------
    os.makedirs(out_path, exist_ok=True)
    fig.savefig(os.path.join(out_path, save_name_prefix + slide_name + '.png'), dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path

def render_neuseg_contours(
    tb_img: np.ndarray,
    gm_mask_thumb_clean: np.ndarray,
    wm_mask_thumb_clean: np.ndarray,
    tissue_bool: np.ndarray,
    line_radius_px: int = 3,
    show_fills: bool = True,
    title: str = None,
    save_path: str = None,
    dpi: int = 300,
    legend: bool = True,
):
    """
    Build contour labels & overlay for NEUSEG outputs and optionally visualize.

    Parameters
    ----------
    tb_img : (H, W, 3) or (H, W, 4) uint8
        Base thumbnail image to render under the overlay.
    gm_mask_thumb_clean, wm_mask_thumb_clean, tissue_bool : (H, W) arrays
        Binary-like masks (0/1 or bool). 'tissue_bool' defines tissue support.
    line_radius_px : int
        Radius for contour thickening (final stroke thickness ~ 2*R+1).
    show_fills : bool
        If True, render very light GM/WM region fills under the contours.
    title : str
        Optional title for the figure.
    save_path : str
        If provided, saves the rendered figure to this path.
    dpi : int
        DPI for saving the figure.
    legend : bool
        If True, attach a small legend for boundary colors (and fills if shown).

    Returns
    -------
    result : dict
        {
          'overlay_rgba' : (H, W, 4) uint8,    # the rendered contour overlay only
          'lbl'          : (H, W) uint8,       # 0=None, 1=GM–WM, 2=GM–BG, 3=WM–BG (1-px before thickening)
          'lbl_thick'    : (H, W) uint8,       # thickened labels with priority
          'masks'        : {                   # boolean masks
              'gm': (H, W) bool,
              'wm': (H, W) bool,
              'bg': (H, W) bool,
              'gm_wm_boundary': (H, W) bool,
              'gm_bg_boundary': (H, W) bool,
              'wm_bg_boundary': (H, W) bool,
          },
          'colors'       : { 'GM_WM': (r,g,b), 'GM_BG': (r,g,b), 'WM_BG': (r,g,b) }  # 0..255
        }
    """
    # ============================ #
    # 0) Ensure exclusivity/support
    # ============================ #
    gm = (gm_mask_thumb_clean.astype(bool)) & tissue_bool
    wm = (wm_mask_thumb_clean.astype(bool)) & tissue_bool
    bg = (~tissue_bool)
    wm = wm & (~gm)  # enforce exclusivity

    H, W = tissue_bool.shape

    # ============================ #
    # 1) Neighborhoods / thickness
    # ============================ #
    SE = np.ones((3, 3), dtype=np.uint8)  # 1-px boundary detect (8-connected)
    SE_THICK = np.ones((2*line_radius_px + 1, 2*line_radius_px + 1), dtype=np.uint8)

    # ================================ #
    # 2) Disjoint 1-px boundary maps
    # ================================ #
    gm_wm_boundary = gm & binary_dilation(wm, structure=SE)
    gm_bg_boundary = gm & binary_dilation(bg, structure=SE)
    wm_bg_boundary = wm & binary_dilation(bg, structure=SE)

    # Priority: GM–WM over GM–BG to keep sets disjoint
    gm_bg_boundary = gm_bg_boundary & (~gm_wm_boundary)

    # ================================ #
    # 3) Encode labels (no overlaps)
    # ================================ #
    lbl = np.zeros((H, W), dtype=np.uint8)   # 0=None, 1=GM–WM, 2=GM–BG, 3=WM–BG
    lbl[gm_wm_boundary] = 1
    lbl[(lbl == 0) & gm_bg_boundary] = 2
    lbl[(lbl == 0) & wm_bg_boundary] = 3

    # ======================================= #
    # 4) Priority-aware thickening of lines
    # ======================================= #
    lbl_thick = np.zeros_like(lbl)
    for k in (1, 2, 3):  # keep priority
        expanded = binary_dilation(lbl == k, structure=SE_THICK)
        lbl_thick[(lbl_thick == 0) & expanded] = k

    # ====================== #
    # 5) Colorized overlay
    # ====================== #
    overlay = np.zeros((H, W, 4), dtype=np.uint8)

    # --- REQUIRED COLORS (boundary lines) ---
    COL_GM_WM = np.array([  0, 200,   0], dtype=np.uint8)  # Green
    COL_GM_BG = np.array([220,   0, 255], dtype=np.uint8)  # Vivid magenta
    COL_WM_BG = np.array([  0,   0,   0], dtype=np.uint8)  # Black
    ALPHA = 250

    overlay[lbl_thick == 1, :3] = COL_GM_WM
    overlay[lbl_thick == 1,  3] = ALPHA
    overlay[lbl_thick == 2, :3] = COL_GM_BG
    overlay[lbl_thick == 2,  3] = ALPHA
    overlay[lbl_thick == 3, :3] = COL_WM_BG
    overlay[lbl_thick == 3,  3] = ALPHA

    # ======================================= #
    # 6) soft GM/WM region fills
    # ======================================= #
    GM_FILL_RGBA = (0.85, 0.70, 0.90, 0.25)   # very light purple (0..1)
    WM_FILL_RGBA = (0.75, 0.95, 0.75, 0.25)   # very light green  (0..1)

    gm_fill = None
    wm_fill = None
    if show_fills:
        gm_fill = np.zeros((H, W, 4), dtype=float)
        wm_fill = np.zeros((H, W, 4), dtype=float)
        gm_fill[gm] = GM_FILL_RGBA
        wm_fill[wm] = WM_FILL_RGBA

    # ======================================= #
    # 7) Visualize (if tb_img is provided)
    # ======================================= #
    if tb_img is not None:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(tb_img)
        if show_fills:
            ax.imshow(gm_fill)
            ax.imshow(wm_fill)
        ax.imshow(overlay)
        if title:
            ax.set_title(title, fontsize=11)
        ax.axis("off")

        if legend:
            handles = [
                mpatches.Patch(color=COL_GM_WM/255.0,  label="GM–WM"),
                mpatches.Patch(color=COL_GM_BG/255.0,  label="GM–BG"),
                mpatches.Patch(color=COL_WM_BG/255.0,  label="WM–BG"),
            ]
            if show_fills:
                handles += [
                    mpatches.Patch(color=GM_FILL_RGBA[:3], alpha=GM_FILL_RGBA[3], label="GM (fill)"),
                    mpatches.Patch(color=WM_FILL_RGBA[:3], alpha=WM_FILL_RGBA[3], label="WM (fill)"),
                ]
            ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=8)

        plt.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        else:
            plt.show()
        plt.close(fig)

    # ======================================= #
    # 8) Return data for saving/analysis
    # ======================================= #
    result = {
        'overlay_rgba': overlay,        # contours RGBA (no base image)
        'lbl': lbl,                     # 1-px label map before thickening
        'lbl_thick': lbl_thick,         # thickened label map
        'masks': {
            'gm': gm,
            'wm': wm,
            'bg': bg,
            'gm_wm_boundary': gm_wm_boundary,
            'gm_bg_boundary': gm_bg_boundary,
            'wm_bg_boundary': wm_bg_boundary,
        },
        'colors': {
            'GM_WM': tuple(COL_GM_WM.tolist()),
            'GM_BG': tuple(COL_GM_BG.tolist()),
            'WM_BG': tuple(COL_WM_BG.tolist()),
        },
    }
    return result

################################################################## Main ##################################################################
debug = True

def main(argv=None) -> int:
    # %% Argument Parser   
    parser = argparse.ArgumentParser(description="NEUSEG Pipeline (GMM) for one slide")
    parser.add_argument("slide", help="Path to input .svs")
    parser.add_argument("out_root", help="Output root directory")
    parser.add_argument("debug_path", help="Directory to save summary slide")
    args = parser.parse_args(argv)
    
    # %% Directory Set-up    
    
    slide_name = os.path.splitext(os.path.basename(args.slide))[0]
    odir = os.path.join(args.out_root, slide_name)
    
    # %% Loading Processed Features
    tb_path = os.path.join(odir, 'thumbnail.png')
    tissue_mask_path = os.path.join(odir, 'tissue_mask.npy')
    feature_heatmap_path = os.path.join(odir, 'feature_heatmap.npy')

    required = [
        ("thumbnail.png", tb_path),
        ("tissue_mask.npy", tissue_mask_path),
        ("feature_heatmap.npy", feature_heatmap_path),
    ]
    missing = [name for name, path in required if not os.path.exists(path)]

    # ---- Guard clause: exit early if anything is missing ----
    if missing:
        print(f"[ERROR] Missing required files in {odir}: {', '.join(missing)}", file=sys.stderr)
        return 1  # non-zero => failure 
    
    tb = sana.image.Frame(tb_path)
    tissue_mask = sana.image.Frame(np.load(tissue_mask_path))
    features_np = np.load(feature_heatmap_path)
    soma_density_np, soma_size_np = features_np[:, :, 0], features_np[:, :, 1]
    
    # %% GM/WM Segmentation Block
    # %% ---------- [1] Downsample Tissue Mask to Match Heatmap Shape ----------
    tissue_mask_thumb = tissue_mask.img.squeeze() 
    # Downsample
    tissue_mask_resized = resize(
        tissue_mask_thumb,
        soma_density_np.shape,    # (n_rows, n_cols)
        order=0,                  # 0 = nearest-neighbor
        preserve_range=True,      # keep original 0/255 values
        anti_aliasing=False       # no blur for discrete labels
    ).astype(tissue_mask_thumb.dtype)
    assert tissue_mask_resized.shape == soma_density_np.shape
    
    # %% ---------- [2] Mask Out Non-Tissue Regions in Features ----------
    tissue_idx = tissue_mask_resized == 1
    soma_density_tissue = soma_density_np[tissue_idx]
    soma_size_tissue = soma_size_np[tissue_idx]
    
    # %% ---------- [3] Remove Outliers to Fit 1D GMM (for each feature) ----------
    # Start from your tissue‐only arrays (unchanged) 
    d0 = soma_density_tissue.copy()
    s0 = soma_size_tissue.copy()

    #  Mask step 1: drop points where BOTH density & size == 0 
    nonzero_mask = ~((d0 == 0) & (s0 == 0))
    num_zero_dropped = np.count_nonzero(~nonzero_mask)

    #  Non-Zero Data 
    d1 = d0[nonzero_mask]
    s1 = s0[nonzero_mask]

    # Compute IQR (factor = 1.5) mask on the survivors of step 1 
    iqr_mask = mask_iqr(d1) & mask_iqr(s1)
    num_iqr_dropped = nonzero_mask.sum() - iqr_mask.sum()

    #  Build the combined inlier mask over the ORIGINAL arrays.
    inlier_mask = np.zeros_like(nonzero_mask, dtype=bool)
    # only zero_mask==True indices can be inliers; within those, pick iqr_mask
    inlier_mask[nonzero_mask] = iqr_mask

    # Extract your final cleaned arrays
    soma_density_tissue_inliner, soma_size_tissue_inliner = d0[inlier_mask], s0[inlier_mask]

    # Print final summary with percentages
    total   = len(d0)
    kept    = inlier_mask.sum()
    dropped = total - kept
    pct     = dropped / total * 100
    print(
        f"Final count: {kept} inliers out of {total} original tissue points "
        f"({100-pct:.2f}% Preserved / {pct:.2f}% Dropped)"
    )
    
    # %% ---------- [4] Feature Standardization ----------
    # Full Tissue Dataset
    X = np.stack([soma_density_tissue, soma_size_tissue], axis=1)

    # Tissue Dataset (Outlier removed)
    X_inliner = np.stack([soma_density_tissue_inliner, soma_size_tissue_inliner], axis=1)
    scaler_inliner = StandardScaler()
    X_scaled_inliner = scaler_inliner.fit_transform(X_inliner)
    
    # %% ---------- [5] 1D GMM Model (n=2 components) for each feature: Soma Density & Size ----------
    gmm_results = get_soma_feature_1d_gmm_stats(X_scaled_inliner[:, 0], X_scaled_inliner[:, 1])
    
    # Get means and stds for density and size from gmm_results
    density_means = [x[0] for x in gmm_results['soma_density']]
    density_stds = [x[1] for x in gmm_results['soma_density']]
    size_means = [x[0] for x in gmm_results['soma_size']]
    size_stds = [x[1] for x in gmm_results['soma_size']]

    # For density: GM = lower mean, WM = higher mean
    if density_means[0] < density_means[1]:
        density_GM_mean, density_GM_std = density_means[0], density_stds[0]
        density_WM_mean, density_WM_std = density_means[1], density_stds[1]
    else:
        density_GM_mean, density_GM_std = density_means[1], density_stds[1]
        density_WM_mean, density_WM_std = density_means[0], density_stds[0]

    # For size: GM = bigger mean, WM = smaller mean
    if size_means[0] > size_means[1]:
        size_GM_mean, size_GM_std = size_means[0], size_stds[0]
        size_WM_mean, size_WM_std = size_means[1], size_stds[1]
    else:
        size_GM_mean, size_GM_std = size_means[1], size_stds[1]
        size_WM_mean, size_WM_std = size_means[0], size_stds[0]
        
    # %% ---------- [6] 2D GMM Model (n=2 components) for both features: Soma Density & Size (Full Tissue Dataset) ----------
    # Fit with or without centroid initialization
    use_init_centroids = False  # <----- Switch this as needed (False: not using 1D GMM initialization)
    gmm_2d, _, _ = fit_2d_gmm(
        X_scaled_inliner,
        density_GM_mean, size_GM_mean,
        density_WM_mean, size_WM_mean,
        use_init_centroids=use_init_centroids,
        random_state=0
    )

    # After fitting gmm_2d, determine which label is GM and which is WM
    gmm_means = gmm_2d.means_
    # Cluster with higher size and lower density = GM
    if (gmm_means[0,1] > gmm_means[1,1]) and (gmm_means[0,0] < gmm_means[1,0]): # [Soma Size] Cluster 0 > Cluster 1 AND [Soma Density] Cluster 0 < Cluster 1 ✅ PASS Aligns with Domain Knowledge
        gm_label, wm_label = 0, 1
    elif (gmm_means[0,1] < gmm_means[1,1]) and (gmm_means[0,0] > gmm_means[1,0]): # [Soma Size] Cluster 0 < Cluster 1 AND [Soma Density] Cluster 0 > Cluster 1 "✅ PASS Aligns with Domain Knowledge
        gm_label, wm_label = 1, 0
    else: # ❌ The GMM clusters do not align with Domain Knowledge
        # Check the mean difference
        size_diff = abs(gmm_means[0, 1] - gmm_means[1, 1])
        den_diff = abs(gmm_means[0, 0] - gmm_means[1, 0])
        # Follow the logic where there is greater diff
        if size_diff >= den_diff:
            if (gmm_means[0,1] < gmm_means[1,1]):   # [Soma Size] Cluster 0 < Cluster 1
                gm_label, wm_label = 1, 0
            else:                                   # [Soma Size] Cluster 0 > Cluster 1
                gm_label, wm_label = 0, 1
        else:
            if (gmm_means[0, 0] < gmm_means[1, 0]): # [Soma Density] Cluster 0 < Cluster 1
                gm_label, wm_label = 0, 1
            else:                                   # [Soma Density] Cluster 0 > Cluster 1
                gm_label, wm_label = 1, 0
        
    # Transform full dataset using the same scaler
    X_scaled_full = scaler_inliner.transform(X)

    # 1. Predict hard labels (0 or 1)
    labels = gmm_2d.predict(X_scaled_full)          # shape (N,)
    # 2. Predict posterior probabilities for each component
    probs  = gmm_2d.predict_proba(X_scaled_full)    # shape (N, 2)

    # %% ---------- [7] Conditional Random Field (CRF) + Constant Pott's Model ----------
    # Find the (y,x) coords of every tissue pixel
    y_all, x_all = np.where(tissue_idx)

    # --- Build low-res probability maps for the entire image ---
    gm_prob_map = np.zeros_like(soma_density_np, dtype=float)
    wm_prob_map = np.zeros_like(soma_density_np, dtype=float)

    # Place only the inlier probabilities back into the full grid
    gm_prob_map[y_all, x_all] = probs[:, gm_label]
    wm_prob_map[y_all, x_all] = probs[:, wm_label]
    
    # Initial labels for "flip" accounting
    initial_labels_binary = np.argmax(
        np.stack([probs[:, gm_label], probs[:, wm_label]], axis=1),
        axis=1
    ).astype(np.uint8)  # {0=GM, 1=WM}

    # Run CRF (constant Potts; 8-connected)
    def suggest_beta(connectivity=8, p_target=0.60, m_attn=1.0, k=None):
        if k is None:
            k = 1.0 if connectivity==4 else 1.5
        deltaU = abs(np.log((1.0 - p_target) / p_target))
        return float(deltaU / (k * max(m_attn, 1e-9)))
    beta_const_8 = suggest_beta(connectivity=8, p_target=0.75, m_attn=1.0) 

    label_map, labels_vec, energy, n_flipped, frac_flipped, flip_map = crf_potts_graphcut(
        gm_prob_map=gm_prob_map,
        wm_prob_map=wm_prob_map,
        tissue_mask=tissue_idx,                  # HxW boolean
        feature_map_for_pairwise=None,           # not needed for constant Potts
        beta=beta_const_8,                               # smoothness strength (tune)
        connectivity=8,                          # 4 or 8; 8 is more isotropic
        contrast_sensitive=False,                # True if using a feature_map
        eps=1e-6,
        initial_labels_binary=initial_labels_binary,
        return_flip_map=True,
    )

    # Masks
    gm_mask = (label_map == 1)
    wm_mask = (label_map == 2)

    print("[CRF]")
    print(f"# tissue nodes: {len(initial_labels_binary)}")
    print("GM pixels:", int(gm_mask.sum()))
    print("WM pixels:", int(wm_mask.sum()))
    print(f"# flipped: {n_flipped}  ({(frac_flipped or 0)*100:.2f}%)")
    
    # Upsample to match the Thumbnail Resolution
    target_shape = tb.img.shape[:2]  # (height, width)
    
    # hard labels & masks (nearest‐neighbor, no AA)
    label_map_thumb   = upsample_map(label_map,   target_shape, order=0, anti_aliasing=False)

    def upsample_mask_sdt(mask_bool, target_shape, order=1, sigma_px=0.0):
        """Upsample a binary mask via signed distance, then threshold at 0."""
        m = mask_bool.astype(bool)
        # signed distance: +inside, -outside (0 on boundary)
        d_in  = distance_transform_edt(m)
        d_out = distance_transform_edt(~m)
        sdt = d_in - d_out   # float, smooth field

        # bilinear (order=1) resize of continuous field
        sdt_up = resize(sdt, target_shape, order=order, mode='reflect',
                        anti_aliasing=True, preserve_range=True)
        if sigma_px and sigma_px > 0:
            sdt_up = gaussian_filter(sdt_up, sigma=sigma_px)

        return (sdt_up > 0)

    # drop-in replacement for the NN upsample:
    gm_mask_thumb = upsample_mask_sdt(gm_mask, target_shape, order=1, sigma_px=6.0)
    wm_mask_thumb = upsample_mask_sdt(wm_mask, target_shape, order=1, sigma_px=6.0)
    # ********************************

    # probability maps (bilinear, with AA)
    gm_prob_map_thumb = upsample_map(gm_prob_map, target_shape, order=1, anti_aliasing=True)
    wm_prob_map_thumb = upsample_map(wm_prob_map, target_shape, order=1, anti_aliasing=True)

    # --- Upsample continuous maps (order=1, anti_aliasing=True) ---
    # put the tissue-only vector back into a 2D array
    soma_density_low = np.full_like(soma_density_np, np.nan, dtype=float)
    soma_density_low[tissue_idx] = soma_density_tissue

    soma_size_low = np.full_like(soma_size_np, np.nan, dtype=float)
    soma_size_low[tissue_idx] = soma_size_tissue

    # --- Upsample continuous maps (order=1, anti_aliasing=True) ---
    soma_density_thumb = upsample_map(
        soma_density_low,
        target_shape,
        order=1,
        anti_aliasing=True
    )

    soma_size_thumb = upsample_map(
        soma_size_low,
        target_shape,
        order=1,
        anti_aliasing=True
    )

    # %% ---------- [8] Post-processing: OUTER-RING  →  ISLAND PRUNING  ----------
    # ======= OUTER-RING relabeling  →  ISLAND pruning (percent-of-tissue) ========
    EDGE_BAND_UM = 400.0        # depth from outer tissue edge to correct (µm)
    ISLAND_FRAC  = 0.0025       # Islands smaller than 0.25% flipepd
    tissue_bool = (np.squeeze(tissue_mask.img) == 1)

    # µm/px at thumbnail
    logger = sana.logging.Logger('normal', os.path.join(odir, 'parameters.pkl'))
    loader = sana.slide.Loader(logger, args.slide)
    H_th, W_th = tb.img.shape[:2]
    W0, H0 = loader.level_dimensions[0]
    downsample_thumb = W0 / float(W_th)
    thumb_mpp = float(loader.mpp) * downsample_thumb

    # Convert microns → pixels for outer ring
    edge_band_px = max(1, int(round(EDGE_BAND_UM / thumb_mpp)))

    # ---------- OUTER-RING relabeling (nearest-core propagation) ----------
    tissue_outer = tissue_bool
    dist_outer_px = distance_transform_edt(tissue_outer.astype(np.uint8))
    ring_mask = tissue_bool & (dist_outer_px > 0) & (dist_outer_px <= edge_band_px)
    core_mask = tissue_bool & (dist_outer_px > edge_band_px)

    # Nearest *core* pixel indices (EDT returns indices to nearest zero → invert core)
    comp_core = ~core_mask
    _, (iy, ix) = distance_transform_edt(comp_core, return_indices=True)

    gm_core_label = gm_mask_thumb[iy, ix]
    wm_core_label = wm_mask_thumb[iy, ix]

    gm_prop = gm_mask_thumb.copy()
    wm_prop = wm_mask_thumb.copy()
    gm_prop[ring_mask] = gm_core_label[ring_mask]
    wm_prop[ring_mask] = wm_core_label[ring_mask]

    # Enforce complementarity inside tissue BEFORE island pruning
    wm_mask_post_ring = (wm_prop & tissue_bool)
    gm_mask_post_ring = (tissue_bool & (~wm_mask_post_ring))

    # ---------- ISLAND PRUNING (size threshold as % of tissue) ----------
    tissue_area_px = int(tissue_bool.sum())
    min_island_area_px = max(1, int(round(ISLAND_FRAC * tissue_area_px)))
    print(f"[islands] threshold = {ISLAND_FRAC*100:.3f}% of tissue  "
        f"-> {min_island_area_px} px (tissue={tissue_area_px} px)")

    def prune_islands(mask_bool, support_bool, min_area_px):
        """
        Remove connected components smaller than min_area_px (within support_bool).
        Returns: (new_mask, num_islands_removed, pixels_removed)
        """
        cc, ncc = label(mask_bool & support_bool)
        if ncc == 0:
            return mask_bool, 0, 0
        counts = np.bincount(cc.ravel())
        small_ids = np.where(counts < min_area_px)[0]
        small_ids = small_ids[small_ids != 0]  # drop background
        if small_ids.size == 0:
            return mask_bool, 0, 0
        drop = np.isin(cc, small_ids)
        removed_px = int(counts[small_ids].sum())
        out = mask_bool.copy()
        out[drop] = False
        return out, int(small_ids.size), removed_px

    # (A) prune WM islands, recompute GM as complement
    wm_mask_pruned, wm_n_islands, wm_removed_px = prune_islands(
        wm_mask_post_ring, tissue_bool, min_island_area_px
    )
    gm_mask_pruned = tissue_bool & (~wm_mask_pruned)

    # (B) prune GM islands, recompute WM as complement
    gm_mask_pruned, gm_n_islands, gm_removed_px = prune_islands(
        gm_mask_pruned, tissue_bool, min_island_area_px
    )
    wm_mask_pruned = tissue_bool & (~gm_mask_pruned)

    # ---------- Final exclusivity + tissue enforcement ----------
    # gm_mask_thumb_clean = (gm_mask_pruned & (~wm_mask_pruned)) & tissue_bool
    wm_mask_thumb_clean = (wm_mask_pruned & (~gm_mask_pruned)) & tissue_bool
    gm_mask_thumb_clean = tissue_bool & (~wm_mask_thumb_clean)

    # ---------- Report -----------------------------------------------------------
    tot_removed_px = wm_removed_px + gm_removed_px
    print(f"[islands] removed WM islands: {wm_n_islands}  ({wm_removed_px} px, "
        f"{100*wm_removed_px/max(1,tissue_area_px):.4f}% of tissue)")
    print(f"[islands] removed GM islands: {gm_n_islands}  ({gm_removed_px} px, "
        f"{100*gm_removed_px/max(1,tissue_area_px):.4f}% of tissue)")
    print(f"[islands] total removed: {tot_removed_px} px "
        f"({100*tot_removed_px/max(1,tissue_area_px):.4f}% of tissue)")
    # ============================================================================

    # %% ---------- [9] QC ----------
    # 1) [Test for Multimodality in Each Feature] Hartigan’s Dip Test for both features
    # Run Hartigan's Dip Test for each feature
    dip_density, pval_density = diptest(soma_density_tissue_inliner)
    dip_size, pval_size = diptest(soma_size_tissue_inliner)
    # Flag if even one feature is unimodal (p > 0.05)
    any_unimodal = (pval_density > 0.05) or (pval_size > 0.05)
    
    # 2) [Test for Class Separability in the Joint (2D) Space] Bayesian Information Criterion (BIC) Testing
    res = compute_bic_1vs2_no_init(X_scaled_inliner, covariance_type='full', random_state=0)

    # 3) [Detect Class Imbalance] Post-GMM Weight Check + effective count version
    # Basic stats from the fit
    n_samples = int(X_scaled_inliner.shape[0])
    weights = np.asarray(gmm_2d.weights_, dtype=float)
    min_weight = float(np.min(weights))             # smallest mixture weight π_min
    min_eff_count = float(n_samples * min_weight)   # expected #points in the smallest component
    # Proportion (imbalance) threshold
    #    - Base floor at 5% of total weight
    #    - Also require > 5/n to avoid false flags when n is tiny
    min_weight_floor = max(0.05, 5.0 / max(1, n_samples))
    # Effective-count (stability) threshold
    #    - Require ~1% of the slide to belong to the smallest component
    #    - Clamp to [50, 200] to reflect 2D covariance stability needs (noise-dependent)
    min_eff_count_thresh = max(50, min(200, int(0.01 * n_samples)))
    # Evaluate flags separately
    imbalance_flag = (min_weight < min_weight_floor)             # tiny fraction (proportion issue)
    stability_flag = (min_eff_count < min_eff_count_thresh)      # too few absolute samples (collapse risk)
    # Final verdict: flag if either problem is present
    collapse_suspected = bool(imbalance_flag or stability_flag)
    
    # 4) [Model Fit Diagnostics] Posterior Entropy Test (Median > 0.5)
    # Responsibilities (posterior probabilities) for each point
    R = gmm_2d.predict_proba(X_scaled_inliner)   # shape: (n_samples, K)
    K = R.shape[1]

    # Safety to avoid log(0)
    R_safe = np.clip(R, 1e-12, 1.0)

    # Entropy per sample (nats), then normalize to [0,1] by dividing by log(K)
    H = -np.sum(R_safe * np.log(R_safe), axis=1)        # (n_samples,)
    H_norm = H / np.log(K)                               # normalized entropy in [0,1]

    # Summary stats
    median_entropy = float(np.median(H_norm))
    mean_entropy   = float(np.mean(H_norm))
    p25, p75       = np.percentile(H_norm, [25, 75])
    frac_ambiguous = float(np.mean(H_norm > 0.5))        # fraction of samples with entropy > 0.5

    # Test: Median posterior entropy > 0.5 ?
    entropy_flag = median_entropy > 0.5
    
    # ---------- [9b] Save the 5 QC checks to JSON ----------
    domain_pass = (
        (gmm_means[0,1] > gmm_means[1,1] and gmm_means[0,0] < gmm_means[1,0]) or
        (gmm_means[0,1] < gmm_means[1,1] and gmm_means[0,0] > gmm_means[1,0])
    )
    multimodality_pass   = not bool(any_unimodal)
    class_sep_2d_pass    = (res["verdict"] == "likely 2 clusters")
    class_imbalance_pass = not bool(collapse_suspected)
    model_entropy_pass   = not bool(entropy_flag)

    qc_flags = {
        "1_domain_alignment":            {"pass": bool(domain_pass)},
        "2_multimodality_each_feature":  {
            "pass": bool(multimodality_pass),
            "diptest": {
                "soma_density": {"dip": float(dip_density), "p_value": float(pval_density)},
                "soma_size":    {"dip": float(dip_size),    "p_value": float(pval_size)}
            }
        },
        "3_class_separability_2d": {
            "pass": bool(class_sep_2d_pass),
            "bic": {
                "bic_1comp": float(res["bic_1comp"]),
                "bic_2comp": float(res["bic_2comp"]),
                "delta_bic_2minus1": float(res["delta_bic_2minus1"]),
                "verdict": res["verdict"]
            }
        },
        "4_class_imbalance": {
            "pass": bool(class_imbalance_pass),
            "weights": np.asarray(gmm_2d.weights_, dtype=float).tolist(),
            "n_samples": int(n_samples),
            "min_weight_floor": float(max(0.05, 5.0 / max(1, n_samples))),
            "min_effective_count": float(min_eff_count)
        },
        "5_model_fit_diagnostics": {
            "pass": bool(model_entropy_pass),
            "posterior_entropy": {
                "median": float(median_entropy),
                "mean": float(mean_entropy),
                "p25": float(p25),
                "p75": float(p75),
                "frac_ambiguous_gt_0.5": float(frac_ambiguous)
            }
        }
    }

    n_passed = sum(1 for v in qc_flags.values() if v["pass"])
    n_failed = len(qc_flags) - n_passed
    if n_failed == 0:
        final_decision = "PASS"
    elif qc_flags["2_multimodality_each_feature"]["pass"] is False and all(
        v["pass"] for k, v in qc_flags.items() if k != "2_multimodality_each_feature"
    ):
        final_decision = "WARNING_unimodality"
    else:
        final_decision = "REVIEW"

    qc_payload = {
        "timestamp": datetime.datetime.now().isoformat(),
        "slide_name": slide_name,
        "summary": {"passed": n_passed, "failed": n_failed, "final_decision": final_decision},
        "checks": qc_flags,
    }

    qc_json_path = os.path.join(odir, "qc_results.json")
    with open(qc_json_path, "w") as f:
        json.dump(qc_payload, f, indent=2)
    print(f"[QC] Saved {qc_json_path}")

    
    # %% ---------- [10] Summary Slide for debugging ----------
    if debug:
        binary_mask = (tissue_mask_resized > 0)
        
         # --- Build high-res overlay directly from the upsampled masks ---
        overlay_thumb = np.zeros((*target_shape, 3), dtype=np.uint8)
        overlay_thumb[gm_mask_thumb] = [0,0,255]   # GM in blue
        overlay_thumb[wm_mask_thumb] = [255, 0, 0]   # WM in red
        # --- Build high-res RGB label map if you still need it separately ---
        label_map_thumb_rgb = np.zeros((*target_shape, 3), dtype=np.uint8)
        label_map_thumb_rgb[label_map_thumb == 1] = [0,0,255]
        label_map_thumb_rgb[label_map_thumb == 2] = [255, 0, 0]

        # --- Create cleaned RGB label map ---
        label_map_thumb_cleaned = np.zeros_like(label_map_thumb)
        label_map_thumb_cleaned[gm_mask_thumb_clean] = 1
        label_map_thumb_cleaned[wm_mask_thumb_clean] = 2
        label_map_thumb_cleaned_rgb = np.zeros((*label_map_thumb.shape, 3), dtype=np.uint8)
        label_map_thumb_cleaned_rgb[label_map_thumb_cleaned == 1] = [0,0,255]   # GM in blue
        label_map_thumb_cleaned_rgb[label_map_thumb_cleaned == 2] = [255, 0, 0]   # WM in red
        
        # Node-order flip mask: shape (N_nodes,), dtype=bool
        flip_mask_nodes = flip_map[y_all, x_all].astype(bool)

        out_png = save_slide_dashboard(tb_img=tb.img,
                                       slide_name=slide_name,
                                       tissue_mask_img=tissue_mask.img,
                                       binary_mask=binary_mask,
                                       soma_density_np=soma_density_np,
                                       soma_size_np=soma_size_np,
                                       soma_density_tissue_inliner=soma_density_tissue_inliner,
                                       soma_size_tissue_inliner=soma_size_tissue_inliner,
                                       gm_prob_map_thumb=gm_prob_map_thumb,
                                       wm_prob_map_thumb=wm_prob_map_thumb,
                                       label_map_thumb_rgb=label_map_thumb_rgb,
                                       label_map_thumb_cleaned_rgb=label_map_thumb_cleaned_rgb,
                                       flip_map=flip_map,
                                       tissue_idx=tissue_idx,
                                       X_scaled_inliner=X_scaled_inliner,
                                       X_scaled_full=X_scaled_full,
                                       gmm_2d=gmm_2d,
                                       gm_label=gm_label,
                                       wm_label=wm_label,
                                       flip_mask_nodes=flip_mask_nodes,   # or None
                                       
                                       any_unimodal=any_unimodal,
                                       verdict_2d=res["verdict"],
                                       collapse_suspected=collapse_suspected,
                                       entropy_flag=entropy_flag,
                                       dip_density=dip_density,
                                       pval_density=pval_density,
                                       dip_size=dip_size,
                                       pval_size=pval_size,
                                       
                                       out_path=args.debug_path,
                                       axis_pad_frac=0.2,
                                       dpi=300
                                       )

    # %% ---------- [11] Save the Final GM/WM Mask ----------
    np.save(os.path.join(odir, 'gm_mask.npy'), gm_mask_thumb_clean)
    np.save(os.path.join(odir, 'wm_mask.npy'), wm_mask_thumb_clean)
    
    # Contours + GM/WM Segmentation figure
    res = render_neuseg_contours(
        tb_img=tb.img,
        gm_mask_thumb_clean=gm_mask_thumb_clean,
        wm_mask_thumb_clean=wm_mask_thumb_clean,
        tissue_bool=tissue_bool,
        line_radius_px=3,
        show_fills=True,             # or False for contours only
        title=slide_name,
        save_path=os.path.join(odir, 'GMWM_Segmentation.png'),
        dpi=300,
        legend=True
    )
    
    return 0
if __name__ == "__main__":
    start_time = time.time()
    rc = main(sys.argv[1:])          # pass only args, not the script name
    elapsed = time.time() - start_time
    mins, secs = divmod(elapsed, 60)
    print(f"Total execution time: {int(mins)} minutes {int(secs)} seconds")
    sys.exit(rc)    