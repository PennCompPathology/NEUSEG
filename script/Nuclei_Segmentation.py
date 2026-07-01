
import os
# force all BLAS libraries to single‑thread per process
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"

import sys
import argparse
import sys
import tempfile
import time
import shutil
import warnings
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cv2
from tqdm import tqdm

from skimage.color import rgb2hed
from skimage.filters import gaussian, threshold_otsu, threshold_multiotsu
from skimage.morphology import remove_small_holes
from skimage import measure

from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

# sys.path.insert(0, '/home/hsroh/Research/sana/src')
import pdnl_sana as sana
import pdnl_sana.logging
import pdnl_sana.slide
import pdnl_sana.filter
import pdnl_sana.process
import pdnl_sana.segment
import pdnl_sana.quantify

import platform
def log_environment():
    print("===== NEUSEG Environment =====")
    print("Python:", sys.version)
    print("Platform:", platform.platform())
    print("NumPy version:", np.__version__)
    print("================================")

# import argparse
# import sys
# import tempfile
# from tqdm import tqdm
# import time
# import shutil
# from multiprocessing import Pool, cpu_count
# from concurrent.futures import ProcessPoolExecutor, as_completed

# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.mixture import GaussianMixture
# from sklearn.cluster import DBSCAN
# import warnings
# warnings.filterwarnings("ignore")

# # Entropy Tissue Mask
# from scipy.signal import argrelextrema
# from skimage.filters.rank import entropy
# from skimage.morphology import disk
# from copy import deepcopy
# from skimage import measure

# from skimage.filters import threshold_multiotsu, threshold_otsu, threshold_triangle
# from scipy.ndimage import gaussian_filter1d
# from scipy.signal import argrelextrema, find_peaks
# from skimage.morphology import remove_small_holes

# # HEM
# from skimage.color import rgb2hed
# from skimage.filters import gaussian, threshold_otsu
# import numpy as np
# from skimage import measure
# from skimage.morphology import remove_small_holes

# import sys
# # sys.path.insert(0, '/home/hsroh/Research/sana/src')
# import pdnl_sana as sana
# import pdnl_sana.logging
# import pdnl_sana.slide
# import pdnl_sana.filter
# import pdnl_sana.process
# import pdnl_sana.quantify

USE_TEMP = True
DEBUG = False

def pick_threshold_rightmost_valley_vs_high(ent, max_valley=4.0, nbins=60, sigma_bins=1.0):
    """
    1) t_high: upper cut from multi-Otsu (classes=2), fallback to single Otsu
    2) t_valley: rightmost local minimum of (smoothed) histogram with center < max_valley
    Returns (t_high, t_valley, t_pick=max(t_high, t_valley or -inf))
    """
    x = np.asarray(ent, np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).ravel()

    # --- t_high ---
    try:
        t_vals = threshold_multiotsu(x, classes=2)
        t_high = float(t_vals[-1])
    except Exception:
        t_high = float(threshold_otsu(x))

    # --- histogram + valley ---
    counts, edges = np.histogram(x, bins=nbins)
    centers = 0.5*(edges[:-1] + edges[1:])
    counts_s = gaussian_filter1d(counts.astype(float), sigma=sigma_bins)

    mins = argrelextrema(counts_s, np.less)[0]
    cand = mins[centers[mins] < max_valley]  # your criterion: center < 4
    t_valley = (centers[cand].max() if cand.size else np.nan)

    # pick conservatively
    t_pick = np.nanmax([v for v in (t_high, t_valley) if np.isfinite(v)])
    return t_high, t_valley, float(t_pick)

def main(argv=None) -> int:
    # %% Argument Parser   
    parser = argparse.ArgumentParser(description="NEUSEG Pipeline (Neuclei Segmentation) for one slide")
    parser.add_argument("slide", help="Path to input .svs")
    parser.add_argument("out_root", help="Output root directory")
    args = parser.parse_args(argv)
    
    slide_f = args.slide # Input .svs WSI
    odir = args.out_root # Output directory
    # if the user passed a 3rd arg, use it; otherwise use all cores
    if len(argv) > 3:
        n_processes = int(argv[3])
    else:
        n_processes = max(1, int(cpu_count() * 2 / 3))
    print(f"Working on {slide_f}")
    print(f"Using {n_processes} worker processes")

    # %% Copying the WSI + Sana Loader
    slide_name = os.path.splitext(os.path.basename(slide_f))[0]
    odir = os.path.join(odir, slide_name)
    os.makedirs(odir, exist_ok=True)

    if USE_TEMP:
        temp_directory = tempfile.TemporaryDirectory()
        temp_dir = temp_directory.name
    else:
        temp_dir = os.path.join(odir, 'tmp')
        os.makedirs(temp_dir, exist_ok=True)

    # Copy the WSI
    #slide_f_tmp = os.path.join(temp_dir, slide_name+'.svs')
    ext = os.path.splitext(slide_f)[1]  # keep .tif/.tiff/.svs
    slide_f_tmp = os.path.join(temp_dir, slide_name + ext)
    shutil.copyfile(slide_f, slide_f_tmp)

    # initialize the slide
    logger = sana.logging.Logger('normal', os.path.join(odir, 'parameters.pkl'))
    loader = sana.slide.Loader(logger, slide_f_tmp)
    tb = loader.load_thumbnail()
    tb.save(os.path.join(odir, 'thumbnail.png'))

    # %% [1] Tissue Mask Extraction
    # --- (1) Color deconvolution --> Hematoxylin Channel ---
    HEM_SIGMA = 1.0  # blurring amount
    hed = rgb2hed(tb.img.astype(np.float32) / 255.0)
    H = hed[..., 0]  # Hematoxylin OD
    
    # Normalization
    perct = 5
    lo, hi = np.percentile(H, perct), np.percentile(H, 100 - perct)
    H = np.clip(H, lo, hi)
    H = (H - H.min()) / (H.max() - H.min() + 1e-8)

    # Blur slightly to stabilize thresholding
    wsi_hem = gaussian(H, sigma=HEM_SIGMA)

    # --- (2) Threshold with Otsu ---
    thresh_localminimal = threshold_otsu(wsi_hem)   # <-- Otsu on hematoxylin OD
    RELAX_H = 0.9
    thresh_localminimal = RELAX_H * thresh_localminimal

    # --- (3) Initial mask + smoothing (closing → opening) ---
    tissue_mask_bool = (wsi_hem > thresh_localminimal).astype(np.uint8)
    tissue_mask = sana.image.frame_like(tb, tissue_mask_bool)
    tissue_mask.to_short()
    tissue_mask.apply_morphology_filter(sana.filter.MorphologyFilter('closing', 'ellipse', 2))
    tissue_mask.apply_morphology_filter(sana.filter.MorphologyFilter('opening', 'ellipse', 15))
    mask_smooth = (tissue_mask.img.squeeze() > 0) # Back to np nadaary

    # --- (4) Connected-component gating with border logic (Remove tissue touching the WSI edge) ---
    mask = mask_smooth.astype(np.uint8)
    Hh, Ww = mask.shape
    labels = measure.label(mask, connectivity=2)

    if labels.max() == 0:
        kept = np.zeros_like(mask, dtype=np.uint8)
    else:
        props = measure.regionprops(labels)

        def touches_border(p):
            minr, minc, maxr, maxc = p.bbox  # maxr/maxc exclusive
            return (minr == 0) or (minc == 0) or (maxr == Hh) or (maxc == Ww)

        areas = np.array([p.area for p in props])
        lbls  = np.array([p.label for p in props])
        touch = np.array([touches_border(p) for p in props])

        largest_label = lbls[np.argmax(areas)]
        area_thresh = 0.05 * Hh * Ww  # keep big border-touching blobs
        big_border_labels = set(lbls[(touch) & (areas >= area_thresh)])

        keep_labels = set(lbls[~touch]) | {largest_label} | big_border_labels
        # Kept tissue mask component Mask
        kept = np.isin(labels, list(keep_labels)).astype(np.uint8)

    # --- (5) Fill small internal holes only ---
    SMALL_HOLE_FRAC = 0.001  # 0.1% of tissue area
    kept_bool = kept.astype(bool)
    tissue_area_px = int(kept_bool.sum())
    min_hole_area_px = max(1, int(round(SMALL_HOLE_FRAC * tissue_area_px)))
    kept_filled = remove_small_holes(kept_bool, area_threshold=min_hole_area_px, connectivity=2)

    # --- (6) Finalize (SANA frame) ---
    kept = kept_filled.astype(np.uint8)
    tissue_mask = sana.image.frame_like(tb, kept)
    tissue_mask.to_short()
    tissue_bodies, tissue_holes = tissue_mask.to_polygons()
    np.save(os.path.join(odir, 'tissue_mask.npy'), tissue_mask.img)

    # %% [2] Patch-Wise Neuclei Segmentation
    # cache the frames
    level = 0
    frame_size = 2048   # Size of the non-overlapping patch (2048 x 2048)
    size = sana.geo.Point(frame_size, frame_size, is_micron=False, level=level)
    framer = sana.slide.Framer(loader, size=size, step=size, level=level, rois={'tissue': tissue_bodies})

    # TODO: doing this here because of pickling bugs, clean this up
    rois = [x.copy() for x in tissue_bodies]
    [framer.loader.converter.rescale(x, 0) for x in rois]

    fn_args = []
    for i, j in [(i,j) for i in range(framer.nframes[0]) for j in range(framer.nframes[1])]:
        args = (temp_dir, i, j, slide_f_tmp, frame_size, level, rois)
        fn_args.append(args)
            
    hem_histograms, dab_histograms = [], []
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = {executor.submit(sana.process.preprocess_wsi_chunk_wrapper, args) for args in fn_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Frame Preprocessing"):
            hem_hist, dab_hist = future.result()
            if not hem_hist is None:
                hem_histograms.append(hem_hist)
                dab_histograms.append(dab_hist)   

    # get the WSI thresholds for the preprocessed HEM and DAB channels
    global_hem_threshold = sana.threshold.triangular_method(
        np.mean(hem_histograms, axis=0)[:,0], strictness=-0.8)
    global_dab_threshold = sana.threshold.triangular_method(
        np.mean(dab_histograms, axis=0)[:,0], strictness=-0.5)
    
    print(f'global_hem_threshold: {global_hem_threshold}')
    print(f'global_dab_threshold: {global_dab_threshold}') 
    
    # segment the cells
    fn_args = []
    for (i,j) in [(i,j) for i in range(framer.nframes[0]) for j in range(framer.nframes[1])]:

        args = [temp_dir, i, j, global_hem_threshold, global_dab_threshold]
        fn_args.append(args)

    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = {executor.submit(sana.segment.segment_wsi_chunk_wrapper, args) for args in fn_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Cell Segmentation"):
            _ = future.result()

    # reload the features: (N,4) -> feats[i] = [x,y,area,intensity]
    feats = []
    for (i,j) in [(i,j) for i in range(framer.nframes[0]) for j in range(framer.nframes[1])]:
        if os.path.exists(os.path.join(temp_dir, f'feats_{i}_{j}.npy')):
            feats.append(np.load(os.path.join(temp_dir, f'feats_{i}_{j}.npy')))
    # feats: feats[i] = [x,y,area,intensity]
    feats = np.concatenate(feats, axis=0)

    # TODO: put this next part into a function and potentially run twice
    # Amount to downsample the thumbnail resolution space
    # NOTE: setting this pretty small to get smooth segmentations
    # ds_thumbnail = 6
    ds_thumbnail = 1

    # Size of window for the gaussian kernel
    # NOTE: setting this really big gives us smooth segmentations
    window_size = 1000 # microns
    window_size = sana.geo.Point(window_size, window_size, is_micron=True)

    # Break the slide coordinate system into more manageable chunks
    # NOTE: this parameter doesn't change the heatmap at all, this is just for processing speed
    chunk_size = 2000 # pixels
    chunk_size = sana.geo.Point(chunk_size, chunk_size, is_micron=False, level=0)

    # define the size of our coordinate systems
    w_out, h_out = loader.converter.to_int(tb.size() / ds_thumbnail)
    w_thumbnail, h_thumbnail = tb.size()
    w_slide, h_slide = loader.level_dimensions[0]

    # initialize the output array
    feature_heatmap = np.zeros((h_out, w_out, 3), dtype=float)

    # amount we're downsampling from the slide resolution
    ds_slide = ds_thumbnail * loader.converter.ds[2]

    # calculate the parameters in slide resolution pixels
    window_size_slide = loader.converter.to_pixels(window_size, level=0)
    chunk_size_slide = loader.converter.to_pixels(chunk_size, level=0)

    # calculate the parameters in output resolution pixels
    window_size_out = loader.converter.to_pixels(window_size, level=2) / ds_thumbnail
    chunk_size_out = loader.converter.to_pixels(chunk_size, level=2) / ds_thumbnail

    # calculate the upper left coordinates of each chunk
    chunk_xs = np.arange(0, w_out + chunk_size_out[0], chunk_size_out[0])
    chunk_ys = np.arange(0, h_out + chunk_size_out[1], chunk_size_out[1])

    # aggregate each chunk independently
    fn_args = []
    for (chunk_y_out, chunk_x_out) in [(y, x) for y in chunk_ys for x in chunk_xs]:

        # pad the chunk by the window since our output pixels are centered
        # TODO: i think these bounds are wrong, and there are too many cells in the chunk
        x0_out = chunk_x_out - window_size_out[0] / 2
        y0_out = chunk_y_out - window_size_out[1] / 2
        x1_out = x0_out + chunk_size_out[0] + window_size_out[0]
        y1_out = y0_out + chunk_size_out[1] + window_size_out[1]
    
        # calculate the chunk coordinates in slide resolution pixels
        x0, y0, x1, y1 = [x*ds_slide for x in [x0_out, y0_out, x1_out, y1_out]]
    
        # get the samples within the padded chunk
        chunk_sample_idxs = (x0 < feats[:,0]) & (feats[:,0] < x1) & (y0 < feats[:,1]) & (feats[:,1] < y1)
        chunk_feats = feats[chunk_sample_idxs].copy()

        # no samples found, no need to process this chunk
        if len(chunk_feats) == 0:
            continue

        # get the output pixels inside the chunk
        i0 = int(round(np.clip(chunk_x_out, 0, w_out-1)))
        j0 = int(round(np.clip(chunk_y_out, 0, h_out-1)))
        i1 = int(round(np.clip(chunk_x_out + chunk_size_out[0], 0, w_out-1)))
        j1 = int(round(np.clip(chunk_y_out + chunk_size_out[1], 0, h_out-1)))
    
        args = (window_size_slide, feats, i0, j0, i1, j1, ds_slide)
        fn_args.append(args)

    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        future_args_mapping = {}
        futures = set()
        for args in fn_args:
            future = executor.submit(sana.quantify.aggregate_features_wrapper, args)
            future_args_mapping[future] = args
            futures.add(future)
        for future in tqdm(as_completed(futures), total=len(futures), desc="Aggregate Features"):
            out = future.result()
            args = future_args_mapping[future]
            i0, j0, i1, j1 = args[2:6]
            feature_heatmap[j0:j1, i0:i1] = out

    # Upscale and save the heatmaps
    feature_heatmap = sana.image.frame_like(tb, feature_heatmap)                        # Feature heatmap downsampled from tb by ds_thumbnail
    np.save(os.path.join(odir, f'feature_heatmap.npy'), feature_heatmap.img)
    
    feature_heatmap_up = feature_heatmap.copy()
    feature_heatmap_up.resize(tb.size(), interpolation=cv2.INTER_CUBIC)
    np.save(os.path.join(odir, f'feature_heatmap_tb_reso.npy'), feature_heatmap_up.img) # Feature heatmap upsampled to thumbnail resolution
    
    # --- Diagnostics: print key resolutions/sizes (h, w) ---
    if DEBUG:
        # 1) WSI full resolution (level-0)
        w0, h0 = loader.level_dimensions[0]          # returns (w, h)
        print(f"WSI level-0 (full-res):        h={h0}, w={w0}")

        # 2) Thumbnail / mask
        wt, ht = tb.size()                           # returns (w, h)
        print(f"Thumbnail image:              h={ht}, w={wt}")

        hm, wm = tissue_mask.img.shape[:2]           # mask is a numpy array
        print(f"Tissue mask:                  h={hm}, w={wm}")

        # 3) Feature heatmap (before upscaling)
        wf, hf = feature_heatmap.size()              # returns (w, h) for the SANA frame
        ch = feature_heatmap.img.shape[2] if feature_heatmap.img.ndim == 3 else 1
        print(f"feature_heatmap (pre-up):     h={hf}, w={wf}, channels={ch}")

        # 4) Feature heatmap after upscaling to thumbnail resolution
        wfu, hfu = feature_heatmap_up.size()         # returns (w, h)
        print(f"feature_heatmap_up (thumb):   h={hfu}, w={wfu}")
        
    return 0


if __name__ == "__main__":
    log_environment()
    start_time = time.time()
    rc = main(sys.argv[1:])      # pass only args, not the script name
    elapsed = time.time() - start_time
    mins, secs = divmod(elapsed, 60)
    print(f"Total execution time: {int(mins)} minutes {int(secs)} seconds") 
    sys.exit(rc)  