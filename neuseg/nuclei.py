
import os
import tempfile

import numpy as np
from tqdm import tqdm

import pdnl_sana as sana
import pdnl_sana.logging
import pdnl_sana.geo
import pdnl_sana.process
import pdnl_sana.utils
import pdnl_sana.filter

from .tissue import get_tissue_mask, segment_wm

from matplotlib import pyplot as plt

def preprocess_counterstain_chunk(tmp_directory, j, i, input_slide, staining_code, level, size, rois, roi_holes):
    """
    This function applies color deconvolution to an RGB chunk and saves the stain information
    """
    logger = sana.logging.Logger('normal', os.path.join(tmp_directory, f'parameters_{j}_{i}.pkl'))
    loader = sana.slide.Loader(logger, input_slide)
    size = sana.geo.Point(size, size, is_micron=False, level=level)
    # TODO: fix for pickling error, either need dill or use __reduce__ in sana.geo.Array
    for key in rois:
        for roi in rois[key]:
            roi.is_micron = False
            roi.level = level
    for roi_hole in roi_holes:
        roi_hole.is_micron = False
        roi_hole.level = level
    framer = sana.slide.Framer(loader, size=size, step=size, level=level, rois=rois, roi_holes=roi_holes)

    # get the frame mask
    mask, _ = framer.load_mask(j,i)

    # decide if it's worth it to process this frame
    if np.sum(mask.img) < 0.005*mask.img.shape[0]*mask.img.shape[1]:
        return None

    # extract the frame from the WSI
    frame = framer.load_frame(j,i)
    
    # preprocess the frame
    if staining_code == 'HDAB':
        processor = sana.process.HDABProcessor(
            logger, frame, main_mask=mask, 
            run_hem=True, run_dab=False,
            apply_smoothing=False, 
            normalize_background=True, radius=300.0, overlap=0.50,
        )
        counterstain = processor.hem
    elif staining_code == 'CVDAB':
        processor = sana.process.CVDABProcessor(
            logger, frame, main_mask=mask,
            run_cv=True, run_dab=False,
            apply_smoothing=False,
            normalize_background=True, radius=300.0, overlap=0.50,
        )
        counterstain = processor.cv
    else:
        logger.error(f'STAINING CODE NOT RECOGNIZED -- {staining_code}')
        return

    # cache the stain data
    # TODO: make this generic rather than specific stain
    counterstain.save(os.path.join(tmp_directory, f"counterstain_{j}_{i}.png"))
    mask.save_compressed(os.path.join(tmp_directory, f"mask_{j}_{i}.npz"))
    logger.write_data()

    # return the histograms in order to calculate a global WSI
    histogram = counterstain.get_histogram(mask=mask)

    return histogram

    
def segment_nuclei_chunk(tmp_directory, j, i, threshold, closing_radius=2, opening_radius=2, minimum_soma_radius=3, maximum_soma_radius=None):
    if not os.path.exists(os.path.join(tmp_directory, f"counterstain_{j}_{i}.png")):
        return np.empty([0,4])

    logger = sana.logging.Logger('normal', os.path.join(tmp_directory, f'parameters_{j}_{i}.pkl'))
    level, mpp, ds = logger.data['level'], logger.data['mpp'], logger.data['ds']
    converter = sana.geo.Converter(mpp=mpp, ds=ds)
    if not maximum_soma_radius is None:
        maximum_soma_area = np.pi*maximum_soma_radius**2
        maximum_polygon_area = 5*maximum_soma_area
    else:
        maximum_soma_area = None
        maximum_polygon_area = None

    frame_loc = logger.data['loc']
    stain = sana.image.Frame(os.path.join(tmp_directory, f"counterstain_{j}_{i}.png"))
    mask = sana.image.Frame(os.path.join(tmp_directory, f"mask_{j}_{i}.npz"))
    stain_int = stain.copy()

    # threshold and filter small objects and holes
    stain.threshold(threshold)
    stain.apply_morphology_filter(sana.filter.MorphologyFilter('closing', 'ellipse', closing_radius))
    stain.apply_morphology_filter(sana.filter.MorphologyFilter('opening', 'ellipse', opening_radius))
    stain.mask(mask)

    # remove huge objects within the chunk
    if not maximum_polygon_area is None:
        polys = stain.to_polygons()[0]
        polys = [x for x in polys if x.get_area() <= maximum_polygon_area]
        stain = sana.image.create_mask_like(stain, polys)

    # find all the somas throughout the counterstain
    ctrs = sana.segment.detect_somas(stain, minimum_soma_radius=minimum_soma_radius)
    
    # segment the somas using polygons
    if len(ctrs) != 0:
        polys, _ = stain.instance_segment(ctrs)[0]
        polys = [p for p in polys if not type(p) is list and len(p) != 0]
    else:
        polys = []
            
    # move the polygons into the slide coordinate system
    [p.translate(-frame_loc) for p in polys]

    # re-calculate the centers of the polygons using the bounding box
    bbs = [p.bounding_box() for p in polys]
    ctrs = np.array([loc + size//2 for (loc, size) in bbs])

    # feature 1: calculate the area of each polygon
    areas = np.array([p.get_area() for p in polys])

    # feature 2: calculate the mean intensity within the polygon
    ints = []
    for poly in polys:

        # move to frame coordinate system
        poly.translate(frame_loc)
        
        # extract tile based on the bounding box of the polygon
        loc, size = poly.bounding_box()
        tile = sana.image.Frame(stain_int.get_tile(loc, size))

        # create a mask of pixels within the polygon
        poly.translate(loc)
        tile_mask = sana.image.create_mask_like(tile, [poly])
        poly.translate(-loc)

        # calculate average intensity
        ints.append(np.mean(tile.img[tile_mask.img != 0]))

        # move back to slide coordinate system
        poly.translate(-frame_loc)
    ints = np.array(ints)

    # combine into an (N,4) array
    if len(ctrs) != 0:
        cells = np.vstack([ctrs[:,0], ctrs[:,1], areas, ints]).T
    else:
        cells = np.empty([0,4])

    # filter by the soma area
    if not maximum_soma_area is None:
        cells = cells[cells[:,2] <= maximum_soma_area]

    return cells

def segment_nuclei_wsi(
    input_slide: str, 
    logger: sana.logging.Logger=None, 
    output_path: str=None, 
    staining_code='HDAB',
    minimum_soma_radius: float=2.0, maximum_soma_radius: float=10.0,
    tmp_directory: str=None, 
    level: int=0, frame_size: int=1024, n_cores: int=1, **kwargs):
    """
    This function loads and segments the nuclei in the counterstain of a WSI. It does so in a memory efficient manner by loading the WSI chunk by chunk. This also allows for multi-processing which makes this method feasible for processing many WSI files.
    :param logger: logging object for saving parameters and displaying info
    :param input_slide: path to WSI image file
    :param output_path: optional path to .npz file to write outputs to
    :param staining_code: how the WSI was stained
    :param minimum_soma_radius: microns
    :param maximum_soma_radius: microns
    :param tmp_directory: optionally cache data to a local directory for inspection, NOTE that NEUSEG does not clean up this data for you
    :param n_cores: amount of cores to use for multi-processing purposes
    :param frame_size: size of frame to load in, 1024x1024 generally seems like a good trade off between memory usage and I/O
    :param level: level at which to process at. NEUSEG generally needs ~0.5um per pixel resolution
    """
    if tmp_directory is None:
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_directory = tmp_dir.name
    if logger is None:
        if not output_path is None:
            logger_path = os.path.splitext(output_path)[0]+'.pkl'
        else:
            logger_path = None
        logger = sana.logging.Logger('normal', fpath=logger_path)

    logger.debug(f'Caching data in temporary directory: {tmp_directory}')

    # prepare the slide I/O
    loader = sana.slide.Loader(logger, input_slide)
    logger.data['mpp'] = loader.converter.mpp
    logger.data['ds'] = loader.converter.ds
    logger.data['level_dimensions'] = loader.level_dimensions

    # TODO: save a 8x tb instead of 16x
    tb = loader.load_thumbnail()
    if not output_path is None:
        sana.utils.save_arrays(output_path, thumbnail=tb.img)
    logger.data['thumbnail_level'] = loader.thumbnail_level

    # segment the tissue pixels from the background pixels
    tissue_mask, _ = get_tissue_mask(tb)
    if not output_path is None:
        sana.utils.save_arrays(output_path, tissue_mask=tissue_mask.img)
    rois, roi_holes = tissue_mask.to_polygons()
    rois = {'Tissue': rois}

    # get the coordinates in the WSI to load chunks from
    size = sana.geo.Point(frame_size, frame_size, is_micron=False, level=level)
    framer = sana.slide.Framer(loader, size=size, step=size, level=level, rois=rois, roi_holes=roi_holes)
    frame_idxs = [(j,i) for j in range(framer.nframes[0]) for i in range(framer.nframes[1])]
    logger.data['frame_size'] = frame_size

    # load each chunk and get the histogram
    job = preprocess_counterstain_chunk
    job_name = "Preprocess RGB Chunks"
    job_args = [{
        'tmp_directory': tmp_directory, 'j': j, 'i': i, 
        'input_slide': input_slide, 'staining_code': staining_code,
        'level': level, 'size': frame_size, 
        'rois': rois, 'roi_holes': roi_holes,
        } for j, i in frame_idxs]
    histograms = []
    for res in sana.utils.dispatch_jobs(job, job_args, n_cores=n_cores, progress_str=job_name):
        if not res is None:
            histograms.append(res)

    # calculate the stain threshold for the image
    global_threshold = sana.threshold.triangular_method(np.mean(histograms, axis=0)[:,0], strictness=-0.8)
    logger.debug(f"Global Stain Threshold: {global_threshold}")
    logger.data['global_threshold'] = global_threshold

    # segment the cells in the WSI
    minimum_soma_radius = int(round(loader.converter.mtop(minimum_soma_radius, level)))
    maximum_soma_radius = int(round(loader.converter.mtop(maximum_soma_radius, level)))
    job = segment_nuclei_chunk
    job_name = 'Segment Nuclei'
    job_args = [{
        'tmp_directory': tmp_directory, 'j': j, 'i': i, 
        'threshold': global_threshold, 
        'minimum_soma_radius': minimum_soma_radius, 
        'maximum_soma_radius': maximum_soma_radius,
        } for (j,i) in frame_idxs]
    cells = []
    for res in sana.utils.dispatch_jobs(job, job_args, n_cores=n_cores, progress_str=job_name):
        cells.append(res)
    cells = np.concatenate(cells, axis=0)
    logger.data['minimum_soma_radius'] = minimum_soma_radius
    logger.data['maximum_soma_radius'] = maximum_soma_radius

    if not output_path is None:
        sana.utils.save_arrays(output_path, cells=cells)
        logger.write_data()
        
    return cells, tissue_mask, tb

def aggregate_nuclei_features(cells: np.ndarray, tb: sana.image.Frame, 
                              mpp: float, ds: list[float], level_dimensions: list[np.ndarray],
                              logger: sana.logging.Logger=None,
                              output_path: str=None,
                              ds_thumbnail: float=1, window_size: float=1000, 
                              n_cores: int=1, debug_directory: str="", **kwargs):
    if logger is None:
        if not output_path is None:
            logger_path = os.path.splitext(output_path)[0]+'.pkl'
        else:
            logger_path = None
        logger = sana.logging.Logger('normal', fpath=logger_path)

    converter = sana.geo.Converter(mpp=mpp, ds=ds)

    # define the size of the coordinate systems
    w_out, h_out = converter.to_int(tb.size() / ds_thumbnail)
    w_thumbnail, h_thumbnail = tb.size()
    w_slide, h_slide = level_dimensions[0]
    ds_slide = ds_thumbnail * converter.ds[tb.level]
    logger.data['ds_thumbnail'] = ds_thumbnail

    # define the upper left coordinates of each chunk 
    # NOTE: this is for memory management purposes and does not affect the heatmaps
    # TODO: test 2048 and 4096?
    chunk_size = 2048
    chunk_size = sana.geo.Point(chunk_size, chunk_size, is_micron=False, level=0)
    chunk_size_out = converter.to_pixels(chunk_size.copy(), level=tb.level) / ds_thumbnail
    chunk_xs = np.arange(0, w_out + chunk_size_out[0], chunk_size_out[0])
    chunk_ys = np.arange(0, h_out + chunk_size_out[1], chunk_size_out[1])

    window_size = sana.geo.Point(window_size, window_size, is_micron=True)
    window_size_slide = converter.to_pixels(window_size.copy(), level=0)
    window_size_out = converter.to_pixels(window_size.copy(), level=tb.level) / ds_thumbnail
    logger.data['window_size'] = window_size

    feature_heatmap = sana.image.Frame(np.zeros((h_out, w_out, 3), dtype=float))
    job_args = []
    for (chunk_y_out, chunk_x_out) in tqdm([(y, x) for y in chunk_ys for x in chunk_xs], desc='Preparing Aggregation'):

        # pad the chunk by the window size to center the output heatmap pixels
        x0_out = chunk_x_out - window_size_out[0] / 2
        y0_out = chunk_y_out - window_size_out[1] / 2
        x1_out = x0_out + chunk_size_out[0] + window_size_out[0]
        y1_out = y0_out + chunk_size_out[1] + window_size_out[1]

        # calculate the chunk coordinates
        x0, y0, x1, y1 = [v*ds_slide for v in [x0_out, y0_out, x1_out, y1_out]]
        loc, size = sana.geo.Point(x0, y0), sana.geo.Point(x1, y1)

        # get the valid cells for this chunk
        chunk_sample_idxs = sana.quantify.find_local_samples(cells[:,0], cells[:,1], loc, size)
        chunk_cells = cells[chunk_sample_idxs].copy()
        if len(chunk_cells) == 0:
            continue

        # get the output pixels for the chunk
        i0 = int(round(np.clip(chunk_x_out, 0, w_out-1)))
        j0 = int(round(np.clip(chunk_y_out, 0, h_out-1)))
        i1 = int(round(np.clip(chunk_x_out + chunk_size_out[0], 0, w_out-1)))
        j1 = int(round(np.clip(chunk_y_out + chunk_size_out[1], 0, h_out-1)))

        job_args.append({'window_size': window_size_slide, 'cells': chunk_cells, 'i0': i0, 'j0': j0, 'i1': i1, 'j1': j1, 'ds': ds_slide})

    # generate the feature heatmap and write to disk
    for (out, i0, j0, i1, j1) in sana.utils.dispatch_jobs(sana.quantify.aggregate_cells, job_args, n_cores=n_cores, progress_str='Aggregating Cells'):
        feature_heatmap.img[j0:j1, i0:i1] = out
    if not output_path is None:
        sana.utils.save_arrays(output_path, feature_heatmap=feature_heatmap.img[:,:,:2].astype(np.float32))
        logger.write_data()
        
    if logger.debug_level in ['debug', 'full']:
        os.makedirs(debug_directory, exist_ok=True)
        fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
        ax = ax.ravel()
        ax[0].imshow(tb.img)
        ds_cells = 4
        ax[0].plot(cells[::ds_cells,0]/ds[tb.level], cells[::ds_cells,1]/ds[tb.level], '*', markersize=1, color='red')
        ax[0].set_title(f'({ds_cells}x) Subsampled Cell Coordinates')
        titles = ['Soma Density', 'Avg. Soma Size', 'Avg. Soma Intensity']
        rng = 5
        for i in range(3):
            im = feature_heatmap.img[:,:,i]
            mu, sd = np.nanmean(im), np.nanstd(im)
            ax[i+1].imshow(im, cmap='gray', vmin=mu-rng*1.96*sd, vmax=mu+rng*1.96*sd, extent=(0,w_thumbnail,h_thumbnail,0))
            ax[i+1].set_title(titles[i])

        # record the aggregation window: localize_coordinates smooths with a gaussian
        # of sigma = window/5, so this is what sets how smooth the heatmaps look
        w_um = float(np.asarray(window_size)[0])
        w_l0 = float(np.asarray(window_size_slide)[0])
        w_tb = float(np.asarray(window_size_out)[0])
        fig.suptitle(f'window_size = {w_um:g} um = {w_l0:.1f} level-0 px = {w_tb:.1f} thumbnail px\n'
                     f'gaussian sigma = window/5 = {w_l0/5:.1f} level-0 px = {w_tb/5:.2f} thumbnail px\n'
                     f'ds_thumbnail = {ds_thumbnail:g}', fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.92))   # leave room for the 3-line suptitle
        features_png = os.path.join(debug_directory, 'run_features_output.png')
        fig.savefig(features_png, dpi=150)
        plt.close(fig)
        logger.debug(f'feature figure saved to: {features_png}')

    return feature_heatmap