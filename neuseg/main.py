#!/usr/bin/env python3

import os
import sys
import argparse
from collections.abc import Callable

import tempfile
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

import pdnl_sana.logging
import pdnl_sana.slide
import pdnl_sana.geo
import pdnl_sana.process
import pdnl_sana.threshold
import pdnl_sana.segment
import pdnl_sana.quantify

def dispatch_jobs(job: Callable, job_args: list[dict], n_cores: int=1, progress_str: str=""):
    if n_cores == 1:
        for args in tqdm(job_args, desc=progress_str):
            yield job(**args)
    else:
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = {executor.submit(job, **args) for args in job_args}
            for future in tqdm(as_completed(futures), total=len(futures), desc=progress_str):
                yield(future.result())

def run_cells(logger: pdnl_sana.logging.Logger, input_slide: str, output_directory: str, tmp_directory: str, level: int=0, frame_size: int=1024, rois: dict[str: list[pdnl_sana.geo.Polygon]]={}, n_cores: int=1, **kwargs):

    # TODO: copy WSI? probably not necessary...
    loader = pdnl_sana.slide.Loader(logger, input_slide)

    # TODO: save a 8x tb instead of 16x
    tb = loader.load_thumbnail()
    tb.save(os.path.join(output_directory, 'thumbnail.png'))

    # create a rudimentary tissue mask
    # TODO: maybe just turn this off? doesn't save a ton of time and adds failure risk
    tissue_mask = pdnl_sana.slide.find_tissue(tb)
    if tissue_mask is None:
        logger.warning("Cannot find tissue in slide!")
        rois, roi_holes = {}, []
    else:
        rois, roi_holes = tissue_mask.to_polygons()
        rois = {'Tissue': rois}

    # get the coordinates in the WSI to load chunks from
    size = pdnl_sana.geo.Point(frame_size, frame_size, is_micron=False, level=level)
    framer = pdnl_sana.slide.Framer(loader, size=size, step=size, level=level, rois=rois, roi_holes=roi_holes)
    frame_idxs = [(j,i) for j in range(framer.nframes[0]) for i in range(framer.nframes[1])]

    # load each chunk and get the histogram
    job_args = [{'tmp_directory': tmp_directory, 'j': j, 'i': i, 'input_slide': input_slide, 'level': level, 'size': frame_size, 'rois': rois, 'roi_holes': roi_holes} for j, i in frame_idxs]
    histograms = [res for res in dispatch_jobs(pdnl_sana.process.preprocess_chunk, job_args, n_cores=n_cores, progress_str="Preprocessing RGB Chunks") if not res is None]

    # calculate the stain threshold for the image
    global_threshold = pdnl_sana.threshold.triangular_method(np.mean(histograms, axis=0)[:,0], strictness=-0.8)
    logger.debug(f"Global Stain Threshold: {global_threshold}")

    # segment the cells in the image
    job_args = [{'tmp_directory': tmp_directory, 'j': j, 'i': i, 'threshold': global_threshold} for (j,i) in frame_idxs]
    cells = np.concatenate([cells for cells in dispatch_jobs(pdnl_sana.segment.segment_chunk, job_args, n_cores=n_cores, progress_str="Segmenting Cells")], axis=0)
    
    # save our work and continue to the next step
    np.save(os.path.join(output_directory, 'cells.npy'), cells)
    run_features(logger=logger, output_directory=output_directory, loader=loader, n_cores=n_cores, **kwargs)

def run_features(logger: pdnl_sana.logging.Logger, output_directory: str, loader: pdnl_sana.slide.Loader, ds_thumbnail: float=1, window_size: float=1000, n_cores: int=1, **kwargs):
    cells_f = os.path.join(output_directory, 'cells.npy')
    tb_f = os.path.join(output_directory, 'thumbnail.png')
    if not os.path.exists(cells_f) or not os.path.exists(tb_f):
        logger.error("CELL ARRAY DOES NOT EXIST, must re-run with: neuseg ... --entrypoint cells")
        return
    cells = np.load(cells_f)
    tb = pdnl_sana.image.Frame(tb_f)

    # define the size of the coordinate systems
    w_out, h_out = loader.converter.to_int(tb.size() / ds_thumbnail)
    w_thumbnail, h_thumbnail = tb.size()
    w_slide, h_slide = loader.level_dimensions[0]
    ds_slide = ds_thumbnail * loader.converter.ds[loader.thumbnail_level]

    # define the upper left coordinates of each chunk 
    # NOTE: this is for memory management purposes and does not affect the heatmaps
    # TODO: test 2048 and 4096?
    chunk_size = 1024
    chunk_size = pdnl_sana.geo.Point(chunk_size, chunk_size, is_micron=False, level=0)
    chunk_size_out = loader.converter.to_pixels(chunk_size, level=loader.thumbnail_level) / ds_thumbnail
    chunk_xs = np.arange(0, w_out + chunk_size_out[0], chunk_size_out[0])
    chunk_ys = np.arange(0, h_out + chunk_size_out[1], chunk_size_out[1])

    window_size = pdnl_sana.geo.Point(window_size, window_size, is_micron=True)
    window_size_slide = loader.converter.to_pixels(window_size, level=0)
    window_size_out = loader.converter.to_pixels(window_size, level=loader.thumbnail_level) / ds_thumbnail

    feature_heatmap = np.zeros((h_out, w_out, 3), dtype=float)
    job_args = []
    for (chunk_y_out, chunk_x_out) in [(y, x) for y in chunk_ys for x in chunk_xs]:

        # pad the chunk by the window size to center the output heatmap pixels
        x0_out = chunk_x_out - window_size_out[0] / 2
        y0_out = chunk_y_out - window_size_out[1] / 2
        x1_out = x0_out + chunk_size_out[0] + window_size_out[0]
        y1_out = y0_out + chunk_size_out[1] + window_size_out[1]

        # calculate the chunk coordinates
        x0, y0, x1, y1 = [v*ds_slide for v in [x0_out, y0_out, x1_out, y1_out]]
        loc, size = pdnl_sana.geo.Point(x0, y0), pdnl_sana.geo.Point(x1, y1)

        # get the valid cells for this chunk
        chunk_sample_idxs = pdnl_sana.quantify.find_local_samples(cells[:,0], cells[:,1], loc, size)
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
    for (out, i0, j0, i1, j1) in dispatch_jobs(pdnl_sana.quantify.aggregate_cells, job_args, n_cores=n_cores, progress_str='Aggregating Cells'):
        feature_heatmap[j0:j1, i0:i1] = out
    np.save(os.path.join(output_directory, 'feature_heatmap.npy'), feature_heatmap)

    # fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
    # ax = ax.ravel()
    # ax[0].imshow(tb.img)
    # ds_cells = 4
    # ax[0].plot(cells[::ds_cells,0]/loader.level_downsamples[2], cells[::ds_cells,1]/loader.level_downsamples[2], '*', markersize=1, color='red')
    # for i in range(3):
    #     im = feature_heatmap[:,:,i]
    #     mu, sd = np.nanmean(im), np.nanstd(im)
    #     ax[i+1].imshow(im, cmap='gray', vmin=mu-2*1.96*sd, vmax=mu+2*1.96*sd, extent=(0,w_thumbnail,h_thumbnail,0))
    # plt.show()

    # run_tissue(logger=logger, **kwargs)

def run_tissue(logger: pdnl_sana.logging.Logger, output_directory: str, **kwargs):
    features_f = os.path.join(output_directory, 'features.npy')
    if not os.path.exists(features_f):
        logger.error("FEATURE ARRAY DOES NOT EXIST, must re-run with: neuseg ... --entrypoint features")
        return
    
    run_cortex(logger=logger, **kwargs)

def run_cortex(logger: pdnl_sana.logging.Logger, output_directory: str, **kwargs):
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_slide', 
                        help="path to WSI file", 
                        required=True)
    parser.add_argument('-o', '--output_directory', 
                        help="directory path to save outputs to", 
                        required=True)
    parser.add_argument('--n_cores',                         
                        help="multiprocessing cpu cores to use", 
                        type=int, default=1)
    parser.add_argument('--tmp_directory', 
                        help="use a specific location for temporary files which will not be auto-deleted",
                        default=None)
    parser.add_argument('--frame_size',
                        help="size of frame chunk to process for I/O purposes",
                        type=int, default=1024)
    parser.add_argument('--ds_thumbnail',
                        help="resolution of feature heatmap to generate with respect to the thumbnail",
                        type=float, default=1.0)
    parser.add_argument('--window_size',
                        help="distance (microns) to look when aggregating cells for feature heatmaps",
                        type=float, default=1000.0)
    parser.add_argument('--entrypoint',
                        help="start from a specific portion of the NEUSEG algorithm", 
                        choices=['cells', 'features', 'tissue', 'cortex'], 
                        default='cells')
    parser.add_argument('--debug_level', 
                        help="amount of information to display to use",
                        choices=['quiet', 'normal', 'debug', 'full'],
                        default='normal')
    args = parser.parse_args()

    logger_fpath = os.path.join(args.output_directory, 'log.pkl')
    logger = pdnl_sana.logging.Logger(args.debug_level, logger_fpath, name="NEUSEG")

    logger.debug(f'Using {args.n_cores} CPU cores out of {int(cpu_count()*2/3)} available')

    if args.tmp_directory is None:
        tmp_dir = tempfile.TemporaryDirectory()
        args.tmp_directory = tmp_dir.name
    logger.debug(f'Caching data in temporary directory: {args.tmp_directory}')

    if args.entrypoint == 'cells':
        run_cells(logger=logger, **vars(args))
    elif args.entrypoint == 'features':
        loader = pdnl_sana.slide.Loader(logger, args.input_slide)
        run_features(logger=logger, loader=loader, **vars(args))
    # elif args.mode == 'tissue':
    #     run_tissue(logger=logger, **args)    
    # elif args.mode == 'cortex':
    #     run_cortex(logger=logger, **args)

if __name__ == "__main__":
    main()
