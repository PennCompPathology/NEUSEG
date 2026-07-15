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

import pdnl_sana.logging
import pdnl_sana.slide
import pdnl_sana.geo
import pdnl_sana.process
import pdnl_sana.threshold
import pdnl_sana.segment

def dispatch_jobs(job: Callable, job_args: list[dict], n_cores: int=1, progress_str: str=""):
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
    
    # TODO: save cells to file and continue with algorithm

    # run_features(**kwargs)
def run_features(logger, **kwargs):
    run_tissue(**kwargs)
def run_tissue(logger, **kwargs):
    run_cortex(**kwargs)
def run_cortex(logger, **args):
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
    parser.add_argument('--checkpoint',
                        help="start from a specific portion of the NEUSEG algorithm", 
                        choices=['cells', 'features', 'tissue', 'cortex'], 
                        default=None)
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


    if args.checkpoint is None or args.checkpoint == 'cells':
        run_cells(logger=logger, **vars(args))
    # elif args.mode == 'features':
    #     run_features(logger=logger, **args)
    # elif args.mode == 'tissue':
    #     run_tissue(logger=logger, **args)    
    # elif args.mode == 'cortex':
    #     run_cortex(logger=logger, **args)

if __name__ == "__main__":
    main()
