#!/usr/bin/env python3

import os

# Single-thread BLAS, before numpy is imported: these are read once at load
# time.  Otherwise every worker process spawns a thread per core on top of
# --n_cores, which oversubscribes a big node and, on a >128 core machine,
# segfaults the GMM fit outright.  --n_cores stays the only knob.  Nothing here
# is a big enough matmul to want threaded BLAS anyway.
for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_var, '1')

import sys
import json
import argparse

from multiprocessing import cpu_count

from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from skimage.transform import resize

import pdnl_sana as sana
import pdnl_sana.logging
import pdnl_sana.slide
import pdnl_sana.geo
import pdnl_sana.process
import pdnl_sana.threshold
import pdnl_sana.segment
import pdnl_sana.quantify

from .nuclei import segment_nuclei_wsi, aggregate_nuclei_features
from .tissue import segment_wm

NEUSEG_VERSION = "v1_0"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_slide', 
                        help="path to WSI file", 
                        required=True)
    parser.add_argument('-o', '--output_directory', 
                        help="directory path to save outputs to", 
                        required=True)
    parser.add_argument('--staining_code', 
                        help="how the WSI was stained", 
                        default='HDAB', choices=['HDAB', 'CVDAB'])
    parser.add_argument('--n_cores',                         
                        help="multiprocessing cpu cores to use", 
                        type=int, default=1)
    parser.add_argument('--debug_directory',
                        help="directory path to save debug figures to (default: --output_directory)",
                        default=None)
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
                        choices=['cells', 'features', 'wm', 'cortex'], 
                        default='cells')
    parser.add_argument('--debug_level', 
                        help="amount of information to display to use",
                        choices=['quiet', 'normal', 'debug', 'full'],
                        default='normal')
    args = parser.parse_args()

    if args.debug_directory is None:
        args.debug_directory = args.output_directory

    # named after the slide: several slides can share one --output_directory now
    slide_name = os.path.splitext(os.path.basename(args.input_slide))[0]
    logger_fpath = os.path.join(args.output_directory, slide_name+'_log.pkl')
    logger = pdnl_sana.logging.Logger(args.debug_level, logger_fpath, name="NEUSEG")
    logger.data['VERSION'] = NEUSEG_VERSION

    output_path = os.path.join(args.output_directory, slide_name+'.npz')
    logger.debug(f'Saving outputs to {args.output_directory}')

    logger.debug(f'Using {args.n_cores} CPU cores out of {int(cpu_count()*2/3)} available')

    thumbnail = None
    tissue_mask = None
    cells = None
    feature_heatmap = None
    segmentations = None
    if os.path.exists(output_path):
        arrs = np.load(output_path)
        if 'thumbnail' in arrs:
            level = logger.data['level']
            converter = sana.geo.Converter(logger.data['mpp'], logger.data['ds'])
            thumbnail = sana.image.Frame(arrs['thumbnail'], level=level, converter=converter)
        else:
            thumbnail = None
        if 'tissue_mask' in arrs:
            tissue_mask = sana.image.frame_like(thumbnail, arrs['tissue_mask'])
        cells = arrs['cells'] \
            if 'cells' in arrs else None
        feature_heatmap = sana.image.Frame(arrs['feature_heatmap']) \
            if 'feature_heatmap' in arrs else None
        arrs.close()

    if args.entrypoint == 'cells' or 'cells' is None:
        # extract the cells from the counterstain
        cells, tissue_mask, thumbnail = segment_nuclei_wsi(
            logger=logger, output_path=output_path,
            **vars(args)
        )
        feature_heatmap = None    
    if args.entrypoint == 'features' or feature_heatmap is None:
        loader = pdnl_sana.slide.Loader(logger, args.input_slide)
        feature_heatmap = aggregate_nuclei_features(
            cells=cells, tb=thumbnail, 
            mpp=logger.data['mpp'], ds=logger.data['ds'], 
            level_dimensions=logger.data['level_dimensions'],
            logger=logger, output_path=output_path,
            **vars(args)
        )
    if args.entrypoint == 'wm' or segmentations is None:
        gm_mask, wm_mask, contours = segment_wm(
            feature_heatmap=feature_heatmap, tissue_mask=tissue_mask, tb=thumbnail, 
            logger=logger, output_path=output_path,
            **vars(args)
        )

if __name__ == "__main__":
    main()
