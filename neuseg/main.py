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

import cv2
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
from .tissue import segment_wm, measure_cortical_angles

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
    wm_mask = None
    layers = None
    print(logger.data)
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
        feature_heatmap = sana.image.frame_like(thumbnail, arrs['feature_heatmap']) \
            if 'feature_heatmap' in arrs else None
        try:
            gm_mask = sana.image.frame_like(thumbnail, arrs['gm_mask']) \
                if 'gm_mask' in arrs else None
            wm_mask = sana.image.frame_like(thumbnail, arrs['wm_mask']) \
                if 'wm_mask' in arrs else None
        except:
            pass
        arrs.close()

    if args.entrypoint == 'cells' or cells is None:
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
    if args.entrypoint == 'wm' or wm_mask is None:
        gm_mask, wm_mask, contours = segment_wm(
            feature_heatmap=feature_heatmap, tissue_mask=tissue_mask, tb=thumbnail, 
            logger=logger, output_path=output_path,
            **vars(args)
        )
    if args.entrypoint == 'cortex' or layers is None:
        gm_mask, wm_mask, cortical_angles = measure_cortical_angles(
            gm_mask=gm_mask, wm_mask=wm_mask, tissue_mask=tissue_mask, 
            tb=thumbnail, cells=cells,
            logger=logger, output_path=output_path,
            **vars(args)
        )
        wm_polys = wm_mask.to_polygons()[0]
        valid = []
        for i in range(cells.shape[0]):
            x,y,area,intensity = cells[i]
            for poly in wm_polys:
                if sana.geo.ray_tracing(x, y, poly*logger.data['ds'][thumbnail.level]):
                    break
            else:
                valid.append(i)
        gm_cells = cells[np.array(valid)]
        print(cells.shape, gm_cells.shape)

        args.window_size = 1000
        cortical_features = aggregate_nuclei_features(cells=gm_cells, tb=thumbnail, 
                                mpp=logger.data['mpp'],
                                ds=logger.data['ds'],
                                level_dimensions=logger.data['level_dimensions'],
                                cortical_angles=cortical_angles.copy(),
                                cortical_mask=gm_mask.copy(),
                                **vars(args),
                                )

        h, w = thumbnail.img.shape[:2]
        fig, axs = plt.subplots(2,3, sharex=True, sharey=True)
        axs = axs.ravel()
        axs[0].imshow(gm_mask.img, cmap='gray')
        axs[1].imshow(wm_mask.img, cmap='gray')
        axs[2].imshow(tissue_mask.img, cmap='gray')
        axs[3].imshow(cortical_angles.img, cmap='gray')
        axs[3].set_title("Tangent Angle of WM Segmentation")
        axs[4].imshow(feature_heatmap.img[:,:,0], cmap='inferno', extent=(0,w,h,0))
        axs[4].set_title("Density used for WM Seg")
        axs[5].imshow(cortical_features.img[:,:,0], cmap='inferno', extent=(0,w,h,0), vmax=0.001)
        axs[5].set_title("Density used for Layer Seg")

        fig, axs = plt.subplots(2,3, sharex=True, sharey=True)
        titles = ['Density', 'Area', 'Intensity']
        for i in range(3):
            if i < 2:
                axs[0][i].imshow(feature_heatmap.img[:,:,i], cmap='inferno')
                axs[0][i].set_title(titles[i])
                [axs[0][i].plot(*x.T / args.ds_thumbnail, color='white') for x in wm_polys]
            #axs[1][i].imshow(cortical_features.img[:,:,i], cmap='inferno')
            x = cortical_features.img[:,:,i]
            mu, sg = np.nanmean(x[x != 0]), np.nanstd(x[x != 0])
            x = np.clip(x, mu-1*sg, mu+3*sg)
            axs[1][i].imshow(x, cmap='inferno')
            axs[1][i].set_title(titles[i])
            [axs[1][i].plot(*x.T / args.ds_thumbnail, color='white') for x in wm_polys]


        plt.show()

if __name__ == "__main__":
    main()
