# NEUSEG (Hitopathology WSI GM/WM Segmentation)
**Interpretable Unsupervised GM/WM Segmentation in Brain Histopathology Using Nuclei Morphometrics**

NEUSEG is a **fully automated, unsupervised, and interpretable pipeline** for gray matter (GM) and white matter (WM) segmentation in brain histopathology whole-slide images (WSIs). The method bridges **cellular-scale nuclei morphometrics** with **tissue-scale segmentation**, enabling robust GM/WM delineation across heterogeneous stains, cortical regions, and neurodegenerative pathologies without requiring training data or GPUs.

---

## Motivation

Accurate GM/WM segmentation is a critical prerequisite for quantitative analysis of brain histopathology, particularly for studying neurodegenerative disease progression. However, WSIs pose several challenges:

- Gigapixel-scale image resolution  
- Large variability in staining, scanners, tissue preparation, and pathology  
- Labor-intensive and subjective manual annotation  
- Limited generalization and interpretability of supervised deep learning approaches  

NEUSEG addresses these challenges by providing a **lightweight, CPU-operable, and biologically grounded alternative** to supervised CNN-based pipelines.

---

## Key Contributions

- **Unsupervised GM/WM segmentation** using nuclei size and density, requiring no annotated training data  
- **Interpretable feature design**, grounded in known cytoarchitectural differences between GM and WM  
- **Robust performance under domain shift**, outperforming a supervised CNN baseline on out-of-distribution slides  
- **Scalable CPU-only implementation**, processing WSIs in ~1–2 minutes per slide  
- **Extensive validation** across 252 WSIs spanning multiple stains, cortical regions, and neurodegenerative pathologies  

---

## Method Overview
![NEUSEG pipeline](figures/Figure1.png)
NEUSEG processes each WSI independently using the following steps:

1. **Tissue Extraction**  
   - Hematoxylin channel isolation via color deconvolution  
   - Global thresholding to separate tissue from background  

2. **Nuclei Segmentation & Feature Extraction**  
   - **Enhance nuclear contrast (pre-processing):** apply **triangular thresholding** to the **hematoxylin channel** within the tissue mask to generate an intensity map that highlights nuclei (neurons + glia)  
   - **Nuclei instance segmentation:** perform **watershed-based segmentation** on non-overlapping patches  
   - **Morphometric feature extraction:** compute  
     - Nuclear size  
     - Local nuclear density


3. **Feature Aggregation**  
   - Gaussian smoothing to generate spatial maps of nuclei size and density at lower resolution  

4. **Unsupervised Clustering**  
   - Two-component Gaussian Mixture Model (GMM) fit to morphometric features  
   - GM and WM labels assigned using biologically informed rules  

5. **Spatial Refinement**  
   - Conditional Random Field (CRF) smoothing  
   - Morphological post-processing to remove artifacts and small disconnected regions  

6. **Contour Extraction**  
   - GM–WM, GM–background, and WM–background boundaries for evaluation and downstream analysis
---
<!--
## Experimental Evaluation

### Datasets

- **WSI-level annotations**: 13 fully annotated slides (FTLD-Tau and FTLD-TDP)  
- **ROI-based evaluation**: 252 WSIs with expert-defined cortical ROIs  
- **Stains & markers**: AT8, TDP-43, GFAP, SMI94, parvalbumin  
- **Pathologies**: FTLD-Tau, FTLD-TDP, Alzheimer’s disease, ALS, PART, and controls  

### Baseline Comparison

NEUSEG was compared against **BrainSec**, a supervised CNN-based GM/WM segmentation method.

- Comparable accuracy on in-distribution slides  
- Substantially improved robustness under scanning artifacts  
- Superior generalization on out-of-distribution slides  

### ROI-Based Accuracy

- Median annotation-to-contour distances:  
  - **GM–background**: ~33 µm  
  - **GM–WM**: ~130 µm  
- Near-perfect agreement in percent area occupied (%AO) between expert-annotated and NEUSEG-derived ROIs (Pearson r ≈ 0.99)  

### Runtime & Scalability

- ~85 seconds per WSI on a CPU-only system  
- Efficient parallelization across large cohorts  

---
-->
## Why NEUSEG?

- ✅ No training data required  
- ✅ Interpretable and biologically grounded  
- ✅ Robust to staining and pathology variability  
- ✅ Scales efficiently to hundreds of WSIs  
- ✅ Suitable for downstream quantitative pathology analyses  

NEUSEG is particularly well-suited for large-scale studies where **reproducibility, robustness, and interpretability** are essential.

---

## Code Usage

This repository provides an end-to-end pipeline for **unsupervised GM/WM segmentation** from brain histopathology whole-slide images (WSIs) using nuclei morphometrics.

### Core Scripts

- **[`script/Nuclei_Segmentation.py`](script/Nuclei_Segmentation.py)**  
  Performs nuclei segmentation from histopathology WSIs. This step generates nuclei masks and extracts morphometric features (e.g., nuclear size and local nuclear density), which are used as inputs for downstream tissue segmentation.

- **[`script/GMM_Segmentation.py`](script/GMM_Segmentation.py)**  
  Performs GM/WM segmentation using a Gaussian Mixture Model (GMM) based on features derived from the nuclei segmentation results.

- **[`script/neuseg_script.sh`](script/neuseg_script.sh)**  
  Wrapper shell script that runs the full NEUSEG pipeline, including nuclei segmentation followed by GMM-based GM/WM segmentation.

### Running the Pipeline

1. **Prepare input data**  
   Place the `.svs` whole-slide images to be processed inside the **[`Data/`](Data/)** directory:
```text
NEUSEG/
├── Data/
│   └── *.svs
```
2. **Run NEUSEG**
From the root of the repository, first set up the Python environment and install all required dependencies.

   2-1. **Create and activate the conda environment**
   ```
   conda create -n neuseg python=3.12 -y
   conda activate neuseg
   ```

   2-2. **Clone the NEUSEG repository and install dependencies**
   ```
   git clone https://github.com/PennCompPathology/NEUSEG.git
   cd NEUSEG
   python -m pip install -r requirements.txt
   ```
   **[`requirements.txt`](requirements.txt)** is a single file covering all three
   components: the core pipeline, the annotator, and the evaluation notebook.
   It is grouped into labelled sections so you can see which package belongs to
   which component, but everything installs in one command.

   2-3. **Install the SANA dependency (required)**
   ```
   git clone https://github.com/penndigitalneuropathlab/sana.git
   cd sana
   python -m pip install -r src/pdnl_sana/requirements.txt
   python -m pip install -e .
   ```
   Run the following command to confirm that sana was installed correctly:
   ```
   python3 -c "import pdnl_sana.image; import pdnl_sana.slide"
   ```
   To confirm it is **this clone** that is active, and not the PyPI build that
   step 2-2 pulls in as a transitive dependency of `pdnl_extract`:
   ```
   python3 -c "import pdnl_sana, os; print(os.path.dirname(pdnl_sana.__file__))"
   ```
   The printed path should point inside the `sana/` directory you just cloned.
   An environment only ever holds one `pdnl_sana`, so the editable install
   above supersedes the PyPI copy; the `pdnl_extract` / `pdnl_process` /
   `pdnl_aggregate` tools used by the evaluation notebook resolve to this
   clone as well.
   
   2-4. **Run the NEUSEG pipeline**

   The pipeline is driven by **[`neuseg/main.py`](neuseg/main.py)**, which processes
   **one WSI per invocation**. From the root of the repository:

   ```
   python neuseg/main.py \
     -i Data/slide.svs \
     -o GM_WM_Seg_Results/slide \
     --n_cores 8 \
     --debug_level debug
   ```

   **Required arguments**

   | Flag | Description |
   |---|---|
   | `-i`, `--input_slide` | Path to the input WSI (`.svs`, or any format OpenSlide can read). One file per run; to process a cohort, loop over slides or use **[`neuseg/run_cohort.py`](neuseg/run_cohort.py)**. |
   | `-o`, `--output_directory` | Directory that receives every output for this slide (see **Outputs** below). Created automatically if it does not exist. Use **one directory per slide**: the output filenames are fixed, so slides sharing a directory would overwrite each other. |

   **Frequently used options**

   | Flag | Default | Description |
   |---|---|---|
   | `--n_cores` | `1` | Number of worker processes used to segment nuclei and aggregate features. Set it to the number of cores you want to occupy; runtime for the nuclei stage scales close to linearly. |
   | `--debug_level` | `normal` | Console verbosity, and whether diagnostic figures are written. `quiet` → errors only; `normal` → progress bars and warnings; `debug` → adds per-stage parameters and statistics; `full` → maximum verbosity. **`debug` and `full` additionally write the four diagnostic PNGs** listed under **Outputs**. |
   | `--entrypoint` | `cells` | Which stage to start from, reusing the intermediate files already in `-o`. See below. |

   **Stages and `--entrypoint`**

   The pipeline runs in four stages, each writing intermediate files that the next
   one reads. `--entrypoint` skips straight to a stage instead of recomputing
   everything, which makes tuning the segmentation cheap: the `cells` stage
   dominates runtime, while `cortex` takes seconds.

   | Value | Starts at | Requires already in `-o` |
   |---|---|---|
   | `cells` *(default)* | Nuclei segmentation over the whole WSI (the full pipeline) | nothing |
   | `features` | Aggregating nuclei into the feature heatmaps | `cells.npy`, `thumbnail.png` |
   | `tissue` | Tissue-mask stage | `feature_heatmap.npy` |
   | `cortex` | Tissue mask → GMM → CRF → contours | `feature_heatmap.npy`, `thumbnail.png` |

   So to re-run only the GM/WM segmentation after an initial full run, for example
   to inspect the diagnostic figures, reuse the existing outputs:

   ```
   python neuseg/main.py \
     -i Data/slide.svs \
     -o GM_WM_Seg_Results/slide \
     --entrypoint cortex \
     --debug_level debug
   ```

   **Remaining options**

   | Flag | Default | Description |
   |---|---|---|
   | `--window_size` | `1000.0` | Radius (µm) over which nuclei are aggregated into the feature heatmaps. Cells are Gaussian-weighted with σ = `window_size` / 5. |
   | `--ds_thumbnail` | `1.0` | Resolution of the feature heatmap relative to the thumbnail. `1.0` computes it at full thumbnail resolution; larger values downsample it. |
   | `--frame_size` | `1024` | Size (px) of the chunks the WSI is read in. Affects memory use and I/O only, not results. |
   | `--tmp_directory` | *(auto)* | Location for intermediate chunk files. Defaults to a temporary directory that is deleted on exit; set this to keep them. |

3. **Outputs**

   All results are written into the directory given by `-o`, one directory per
   slide. Which files appear depends on `--entrypoint`; a default full run
   produces all of them.

   **Data files** (always written)

   | File | Stage | Contents |
   |---|---|---|
   | `thumbnail.png` | `cells` | Low-resolution RGB thumbnail of the WSI. Every later stage works on this grid, and the final masks are returned at this resolution. |
   | `tissue_mask.npy` | `cells`, then `cortex` | Tissue mask at thumbnail resolution. The `cells` stage writes a preliminary mask, used only to skip background while segmenting nuclei. The `cortex` stage then **overwrites it** with the refined mask from [`tissue_extraction.py`](neuseg/tissue_extraction.py), so the file always holds the mask the GM/WM segmentation was actually run against. |
   | `cells.npy` | `cells` | `(N, 4)` array, one row per segmented nucleus: `x`, `y` (level-0 slide pixels), soma area, and mean hematoxylin intensity. |
   | `feature_heatmap.npy` | `features` | `(H, W, 3)` array of the aggregated morphometric maps: channel `0` soma density, `1` average soma size, `2` average soma intensity. |
   | `gm_mask.npy` | `cortex` | `(h, w)` boolean gray matter mask at thumbnail resolution. |
   | `wm_mask.npy` | `cortex` | `(h, w)` boolean white matter mask; together with `gm_mask.npy` it partitions the tissue. |
   | `gmwm_contours.json` | `cortex` | Boundary polygons in thumbnail pixel coordinates: `{"shape": [h, w], "gm_wm": [...], "gm_csf": [...]}`, where each boundary is a list of `[[x, y], ...]` rings. |

   **Diagnostic figures** (only when `--debug_level` is `debug` or `full`)

   | File | Stage | What it shows |
   |---|---|---|
   | `run_features_output.png` | `features` | Four panels: the segmented nuclei plotted over the thumbnail, then the three feature maps (soma density, average soma size, average soma intensity). The title records `window_size` in µm, level-0 and thumbnail pixels, the Gaussian σ used to smooth them, and `ds_thumbnail`. |
   | `gmm_result.png` | `cortex` | Eight panels tracing the GM/WM fit from raw cells to labels. **Top row:** thumbnail, segmented nuclei, and the density and size maps restricted to tissue, with pixels excluded from the fit marked in cyan. **Bottom row:** the resulting GM/WM labels over the thumbnail, then the standardized feature space as a scatter, coloured by fitted component with each component's mean and 1/2/3-σ ellipses, and again coloured by raw density and raw size. Useful for confirming the two components really separate. |
   | `post_process.png` | `cortex` | Three panels showing spatial refinement step by step: the raw GMM labels, the result after CRF smoothing, and the result after small disconnected regions are pruned. Pixels changed by each step are marked in red and counted in the legend. |
   | `GMWM_contour.png` | `cortex` | The final segmentation: the thumbnail with GM and WM tinted, overlaid with the three extracted boundaries: GM–WM (green), GM–background (magenta), and WM–background (black). |

   A plain-text run log is also written to `NEUSEG.log`, in the **current working
   directory** rather than in `-o`.

## Citation
[NEUSEG: Interpretable Unsupervised Gray/White Matter Segmentation for Brain Histopathology WSIs](https://ieeexplore.ieee.org/document/11515902)
*IEEE ISBI 2026*

If you use NEUSEG in your work, please cite:
```text
Roh, Hyung Seok, et al. "NEUSEG: Interpretable Unsupervised GM/WM Segmentation in Brain Histopathology Using Nuclei Morphometrics." 2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI). IEEE, 2026.
```

---

## Acknowledgments

This work was supported by the National Institutes of Health and institutional funding at the University of Pennsylvania.

