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
   components — the core pipeline, the annotator, and the evaluation notebook.
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
   above supersedes the PyPI copy — the `pdnl_extract` / `pdnl_process` /
   `pdnl_aggregate` tools used by the evaluation notebook resolve to this
   clone as well.
   
   2-4. **Run the NEUSEG pipeline**

   > ⚠️ **Under active development.** The pipeline is currently being refactored
   > from the standalone scripts in **[`script/`](script/)** into the installable
   > **[`neuseg/`](neuseg/)** package, and the run instructions are being revised
   > along with it. This section will be updated once the refactor is complete.

3. **Outputs**

   Once the pipeline is runnable again, results will be written to
   **GM_WM_Seg_Results/** — the final **gray matter (GM) / white matter (WM)
   segmentation** produced from the whole-slide images (WSIs) placed in
   **[`Data/`](Data/)**.

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

