# NEUSEG (Hitopathology WSI GM/WM Segmentation)
**Interpretable Unsupervised GM/WM Segmentation in Brain Histopathology Using Nuclei Morphometrics**

NEUSEG is a **fully automated, unsupervised, and interpretable pipeline** for gray matter (GM) and white matter (WM) segmentation in brain histopathology whole-slide images (WSIs). The method bridges **cellular-scale nuclei morphometrics** with **tissue-scale segmentation**, enabling robust GM/WM delineation across heterogeneous stains, cortical regions, and neurodegenerative pathologies—without requiring training data or GPUs.

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

NEUSEG processes each WSI independently using the following steps:

1. **Tissue Extraction**  
   - Hematoxylin channel isolation via color deconvolution  
   - Global thresholding to separate tissue from background  

2. **Nuclei Segmentation & Feature Extraction**  
   - Watershed-based nuclei segmentation on non-overlapping patches  
   - Extraction of two morphometric features:  
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

## Why NEUSEG?

- ✅ No training data required  
- ✅ Interpretable and biologically grounded  
- ✅ Robust to staining and pathology variability  
- ✅ Scales efficiently to hundreds of WSIs  
- ✅ Suitable for downstream quantitative pathology analyses  

NEUSEG is particularly well-suited for large-scale studies where **reproducibility, robustness, and interpretability** are essential.

---

## Availability

- **Paper**: ISBI 2026 submission  
- **Code**: https://github.com/HyungSeokRoh/NEUSEG  
- **Implementation**: Python (CPU-only)

---

## Citation

If you use NEUSEG in your work, please cite:
Roh, H. S., Capp, N., Ohm, D. T., Irwin, D. J., Gee, J. C., & Chen, M.
NEUSEG: Interpretable Unsupervised GM/WM Segmentation in Brain Histopathology Using Nuclei Morphometrics.


---

## Acknowledgments

This work was supported by the National Institutes of Health and institutional funding at the University of Pennsylvania.

