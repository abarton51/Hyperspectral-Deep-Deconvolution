# Single-shot Hyperspectral Imaging via Deep Chromatic Aberration Deconvolution

### Contributors: Corey Zheng, Austin Barton, Mohammad Taher, Abdulaziz Memesh

## Overview
This work addresses the challenge of chromatic aberration in snapshot hyperspectral imaging, where the introduction of a third spectral dimension complicates encoding information into a two-dimensional detector plane. While traditional methods rely on scan-based approaches, our proposed method aims to enhance the quality of hyperspectral images by mitigating distortions inherent in snapshot acquisitions by leveraging a blind deconvolution approach with a U-Net neural network architecture for single-shot hyperspectral imaging. Our approach allows real-time correction without the need for complex scanning mechanisms nor knowledge of point spread functions. Through experimental validation, we demonstrate the efficacy of our method in preserving image structure and spectral composition information, contributing to improved imaging throughput and simplified hardware requirements. Our best performing model is capable of restoring spectral information among test data, even in pixel locations with highly varying wavelength intensities, while deblurring and restoring an approximation of the latent sharp image. Our paper represents a simple yet effective approach to circumventing issues in snapshot hyperspectral imaging, providing a practical solution for applications in medical imaging, agriculture, materials identification, and geological surveillance.

## Setup
**Option One (Recommended)**:
- Open Anaconda Prompt 
- Navigate to `src`. Then, use the following command to create the environment: `conda env create -n env_name -f environment.yaml`
- Activate the environment with: `conda activate env_name`
- Verify installation with: `conda list`

**Option Two**:
- Open Command Prompt and run the following command in the repository directory: `pip install -r requirements.txt`
- This also works for virtual environments but isn't recommended
- Verify installation with: `pip list`

## Directory

### src
-  Data synthesization.
-  Baseline method (RL Deconvolution).
-  Data loading.
-  Model creation (U-Nets).
-  Training and evaluation.
-  Figure generation.
-  Pixel comparison.

### overleaf
- Manuscript TeX files.

### figs
- Highlighted figures including:
  - Data synthesis.
  - Point spread functions.
  - Optical system.
  - Data flowchart.
  - Best results.
  - Mode collapse results.
  - Loss curves.
  - Model architecture.

## Acknowledgements
This project is done as part of CS 7643 Deep Learning at Georgia Tech. We would like to directly thank Professor Danfei Xu and all of the teaching faculty for CS 7643 Deep Learning, Fall 2023 at Georgia Tech.
