# Single-shot Hyperspectral Imaging via Deep Chromatic Aberration Deconvolution
CS 4644 Deep Learning at Georgia Tech Course Project

### Contributors: Corey Zheng, Austin Barton, Mohammad Taher, Abdulaziz Memesh

## Overview
This work addresses the challenge of chromatic aberration in snapshot hyperspectral imaging, where the introduction of a third spectral dimension complicates encoding information into a two-dimensional detector plane. While traditional methods rely on scan-based approaches, our proposed method aims to enhance the quality of hyperspectral images by mitigating distortions inherent in snapshot acquisitions by leveraging a blind deconvolution approach with a U-Net neural network architecture for single-shot hyperspectral imaging. Our approach allows real-time correction without the need for complex scanning mechanisms nor knowledge of point spread functions. Through experimental validation, we demonstrate the efficacy of our method in preserving image structure and spectral composition information, contributing to improved imaging throughput and simplified hardware requirements. Our best performing model is capable of restoring spectral information among test data, even in pixel locations with highly varying wavelength intensities, while deblurring and restoring an approximation of the latent sharp image. Our paper represents a simple yet effective approach to circumventing issues in snapshot hyperspectral imaging, providing a practical solution for applications in medical imaging, agriculture, materials identification, and geological surveillance.

## Directory

### src
-  Training and evaluating models.
-  Data synthesization.
-  Figure analysis.
-  Data loading. 

### overleaf
- Paper creation.

### figs
- Highlighted figures from model training, evaluation, and analysis of results.
