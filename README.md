Here's a description for a GitHub README that includes the process and links, formatted appropriately:

---

## Description

This repository contains the code used in the paper, which focuses on the classification of heart sound signals using multiple features. The dataset used for this study can be found [here](https://github.com/yaseen21khan/Classification-of-Heart-Sound-Signal-Using-Multiple-Features).

### Process Overview

The following process was followed in the study:

1. **Denoising**: 
   - `denoise_sound_variousmethod_0.m`: This script applies various methods to denoise heart sound signals, preparing them for further analysis.

2. **Feature Extraction**:
   - `generate_stft_2.m`: This script generates Short-Time Fourier Transform (STFT) images from the denoised signals, which are used as features for model training.

3. **Denoising with GAN and U-Net**:
   - `Unet_for_denoisygan.ipynb` and `Unet_for_denoisy.ipynb`: These notebooks implement U-Net architectures for denoising, with one utilizing a Generative Adversarial Network (GAN) framework for enhanced denoising capability.

4. **Combining STFTs**:
   - `combainestft_inversestft_4.m`: This script combines multiple STFT images and performs an inverse STFT to reconstruct the denoised signal.

5. **Merging Spectrograms**:
   - `2.merge_allspecto_one.ipynb`: This notebook merges all spectrograms into a single dataset, preparing it for classification.

6. **Classification**:
   - `4.Model_classify_noisy_denoisy.ipynb`: This notebook contains the classification model used to classify noisy and denoised heart sound signals, evaluating the model's performance on various metrics.

---

This structured overview should help users understand the workflow and how each part of the code contributes to the overall process described in the paper. You can further expand on each step with more details specific to your implementations.
