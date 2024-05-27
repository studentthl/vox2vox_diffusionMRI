# vox2vox_diffusionMRI

This project utilizes a 3D volume-to-volume Generative Adversarial Network (GAN) for the denoising (reconstruction) of q-space diffusion MRI images. The base model for our approach is the Vox2Vox model, which is traditionally used for generating realistic segmentation outputs from multi-channel 3D MR images.

Key Features:
Model Used: Vox2Vox, a GAN model with a U-Net based generator and a PatchGAN discriminator.
Task: Denoising and reconstructing 3D medical images, specifically transforming undersampled q-space images into fully sampled counterparts.
Performance Metrics: Evaluation using Mean Squared Error (MSE) and Structural Similarity Index (SSIM).

Based on original model: https://github.com/enochkan/vox2vox
paper: https://www.diva-portal.org/smash/get/diva2:1540853/FULLTEXT01.pdf
