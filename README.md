# NeuroSeg - Module 1: Preprocessing & Data Pipeline

## Overview
Module 1 transforms multimodal 3D brain MRI volumes (T1, T1CE, T2, FLAIR) into standardized tensors ($4 \times 128 \times 128 \times 128$) with aligned ground-truth segmentation masks.

## Pipeline Steps
1. **Multimodal Alignment:** Reoriented all volumes to standard RAS orientation.
2. **Resampling:** Uniform 1.0 mm³ isotropic voxel resolution.
3. **Z-Score Normalization:** Standardized intensity on non-zero brain voxels per sequence.
4. **Patch Generation:** Extracted $128 \times 128 \times 128$ balanced positive/negative patches.

# NeuroSeg - Module 2: 3D Model Architecture & Loss Setup

## Overview
Module 2 configures the 3D deep learning segmentation backbone (SegResNet) and composite loss function to handle multi-sequence MRI inputs and severe sub-region voxel class imbalance[cite: 6, 8, 18].

## Architecture Steps
1. **3D SegResNet Backbone:** Initialized an encoder-decoder architecture with 4 input channels (T1, T1CE, T2, FLAIR) and 3 output channels (WT, TC, ET)[cite: 6, 8, 18].
2. **Sub-Region Class Remapping:** Converted raw multi-class integer labels into 3 clinical evaluation channels: Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET)[cite: 18].
3. **Composite Loss Formulation:** Implemented multi-label `DiceFocalLoss` combining boundary Dice Loss and Focal Cross-Entropy to handle foreground-background class imbalance[cite: 6, 8].
4. **Pipeline Verification:** Validated the forward and backward passes on $128 \times 128 \times 128$ voxel patches, confirming output shape `torch.Size([1, 3, 128, 128, 128])` and gradient backpropagation[cite: 18].