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
Module 2 configures the 3D deep learning segmentation backbone (SegResNet) and composite loss function to handle multi-sequence MRI inputs and severe sub-region voxel class imbalance.

## Architecture Steps
1. **3D SegResNet Backbone:** Initialized an encoder-decoder architecture with 4 input channels (T1, T1CE, T2, FLAIR) and 3 output channels (WT, TC, ET).
2. **Sub-Region Class Remapping:** Converted raw multi-class integer labels into 3 clinical evaluation channels: Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET).
3. **Composite Loss Formulation:** Implemented multi-label `DiceFocalLoss` combining boundary Dice Loss and Focal Cross-Entropy to handle foreground-background class imbalance.
4. **Pipeline Verification:** Validated the forward and backward passes on $128 \times 128 \times 128$ voxel patches, confirming output shape `torch.Size([1, 3, 128, 128, 128])` and gradient backpropagation.

# NeuroSeg - Module 3: Training & Validation Loop

## Overview
Module 3 establishes the end-to-end training and validation engine, integrating multi-label optimization, sliding-window inference, sub-region Dice metric evaluation, and model checkpointing.

## Pipeline Steps
1. **Optimization Pipeline:** Adam optimizer with weight decay coupled with Cosine Annealing learning rate scheduling.
2. **Sliding-Window Inference:** Evaluates full 3D volumes using overlapping $128 \times 128 \times 128$ patches with Gaussian blending to eliminate boundary artifacts.
3. **Sub-Region Metric Evaluation:** Tracks validation Dice scores across Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET) sub-regions using `DiceMetric`.
4. **Checkpoint Management:** Automatically persists top-performing weights (`best_segresnet_metric.pth`) based on mean validation Dice.
5. **Verification Suite:** Validates training step backward pass, sliding-window inference, metric aggregation, and overlay visualization in `test_module3.py`.