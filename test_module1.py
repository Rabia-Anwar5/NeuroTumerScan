"""
NeuroSeg - Module 1 Verification Suite:
Tests Data Preprocessing, Multi-Sequence Alignment, Skull-Stripping Validation,
Isotropic Resampling, and Balanced 3D Patch Extraction.
"""

import os
import sys
# pyrefly: ignore[missing-import]
import numpy as np
# pyrefly: ignore[missing-import]
import nibabel as nib
# pyrefly: ignore[missing-import]
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from src.dataset_pipeline import (
    load_brats_file_list,
    validate_skull_stripping,
    verify_and_align_multimodal_sequences,
    get_module1_transforms,
    get_brats_dataloader
)
# pyrefly: ignore[missing-import]
from monai.data import Dataset, DataLoader


def test_module1():
    print("=" * 65)
    print("  VERIFYING MODULE 1: PREPROCESSING & DATA PIPELINE")
    print("=" * 65)

    test_dir = os.path.join(PROJECT_ROOT, "data", "BraTS2021_Sample", "BraTS2021_00001")
    os.makedirs(test_dir, exist_ok=True)
    affine = np.eye(4)
    shape = (140, 140, 140)

    # 1. Synthesize realistic skull-stripped multimodal scan
    z, y, x = np.ogrid[:140, :140, :140]
    brain_mask = (((x - 70) / 48)**2 + ((y - 70) / 52)**2 + ((z - 70) / 44)**2) <= 1.0

    seq_paths = []
    for seq in ["t1", "t1ce", "t2", "flair"]:
        fpath = os.path.join(test_dir, f"BraTS2021_00001_{seq}.nii.gz")
        seq_paths.append(fpath)
        
        vol = np.zeros(shape, dtype=np.float32)
        noise = (np.random.randn(int(np.sum(brain_mask))) * 18 + 100).astype(np.float32)
        noise[noise < 0] = 0
        vol[brain_mask] = noise
        nib.save(nib.Nifti1Image(vol, affine), fpath)

    seg_path = os.path.join(test_dir, "BraTS2021_00001_seg.nii.gz")
    mask = np.zeros(shape, dtype=np.uint8)
    mask[55:85, 55:85, 55:85] = 1   # Necrotic core
    mask[45:95, 45:95, 45:95] = 2   # Edema
    mask[65:80, 65:80, 65:80] = 4   # Enhancing margin
    mask = mask * brain_mask.astype(np.uint8)  # Confine inside brain
    nib.save(nib.Nifti1Image(mask, affine), seg_path)

    print(" [1/5] Skull-stripped multimodal scan files prepared.")

    # 2. Test Skull-Stripping Validation
    skull_check = validate_skull_stripping(seq_paths[0])
    print(f" [2/5] Skull-Stripping Validation: {skull_check['status']}")
    print(f"       Brain Volume Fraction:    {skull_check['metrics']['brain_volume_fraction'] * 100:.1f}%")
    print(f"       Outer Boundary Leakage:   {skull_check['metrics']['outer_boundary_leakage'] * 100:.2f}%")
    assert skull_check["is_valid"], f"Skull-stripping validation check failed: {skull_check}"

    # 3. Test Spatial Co-Registration & Affine Alignment
    alignment_check = verify_and_align_multimodal_sequences(seq_paths)
    print(f" [3/5] Spatial Co-Registration:   {alignment_check['reason']}")
    assert alignment_check["aligned"], "Sequence co-registration alignment failed."

    # 4. Test DataLoader & Balanced 3D Patch Extraction
    file_list = load_brats_file_list(os.path.join(PROJECT_ROOT, "data", "BraTS2021_Sample"))
    assert len(file_list) > 0, "No cases found by loader."
    
    loader = get_brats_dataloader(file_list, batch_size=1, patch_size=(128, 128, 128), shuffle=False)
    batch = next(iter(loader))
    img_patch = batch["image"][0].numpy()
    seg_patch = batch["label"][0, 0].numpy()

    print(f" [4/5] Extracted 3D Patch (C, H, W, D): {img_patch.shape}")
    print(f"       Label Patch Shape (H, W, D):    {seg_patch.shape}")
    assert img_patch.shape == (4, 128, 128, 128), f"Unexpected patch shape: {img_patch.shape}"
    assert seg_patch.shape == (128, 128, 128), f"Unexpected label patch shape: {seg_patch.shape}"

    # 5. Save Verification Visual
    outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    out_plot = os.path.join(outputs_dir, "Module_1_Verification.png")

    slice_z = 64
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    titles = ["T1 (Anatomy)", "T1CE (Active Core)", "T2 (Edema)", "FLAIR (Infiltration)", "Ground Truth Overlay"]
    for i in range(4):
        axes[i].imshow(img_patch[i, :, :, slice_z], cmap="gray")
        axes[i].set_title(titles[i], fontsize=10, fontweight="bold")
        axes[i].axis("off")

    axes[4].imshow(img_patch[1, :, :, slice_z], cmap="gray")
    axes[4].imshow(seg_patch[:, :, slice_z], cmap="tab10", alpha=0.55)
    axes[4].set_title(titles[4], fontsize=10, fontweight="bold")
    axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(out_plot, dpi=200)
    plt.close(fig)
    print(f" [5/5] Verification visual saved to: {out_plot}")
    print(" >>> MODULE 1 VERIFICATION PASSED <<<\n")


if __name__ == "__main__":
    test_module1()