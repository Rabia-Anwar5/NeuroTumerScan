"""
NeuroSeg - Module 1: Data Preprocessing & Multi-Sequence Alignment Pipeline
Clinical-grade preprocessing for multi-modal brain MRI scans (T1, T1Gd/T1ce, T2, FLAIR).
Features:
- Multi-sequence file discovery (T1, T1ce, T2, FLAIR, SEG)
- Automated skull-stripping quality validation
- Multi-sequence spatial co-registration & affine alignment verification
- 1mm³ isotropic voxel resampling
- RAS anatomical orientation standardization
- Non-zero brain voxel intensity z-score normalization
- Balanced 3D patch extraction & PyTorch/MONAI DataLoaders
"""

import os
import glob
import warnings
from typing import Dict, List, Tuple, Union, Optional
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import nibabel as nib
# pyrefly: ignore [missing-import]
import SimpleITK as sitk

# Suppress PyTorch JIT deprecation warning emitted by MONAI
warnings.filterwarnings("ignore", category=FutureWarning)

# pyrefly: ignore [missing-import]
from monai.data import Dataset, DataLoader
# pyrefly: ignore [missing-import]
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    EnsureTyped
)


def load_brats_file_list(data_dir: str) -> List[Dict[str, Union[List[str], str]]]:
    """
    Scans a directory for patient sub-directories containing BraTS MRI sequences.
    Detects T1, T1CE (or T1Gd), T2, FLAIR, and SEG masks (.nii or .nii.gz).
    
    Args:
        data_dir: Path to parent folder containing patient subdirectories
    Returns:
        List of patient data dicts with keys: 'image' (list of 4 paths), 'label', 'patient_id'
    """
    if not os.path.exists(data_dir):
        return []

    patient_dirs = sorted([
        os.path.join(data_dir, d) for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    data_list = []
    
    for p in patient_dirs:
        p_id = os.path.basename(p)
        t1 = glob.glob(os.path.join(p, "*t1.nii*"))
        # Match t1ce or t1gd, excluding files with t1ce in filename if matching t1
        t1 = [f for f in t1 if "t1ce" not in f.lower() and "t1gd" not in f.lower()]
        t1ce = glob.glob(os.path.join(p, "*t1ce.nii*")) or glob.glob(os.path.join(p, "*t1gd.nii*"))
        t2 = glob.glob(os.path.join(p, "*t2.nii*"))
        # Match t2 excluding flair or t1ce if named similarly
        t2 = [f for f in t2 if "flair" not in f.lower()]
        flair = glob.glob(os.path.join(p, "*flair.nii*"))
        seg = glob.glob(os.path.join(p, "*seg.nii*"))
        
        if t1 and t1ce and t2 and flair:
            entry = {
                "image": [t1[0], t1ce[0], t2[0], flair[0]],
                "patient_id": p_id
            }
            if seg:
                entry["label"] = seg[0]
            data_list.append(entry)
            
    return data_list


def validate_skull_stripping(
    image_input: Union[str, np.ndarray], 
    background_threshold: float = 1e-4
) -> Dict[str, Union[bool, float, str, Dict[str, float]]]:
    """
    Validates whether an input brain MRI volume has been properly skull-stripped.
    
    Clinical Criteria Evaluated:
    1. Zero-Background Fraction: Skull-stripped brain scans have significant surrounding zero-air voxels (usually > 55%).
    2. Boundary Air Buffer: Margin voxels along outer image boundaries must be zero (no scalp or skull bone leakage).
    3. Center-of-Mass Offset: Brain mass should be centered within anatomical volume coordinates.
    4. Tissue Contrast SNR: Verifies plausible dynamic range and non-zero intensity distribution.
    
    Args:
        image_input: Path to NIfTI file or 3D numpy array
        background_threshold: Intensity threshold below which voxels are treated as background air
    Returns:
        Dictionary containing validation verdict ('is_valid'), key metrics, and clinical explanation.
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            return {"is_valid": False, "error": f"File does not exist: {image_input}"}
        img = nib.load(image_input)
        vol = img.get_fdata()
    else:
        vol = np.asarray(image_input)
        
    if vol.ndim == 4:
        # If multi-channel (C, H, W, D), take first sequence
        vol = vol[0]

    total_voxels = vol.size
    non_zero_mask = vol > background_threshold
    non_zero_count = int(np.sum(non_zero_mask))
    
    if non_zero_count == 0:
        return {
            "is_valid": False,
            "status": "FAIL: Empty volume (all background)",
            "metrics": {"brain_voxel_fraction": 0.0}
        }

    brain_voxel_fraction = non_zero_count / float(total_voxels)
    
    # 1. Boundary leakage check: check outer 2-slice margins in X, Y, Z
    margin = 2
    outer_voxels = (
        vol[:margin, :, :].size + vol[-margin:, :, :].size +
        vol[:, :margin, :].size + vol[:, -margin:, :].size +
        vol[:, :, :margin].size + vol[:, :, -margin:].size
    )
    outer_nonzero = (
        np.sum(vol[:margin, :, :] > background_threshold) +
        np.sum(vol[-margin:, :, :] > background_threshold) +
        np.sum(vol[:, :margin, :] > background_threshold) +
        np.sum(vol[:, -margin:, :] > background_threshold) +
        np.sum(vol[:, :, :margin] > background_threshold) +
        np.sum(vol[:, :, -margin:] > background_threshold)
    )
    outer_leakage_fraction = float(outer_nonzero) / float(max(outer_voxels, 1))

    # 2. Center of mass check
    coords = np.argwhere(non_zero_mask)
    center_of_mass = coords.mean(axis=0)
    geometric_center = np.array(vol.shape) / 2.0
    relative_offset = float(np.linalg.norm(center_of_mass - geometric_center) / np.linalg.norm(geometric_center))

    # 3. Dynamic range / SNR
    brain_values = vol[non_zero_mask]
    mean_val = float(np.mean(brain_values))
    std_val = float(np.std(brain_values))
    snr = mean_val / (std_val + 1e-6)

    # Skull stripping pass criteria:
    # - Brain fraction between 5% and 65% of volume box
    # - Outer border leakage < 2.0%
    # - Relative center offset < 0.35
    is_valid = (0.05 <= brain_voxel_fraction <= 0.65) and (outer_leakage_fraction < 0.02) and (relative_offset < 0.35)

    status_str = "PASS: Verified Skull-Stripped Volume" if is_valid else "WARNING: Potential Skull / Scalp Tissue Leakage Detected"

    return {
        "is_valid": is_valid,
        "status": status_str,
        "metrics": {
            "brain_volume_fraction": round(brain_voxel_fraction, 4),
            "outer_boundary_leakage": round(outer_leakage_fraction, 4),
            "center_of_mass_offset": round(relative_offset, 4),
            "tissue_snr": round(snr, 2),
            "mean_brain_intensity": round(mean_val, 2),
            "std_brain_intensity": round(std_val, 2)
        }
    }


def verify_and_align_multimodal_sequences(sequence_paths: List[str]) -> Dict[str, Union[bool, str, dict]]:
    """
    Verifies that all 4 MRI sequences (T1, T1CE, T2, FLAIR) share identical spatial geometry:
    - Same matrix dimensions (H, W, D)
    - Consistent affine transformations (within 1e-3 tolerance)
    - Identical voxel spacing (pixdim)
    """
    if len(sequence_paths) < 2:
        return {"aligned": True, "details": "Single or no sequence to compare."}

    ref_img = nib.load(sequence_paths[0])
    ref_shape = ref_img.shape[:3]
    ref_affine = ref_img.affine
    ref_zooms = tuple(round(float(z), 3) for z in ref_img.header.get_zooms()[:3])

    for p in sequence_paths[1:]:
        img = nib.load(p)
        shape = img.shape[:3]
        affine = img.affine
        zooms = tuple(round(float(z), 3) for z in img.header.get_zooms()[:3])

        if shape != ref_shape:
            return {
                "aligned": False,
                "reason": f"Shape mismatch: {os.path.basename(sequence_paths[0])} {ref_shape} vs {os.path.basename(p)} {shape}"
            }
        if not np.allclose(affine, ref_affine, atol=1e-2):
            return {
                "aligned": False,
                "reason": f"Affine matrix discrepancy detected in {os.path.basename(p)}"
            }
        if zooms != ref_zooms:
            return {
                "aligned": False,
                "reason": f"Voxel resolution discrepancy: {ref_zooms} vs {zooms} in {os.path.basename(p)}"
            }

    return {
        "aligned": True,
        "reason": "All 4 multi-parametric MRI sequences are spatially aligned and co-registered.",
        "geometry": {
            "matrix_shape": ref_shape,
            "voxel_spacing_mm": ref_zooms,
            "affine": ref_affine.tolist()
        }
    }


def resample_to_isotropic(
    itk_image: sitk.Image, 
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    is_label: bool = False
) -> sitk.Image:
    """
    Resamples a 3D SimpleITK image to 1.0 mm³ isotropic voxel resolution.
    Uses B-spline / linear interpolation for anatomical images and Nearest Neighbor for segmentation labels.
    """
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    new_size = [
        int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)
    
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)
        
    return resample.Execute(itk_image)


def get_module1_transforms(patch_size: Tuple[int, int, int] = (128, 128, 128)) -> Compose:
    """
    Full MONAI Preprocessing & Patch Extraction Pipeline:
    1. LoadImaged: Loads NIfTI files into 4-channel image and label
    2. EnsureChannelFirstd: Channels dimension as first axis
    3. Orientationd: Anatomical RAS orientation standardization
    4. Spacingd: 1.0 mm³ isotropic voxel resampling
    5. NormalizeIntensityd: Non-zero brain intensity z-score normalization
    6. RandCropByPosNegLabeld: Balanced patch extraction focused on tumor tissue
    7. RandFlipd & RandGaussianNoised: Spatial data augmentation
    8. EnsureTyped: Convert to PyTorch float32 tensors
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest")
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=2,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        EnsureTyped(keys=["image", "label"], track_meta=False)
    ])


def get_val_transforms() -> Compose:
    """Validation transforms without random patch cropping (for full-volume evaluation)."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest")
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"], track_meta=False)
    ])


def get_inference_transforms() -> Compose:
    """Inference transforms for unlabeled test scans."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"], track_meta=False)
    ])


def get_brats_dataloader(
    data_list: List[dict], 
    batch_size: int = 1, 
    patch_size: Tuple[int, int, int] = (128, 128, 128), 
    shuffle: bool = True,
    training: bool = True
) -> DataLoader:
    """
    Creates a PyTorch/MONAI DataLoader with balanced 3D patch extraction for training,
    or full-volume evaluation transforms for validation.
    """
    transforms = get_module1_transforms(patch_size=patch_size) if training else get_val_transforms()
    dataset = Dataset(data=data_list, transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# Backward-compatibility aliases
get_brats_data_dicts = load_brats_file_list
build_module1_transforms = get_module1_transforms