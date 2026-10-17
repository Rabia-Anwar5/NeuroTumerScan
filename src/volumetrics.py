"""
NeuroSeg - Module 4: Morphological Post-Processing, Volumetrics & RANO Criteria Engine
Implements:
- 3D Connected-component analysis for noise and false-positive island filtering
- Anatomical hierarchical consistency enforcement (ET in TC in WT)
- Physical volumetric quantification in mm³ and mL / cm³
- Derivation of Peritumoral Edema (WT - TC) and Necrotic Core (TC - ET)
- Automated clinical RANO (Response Assessment in Neuro-Oncology) criteria calculations:
  * Maximal perpendicular diameters (D1 x D2 in mm)
  * Sum of Products of Diameters (SPD in mm²)
  * Measurable lesion classification
- Clinical text and tabular report formatting
"""

from typing import Dict, Tuple, Optional, Any
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import scipy.ndimage as ndi
import pandas as pd


def remove_spurious_islands(binary_mask: np.ndarray, min_size_voxels: int = 50) -> np.ndarray:
    """
    Applies 3D connected-component labeling (26-connectivity structuring)
    to eliminate isolated false-positive noise clusters smaller than the threshold.
    
    Args:
        binary_mask: 3D binary array (H, W, D) of type bool or int
        min_size_voxels: Minimum connected volume threshold to preserve
    Returns:
        Cleaned 3D binary mask (uint8)
    """
    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=np.uint8)

    # 26-connectivity 3x3x3 structuring element
    structure = ndi.generate_binary_structure(3, 3)
    labeled_array, num_features = ndi.label(binary_mask, structure=structure)
    
    if num_features == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    # Measure voxel count of each connected cluster
    component_sizes = ndi.sum(binary_mask, labeled_array, range(1, num_features + 1))
    
    # Keep-table indexed by component id (0 = background): O(volume) instead of
    # O(volume x components), which matters for noisy predictions with thousands of islands.
    keep = np.zeros(num_features + 1, dtype=bool)
    keep[1:] = np.asarray(component_sizes) >= min_size_voxels
    cleaned_mask = keep[labeled_array]
            
    return cleaned_mask.astype(np.uint8)


def enforce_hierarchical_consistency(wt_mask: np.ndarray, tc_mask: np.ndarray, et_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Enforces biological nested anatomical hierarchy:
    ET (Enhancing Tumor) must be inside TC (Tumor Core),
    and TC must be inside WT (Whole Tumor).
    ET subset of TC subset of WT
    """
    # 1. Cleaned TC must be contained in WT
    tc_consistent = tc_mask & wt_mask
    # 2. Cleaned ET must be contained in TC
    et_consistent = et_mask & tc_consistent
    
    return wt_mask, tc_consistent, et_consistent


def compute_rano_criteria_metrics(
    target_mask_3d: np.ndarray, 
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Dict[str, Any]:
    """
    Automated RANO (Response Assessment in Neuro-Oncology) measurement:
    Finds the axial slice with the largest tumor cross-sectional surface area,
    computes maximal diameter (D1) and the largest perpendicular diameter (D2),
    and calculates the Product of Diameters (D1 x D2 in mm²).
    
    Args:
        target_mask_3d: 3D binary array of the target lesion (typically ET or TC)
        voxel_spacing: (sx, sy, sz) in mm (e.g. 1.0, 1.0, 1.0)
    Returns:
        Dictionary with D1 (mm), D2 (mm), product (mm²), slice index, and measurable flag
    """
    if not np.any(target_mask_3d):
        return {
            "max_diameter_d1_mm": 0.0,
            "perp_diameter_d2_mm": 0.0,
            "product_spd_mm2": 0.0,
            "best_slice_idx": 0,
            "is_measurable": False,
            "rano_category": "Non-Measurable (No enhancing lesion detected)"
        }

    # Sum voxels per axial slice (dim 2)
    axial_slice_counts = np.sum(target_mask_3d, axis=(0, 1))
    best_slice_idx = int(np.argmax(axial_slice_counts))
    slice_2d = target_mask_3d[:, :, best_slice_idx]

    # Find coordinates of active voxels in the chosen slice
    coords = np.argwhere(slice_2d)
    if len(coords) < 4:
        return {
            "max_diameter_d1_mm": 0.0,
            "perp_diameter_d2_mm": 0.0,
            "product_spd_mm2": 0.0,
            "best_slice_idx": best_slice_idx,
            "is_measurable": False,
            "rano_category": "Non-Measurable (Lesion < 4 voxels)"
        }

    # Physical scaling (X, Y)
    sx, sy = voxel_spacing[0], voxel_spacing[1]
    pts = coords * np.array([sx, sy])

    # Compute pairwise Euclidean distances to find maximal diameter D1
    # For speed on large lesions, sample points if dense
    # Exact and deterministic: the farthest pair and the extreme projections of a point set
    # always lie on its convex hull, so no random sampling is needed (the old random sample
    # made D1/D2 change between runs and between Streamlit reruns).
    sample_pts = pts
    if len(pts) > 3:
        try:
            # pyrefly: ignore[missing-import]
            from scipy.spatial import ConvexHull
            sample_pts = pts[ConvexHull(pts).vertices]
        except Exception:
            # Degenerate (e.g. collinear) point sets: fall back to a fixed-seed subsample.
            if len(pts) > 600:
                sample_pts = pts[np.random.default_rng(0).choice(len(pts), 600, replace=False)]

    diff = sample_pts[:, np.newaxis, :] - sample_pts[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    max_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    
    pt_a = sample_pts[max_idx[0]]
    pt_b = sample_pts[max_idx[1]]
    d1 = float(dist_matrix[max_idx])

    # Vector along D1
    vec_ab = pt_b - pt_a
    norm_ab = np.linalg.norm(vec_ab)
    if norm_ab > 1e-4:
        unit_ab = vec_ab / norm_ab
        # Perpendicular normal vector (-y, x)
        perp_vec = np.array([-unit_ab[1], unit_ab[0]])
        
        # Project all points onto perpendicular direction
        projections = np.dot(sample_pts - pt_a, perp_vec)
        d2 = float(np.max(projections) - np.min(projections))
    else:
        d2 = d1

    product_spd = round(d1 * d2, 2)
    d1_round = round(d1, 2)
    d2_round = round(d2, 2)

    # RANO guidelines: Measurable lesion must be >= 10mm x 10mm on 1.0mm slice
    is_measurable = (d1 >= 10.0 and d2 >= 10.0)
    category = "Measurable Lesion (>= 10x10 mm)" if is_measurable else "Non-Measurable Focus (< 10 mm)"

    return {
        "max_diameter_d1_mm": d1_round,
        "perp_diameter_d2_mm": d2_round,
        "product_spd_mm2": product_spd,
        "best_slice_idx": best_slice_idx,
        "is_measurable": is_measurable,
        "rano_category": category
    }


def compute_subregion_volumetrics(
    pred_3channel: np.ndarray, 
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_island_thresholds: Tuple[int, int, int] = (50, 30, 10)
) -> Dict[str, Any]:
    """
    Processes 3-channel binary predictions [WT, TC, ET],
    performs adaptive connected-component cleaning, enforces anatomical hierarchy,
    and calculates volume in mm³ and mL (1 mL = 1 cm³ = 1000 mm³).
    
    Args:
        pred_3channel: Array of shape (3, H, W, D) [WT, TC, ET]
        voxel_spacing: Resolution per voxel (sx, sy, sz) in mm (default: 1.0mm isotropic)
        min_island_thresholds: Component filter thresholds for (WT, TC, ET)
    Returns:
        Dictionary with:
        - summary_table (DataFrame)
        - edema_volume_ml
        - necrosis_volume_ml
        - cleaned_masks (3, H, W, D)
        - rano_metrics (RANO criteria assessment)
    """
    voxel_vol_mm3 = float(voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2])
    
    subregion_names = ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"]
    raw_masks = [pred_3channel[c] > 0.5 for c in range(3)]
    
    # Adaptive island cleaning
    cleaned = [
        remove_spurious_islands(raw_masks[c], min_size_voxels=min_island_thresholds[c])
        for c in range(3)
    ]
    
    # Enforce anatomical hierarchy: ET in TC in WT
    wt_clean, tc_clean, et_clean = enforce_hierarchical_consistency(cleaned[0], cleaned[1], cleaned[2])
    final_cleaned_masks = [wt_clean, tc_clean, et_clean]

    report_data = []
    for c in range(3):
        raw_count = int(np.sum(raw_masks[c]))
        clean_count = int(np.sum(final_cleaned_masks[c]))
        filtered_count = raw_count - clean_count
        vol_mm3 = clean_count * voxel_vol_mm3
        vol_ml = vol_mm3 / 1000.0

        report_data.append({
            "Sub-Region": subregion_names[c],
            "Raw Voxel Count": raw_count,
            "Cleaned Voxel Count": clean_count,
            "Noise Filtered": filtered_count,
            "Volume (mm³)": round(vol_mm3, 2),
            "Volume (mL / cm³)": round(vol_ml, 3)
        })

    df_report = pd.DataFrame(report_data)

    # Derived compartment volumes:
    # Edema = WT - TC
    wt_ml = df_report.loc[df_report["Sub-Region"] == "Whole Tumor (WT)", "Volume (mL / cm³)"].values[0]
    tc_ml = df_report.loc[df_report["Sub-Region"] == "Tumor Core (TC)", "Volume (mL / cm³)"].values[0]
    et_ml = df_report.loc[df_report["Sub-Region"] == "Enhancing Tumor (ET)", "Volume (mL / cm³)"].values[0]
    
    edema_ml = max(0.0, round(wt_ml - tc_ml, 3))
    # Necrotic / non-enhancing core = TC - ET
    necrosis_ml = max(0.0, round(tc_ml - et_ml, 3))

    # Evaluate RANO criteria on the active enhancing tumor (ET)
    rano_metrics = compute_rano_criteria_metrics(et_clean, voxel_spacing=voxel_spacing)

    return {
        "summary_table": df_report,
        "edema_volume_ml": edema_ml,
        "necrosis_volume_ml": necrosis_ml,
        "cleaned_masks": np.stack(final_cleaned_masks, axis=0),
        "rano_metrics": rano_metrics
    }


def generate_clinical_text_report(patient_id: str, volumetrics_dict: Dict[str, Any]) -> str:
    """Formats volumetric metrics into a standard neuro-oncology diagnostic summary."""
    df = volumetrics_dict["summary_table"]
    edema_ml = volumetrics_dict["edema_volume_ml"]
    necrosis_ml = volumetrics_dict["necrosis_volume_ml"]
    rano = volumetrics_dict.get("rano_metrics", {})
    
    lines = [
        "=" * 68,
        f"  NEUROSEG QUANTITATIVE BRAIN MRI VOLUMETRICS REPORT",
        f"  Patient / Scan Identifier: {patient_id}",
        "=" * 68,
        f"{'Compartment':<26} | {'Voxel Count':<12} | {'Volume (mL)':<14}",
        "-" * 68
    ]
    
    for _, row in df.iterrows():
        lines.append(f"{row['Sub-Region']:<26} | {row['Cleaned Voxel Count']:<12} | {row['Volume (mL / cm³)']:<14.3f}")
        
    lines.append("-" * 68)
    lines.append(f"{'Peritumoral Edema':<26} | {'Derived':<12} | {edema_ml:<14.3f}")
    lines.append(f"{'Necrotic Cavity / Core':<26} | {'Derived':<12} | {necrosis_ml:<14.3f}")
    lines.append("=" * 68)
    lines.append("CLINICAL RANO CRITERIA ASSESSMENT:")
    lines.append(f"  Target Enhancing Diameter (D1): {rano.get('max_diameter_d1_mm', 0.0)} mm")
    lines.append(f"  Perpendicular Diameter (D2):   {rano.get('perp_diameter_d2_mm', 0.0)} mm")
    lines.append(f"  Product of Diameters (SPD):    {rano.get('product_spd_mm2', 0.0)} mm² (Axial Slice #{rano.get('best_slice_idx', 0)})")
    lines.append(f"  Classification:                {rano.get('rano_category', 'N/A')}")
    lines.append("=" * 68)
    
    return "\n".join(lines)