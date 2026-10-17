"""
NeuroSeg - Module 4 Verification Suite:
Tests Morphological Post-Processing, 3D Connected-Component Island Filtering,
Hierarchical Consistency, Quantitative Volumetrics, and Clinical RANO Criteria.
"""

import os
import sys
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from src.volumetrics import (
    remove_spurious_islands,
    enforce_hierarchical_consistency,
    compute_rano_criteria_metrics,
    compute_subregion_volumetrics,
    generate_clinical_text_report
)


def test_module4():
    print("=" * 65)
    print("  VERIFYING MODULE 4: MORPHOLOGICAL FILTERING & RANO ENGINE")
    print("=" * 65)

    shape = (128, 128, 128)
    pred_3channel = np.zeros((3, *shape), dtype=np.uint8)

    # 1. Synthesize ground-truth tumor mass
    z, y, x = np.ogrid[:128, :128, :128]
    dist_center = np.sqrt((x - 64)**2 + (y - 64)**2 + (z - 64)**2)

    pred_3channel[0][dist_center <= 22] = 1  # WT main mass
    pred_3channel[1][dist_center <= 15] = 1  # TC solid core
    pred_3channel[2][(dist_center >= 10) & (dist_center <= 15)] = 1  # ET rim

    # Inject spurious false-positive noise islands
    noise_coords = [(15, 15, 15), (105, 105, 20), (25, 95, 110)]
    for coord in noise_coords:
        cz, cy, cx = coord
        pred_3channel[0, cz:cz+3, cy:cy+3, cx:cx+3] = 1  # ~27-voxel noise speck
        pred_3channel[1, cz:cz+2, cy:cy+2, cx:cx+2] = 1  # ~8-voxel noise speck

    print(" [1/5] Synthesized 3D tumor volume with injected false-positive noise clusters.")

    # 2. Test Island Filtering & Volumetric Computation
    results = compute_subregion_volumetrics(
        pred_3channel, 
        voxel_spacing=(1.0, 1.0, 1.0),
        min_island_thresholds=(50, 30, 10)
    )
    df_vol = results["summary_table"]
    edema_ml = results["edema_volume_ml"]
    necrosis_ml = results["necrosis_volume_ml"]
    cleaned_masks = results["cleaned_masks"]
    rano = results["rano_metrics"]

    print(" [2/5] Morphological Filtering Results:")
    for _, row in df_vol.iterrows():
        print(f"       * {row['Sub-Region']:<22}: Filtered {row['Noise Filtered']} noise voxels | Cleaned Vol = {row['Volume (mL / cm³)']} mL")

    # Verify that noise was removed
    assert df_vol.loc[df_vol["Sub-Region"] == "Whole Tumor (WT)", "Noise Filtered"].values[0] > 0, "Noise specks were not filtered."

    # 3. Test Hierarchical Consistency (ET subset of TC subset of WT)
    wt_c = cleaned_masks[0] > 0
    tc_c = cleaned_masks[1] > 0
    et_c = cleaned_masks[2] > 0

    assert np.all(tc_c <= wt_c), "Biological violation: Tumor Core exceeds Whole Tumor boundaries!"
    assert np.all(et_c <= tc_c), "Biological violation: Enhancing Tumor exceeds Tumor Core boundaries!"
    print(" [3/5] Anatomical Hierarchical Consistency Verified: ET subset of TC subset of WT")

    # 4. Test RANO Criteria Engine
    print(" [4/5] Clinical RANO Criteria Evaluated:")
    print(f"       Maximal Diameter (D1):        {rano['max_diameter_d1_mm']} mm")
    print(f"       Perpendicular Diameter (D2):  {rano['perp_diameter_d2_mm']} mm")
    print(f"       Product of Diameters (SPD):   {rano['product_spd_mm2']} mm²")
    print(f"       Measurement Slice Index:      Axial #{rano['best_slice_idx']}")
    print(f"       Classification:               {rano['rano_category']}")
    assert rano["max_diameter_d1_mm"] > 0, "RANO diameter computation failed."
    assert rano["product_spd_mm2"] > 0, "RANO SPD computation failed."

    # 5. Format & Display Clinical Text Report
    report_txt = generate_clinical_text_report("BraTS2021_00001", results)
    print("\n" + report_txt)

    # 6. Save Verification Visual
    outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    out_plot = os.path.join(outputs_dir, "Module_4_Verification.png")

    slice_idx = 64
    raw_wt = pred_3channel[0, :, :, slice_idx]
    clean_wt = cleaned_masks[0, :, :, slice_idx]
    clean_tc = cleaned_masks[1, :, :, slice_idx]
    clean_et = cleaned_masks[2, :, :, slice_idx]

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(raw_wt, cmap="gray")
    ax[0].set_title("Raw WT (With Noise Islands)", fontweight="bold")
    ax[0].axis("off")

    ax[1].imshow(clean_wt, cmap="Blues")
    ax[1].set_title(f"Cleaned WT ({df_vol.iloc[0]['Volume (mL / cm³)']:.1f} mL)", fontweight="bold")
    ax[1].axis("off")

    ax[2].imshow(clean_tc, cmap="Oranges")
    ax[2].set_title(f"Tumor Core ({df_vol.iloc[1]['Volume (mL / cm³)']:.1f} mL)", fontweight="bold")
    ax[2].axis("off")

    ax[3].imshow(clean_et, cmap="Reds")
    ax[3].set_title(f"Enhancing Rim ({df_vol.iloc[2]['Volume (mL / cm³)']:.1f} mL)", fontweight="bold")
    ax[3].axis("off")

    plt.tight_layout()
    plt.savefig(out_plot, dpi=200)
    plt.close(fig)
    print(f" [5/5] Verification visual saved to: {out_plot}")

    print(" >>> MODULE 4 VERIFICATION PASSED <<<\n")


if __name__ == "__main__":
    test_module4()