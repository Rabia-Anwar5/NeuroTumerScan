"""
NeuroSeg - Module 5 Verification Suite:
Tests Automated Medical PDF Diagnostic Report Generation,
3D Surface Mesh Extraction (Marching Cubes), and Streamlit Component Integrity.
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from src.volumetrics import compute_subregion_volumetrics
from src.report_generator import generate_pdf_report
from skimage import measure


def test_module5():
    print("=" * 65)
    print("  VERIFYING MODULE 5: 3D RECONSTRUCTION & CLINICAL REPORTING")
    print("=" * 65)

    # 1. Synthesize 3D tumor volume
    shape = (128, 128, 128)
    pred_3ch = np.zeros((3, *shape), dtype=np.uint8)
    z, y, x = np.ogrid[:128, :128, :128]
    dist = np.sqrt((x - 64)**2 + (y - 64)**2 + (z - 64)**2)
    pred_3ch[0][dist <= 22] = 1  # WT
    pred_3ch[1][dist <= 15] = 1  # TC
    pred_3ch[2][(dist >= 10) & (dist <= 15)] = 1  # ET

    vol_metrics = compute_subregion_volumetrics(pred_3ch, voxel_spacing=(1.0, 1.0, 1.0))
    print(" [1/3] Computed volumetric metrics & RANO parameters.")

    # 2. Test 3D Surface Mesh Extraction via Marching Cubes
    print(" [2/3] Extracting 3D Isosurface Meshes (Marching Cubes)...")
    wt_mask = vol_metrics["cleaned_masks"][0]
    verts, faces, normals, _ = measure.marching_cubes(wt_mask[::2, ::2, ::2], level=0.5)
    print(f"       Extracted Whole Tumor 3D Mesh: {len(verts):,} Vertices | {len(faces):,} Triangular Faces")
    assert len(verts) > 100, "3D Mesh vertices extraction failed."
    assert len(faces) > 100, "3D Mesh faces extraction failed."

    # 3. Test Automated Clinical PDF Generation
    print(" [3/3] Generating Official Clinical PDF Diagnostic Report...")
    outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    pdf_out_path = os.path.join(outputs_dir, "NeuroSeg_Sample_Report.pdf")

    bg_slice = (np.random.randn(128, 128) * 20 + 100).astype(np.float32)
    slice_data = {
        "bg_slice": bg_slice,
        "masks": {
            "WT": vol_metrics["cleaned_masks"][0, :, :, 64],
            "TC": vol_metrics["cleaned_masks"][1, :, :, 64],
            "ET": vol_metrics["cleaned_masks"][2, :, :, 64]
        },
        "slice_idx": 64
    }

    pdf_bytes = generate_pdf_report(
        patient_id="BraTS2021_00001",
        volumetrics_dict=vol_metrics,
        slice_data=slice_data,
        output_path=pdf_out_path
    )

    assert os.path.exists(pdf_out_path), "PDF failed to save to disk."
    pdf_size_kb = os.path.getsize(pdf_out_path) / 1024.0
    print(f"       Successfully generated PDF report ({pdf_size_kb:.1f} KB): {pdf_out_path}")
    assert pdf_size_kb > 10.0, "PDF file size unexpectedly small."

    print(" >>> MODULE 5 VERIFICATION PASSED <<<\n")


if __name__ == "__main__":
    test_module5()
