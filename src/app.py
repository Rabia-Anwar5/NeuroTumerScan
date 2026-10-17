"""
NeuroSeg - Module 5: Clinical Multi-Parametric MRI Brain Tumor AI Workstation
Interactive Clinical Streamlit Application featuring:
- Tri-Planar Orthogonal 2D Slice Overlays (Axial, Coronal, Sagittal)
- Real-time 3D Interactive Surface Mesh Reconstruction (Marching Cubes & Plotly 3D)
- Deep Learning 3D SegResNet Sliding-Window Inference Engine
- Quantitative Tumor Burden Volumetrics & Clinical RANO Criteria Assessment
- Automated Clinical PDF Diagnostic Summary Export (ReportLab)
- BraTS Benchmark Metrics Validation Performance Dashboard
"""

import os
import sys
import io
import json
import time
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
# pyrefly: ignore [missing-import]
import nibabel as nib
from skimage import measure

# Royal glass theme (neuroseg_theme.py lives next to this file)
from neuroseg_theme import apply_theme, hero_header, PLOT, MASK_COLORS

# ----------------- PATH CONFIGURATION -----------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

try:
    from src.volumetrics import compute_subregion_volumetrics, generate_clinical_text_report
    from src.dataset_pipeline import load_brats_file_list, validate_skull_stripping, verify_and_align_multimodal_sequences
    from src.inference import BrainTumorInferenceEngine, compute_clinical_validation_metrics
    from src.report_generator import generate_pdf_report
except ModuleNotFoundError:
    from volumetrics import compute_subregion_volumetrics, generate_clinical_text_report
    from dataset_pipeline import load_brats_file_list, validate_skull_stripping, verify_and_align_multimodal_sequences
    from inference import BrainTumorInferenceEngine, compute_clinical_validation_metrics
    from report_generator import generate_pdf_report

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="NeuroSeg - Clinical 3D Brain Tumor AI Workstation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Royal glass theme (plum / navy / sky / pink / cream)
apply_theme()

# ----------------- HEADER -----------------
col_hdr1, col_hdr2 = st.columns([3, 1])
with col_hdr1:
    hero_header(
        "NeuroSeg",
        "Automated Multi-Sequence MRI Brain Tumor Sub-Region AI",
        "Deep Supervision 3D SegResNet Segmentation, Volumetric Quantification & RANO Response Criteria",
    )
with col_hdr2:
    st.markdown("""
        <div style="text-align: right; padding-top: 14px;">
            <span class="ns-badge"><span class="ns-dot"></span>MONAI 1.6 Medical Core</span>
            <div style="font-size: 12px; color: rgba(242,225,212,.6); margin-top: 10px;">BraTS Standard: 1.0mm³ RAS</div>
        </div>
    """, unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("Patient Case Selector")
data_dir = os.path.join(PROJECT_ROOT, "data", "BraTS2021_Training_Data")
sample_dir = os.path.join(PROJECT_ROOT, "data", "BraTS2021_Sample")

patient_cases = load_brats_file_list(data_dir)
if not patient_cases:
    patient_cases = load_brats_file_list(sample_dir)

selected_case = None
if patient_cases:
    case_ids = [c["patient_id"] for c in patient_cases]
    selected_id = st.sidebar.selectbox("Select Patient Scan:", case_ids)
    selected_case = next((c for c in patient_cases if c["patient_id"] == selected_id), None)
else:
    st.sidebar.info("No local scans detected. Running in synthetic mode.")

st.sidebar.markdown("---")
st.sidebar.header("MRI Sequence & Plane")
modality_choice = st.sidebar.selectbox(
    "Active Display Sequence:",
    ["T1CE (Contrast-Enhanced Active Rim)", "T1 (Anatomical Reference)", "T2 (Hyperintense Edema)", "FLAIR (Infiltration Zone)"]
)
modality_map = {
    "T1 (Anatomical Reference)": 0, 
    "T1CE (Contrast-Enhanced Active Rim)": 1, 
    "T2 (Hyperintense Edema)": 2, 
    "FLAIR (Infiltration Zone)": 3
}
mod_idx = modality_map[modality_choice]

plane_view_mode = st.sidebar.radio("View Mode:", ["Orthogonal Tri-Planar (All 3)", "Single Plane Focus"])
single_plane = "Axial (Transverse)"
if plane_view_mode == "Single Plane Focus":
    single_plane = st.sidebar.selectbox("Select Plane:", ["Axial (Transverse)", "Coronal (Frontal)", "Sagittal (Profile)"])

st.sidebar.markdown("---")
st.sidebar.header("Overlay Controls")
show_wt = st.sidebar.checkbox("Show Whole Tumor (WT / Edema - Sky)", value=True)
show_tc = st.sidebar.checkbox("Show Tumor Core (TC - Pink)", value=True)
show_et = st.sidebar.checkbox("Show Enhancing Tumor (ET - Cream)", value=True)
overlay_opacity = st.sidebar.slider("Mask Overlay Opacity:", 0.1, 1.0, 0.65, 0.05)


# ----------------- DATA LOADING -----------------
@st.cache_data
def load_scan_data(case_info):
    if case_info is not None:
        try:
            images = [nib.load(p).get_fdata() for p in case_info["image"]]
            volume = np.stack(images, axis=0)
            
            if "label" in case_info and os.path.exists(case_info["label"]):
                seg = nib.load(case_info["label"]).get_fdata()
            else:
                seg = np.zeros(volume.shape[1:], dtype=np.uint8)
            return volume, seg, False
        except Exception as e:
            pass

    # High-fidelity synthetic fallback
    shape = (128, 128, 128)
    vol = np.random.randn(4, *shape) * 12 + 100
    vol[vol < 0] = 0
    seg = np.zeros(shape, dtype=np.uint8)
    z, y, x = np.ogrid[:128, :128, :128]
    dist = np.sqrt((x - 64)**2 + (y - 64)**2 + (z - 64)**2)
    seg[dist <= 24] = 2  # Edema
    seg[dist <= 16] = 1  # Necrotic Core
    seg[(dist >= 11) & (dist <= 16)] = 4  # Enhancing margin
    return vol, seg, True

volume, seg, is_synthetic = load_scan_data(selected_case)

# Initialize Session State for AI Predictions
if "pred_masks" not in st.session_state or st.session_state.get("current_case") != (selected_case["patient_id"] if selected_case else "synthetic"):
    # Initial Ground-Truth or base conversion: WT, TC, ET
    wt_raw = (seg == 1) | (seg == 2) | (seg == 4)
    tc_raw = (seg == 1) | (seg == 4)
    et_raw = (seg == 4)
    pred_3ch = np.stack([wt_raw, tc_raw, et_raw], axis=0)
    
    st.session_state["pred_masks"] = pred_3ch
    st.session_state["current_case"] = selected_case["patient_id"] if selected_case else "synthetic"
    st.session_state["inference_performed"] = False

# Compute Volumetrics & RANO
vol_metrics = compute_subregion_volumetrics(
    st.session_state["pred_masks"], 
    voxel_spacing=(1.0, 1.0, 1.0),
    min_island_thresholds=(50, 30, 10)
)
df_vol = vol_metrics["summary_table"]
edema_ml = vol_metrics["edema_volume_ml"]
necrosis_ml = vol_metrics["necrosis_volume_ml"]
rano = vol_metrics["rano_metrics"]
cleaned_masks = vol_metrics["cleaned_masks"]

# ----------------- TABS INTERFACE -----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Orthogonal 2D Slices", 
    "3D Surface Reconstruction", 
    "Quantitative Volumetrics & RANO", 
    "Deep Learning Inference", 
    "Clinical PDF Diagnostic Report",
    "Validation Benchmarks"
])


# ==============================================================================
# TAB 1: ORTHOGONAL 2D SLICE VIEWER
# ==============================================================================
with tab1:
    st.subheader(f"Tri-Planar Orthogonal Multi-Slice Inspection — {modality_choice.split(' ')[0]}")
    
    # Sliders for coordinate navigation
    dim_x, dim_y, dim_z = volume.shape[1], volume.shape[2], volume.shape[3]
    
    c_s1, c_s2, c_s3 = st.columns(3)
    with c_s1:
        slice_z = st.slider("Axial Slice (Z - Superior/Inferior):", 0, dim_z - 1, dim_z // 2)
    with c_s2:
        slice_y = st.slider("Coronal Slice (Y - Anterior/Posterior):", 0, dim_y - 1, dim_y // 2)
    with c_s3:
        slice_x = st.slider("Sagittal Slice (X - Left/Right):", 0, dim_x - 1, dim_x // 2)

    def render_slice_with_overlay(bg_2d, wt_2d, tc_2d, et_2d, title):
        norm_bg = (bg_2d - bg_2d.min()) / (bg_2d.max() - bg_2d.min() + 1e-6)
        rgb_img = np.stack([norm_bg, norm_bg, norm_bg], axis=-1)

        # Alpha blend masks
        alpha = overlay_opacity
        if show_wt:
            mask = wt_2d > 0
            rgb_img[mask] = (1 - alpha) * rgb_img[mask] + alpha * np.array(MASK_COLORS["WT"]) / 255.0
        if show_tc:
            mask = tc_2d > 0
            rgb_img[mask] = (1 - alpha) * rgb_img[mask] + alpha * np.array(MASK_COLORS["TC"]) / 255.0
        if show_et:
            mask = et_2d > 0
            rgb_img[mask] = (1 - alpha) * rgb_img[mask] + alpha * np.array(MASK_COLORS["ET"]) / 255.0

        fig = px.imshow(np.transpose(rgb_img, (1, 0, 2)), origin="lower")
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=30),
            title=dict(text=title, x=0.5, font=dict(size=13, color="#F2E1D4")),
            coloraxis_showscale=False,
            **PLOT
        )
        return fig

    # Axial slices
    bg_ax = volume[mod_idx, :, :, slice_z]
    wt_ax = cleaned_masks[0, :, :, slice_z]
    tc_ax = cleaned_masks[1, :, :, slice_z]
    et_ax = cleaned_masks[2, :, :, slice_z]

    # Coronal slices
    bg_cor = volume[mod_idx, :, slice_y, :]
    wt_cor = cleaned_masks[0, :, slice_y, :]
    tc_cor = cleaned_masks[1, :, slice_y, :]
    et_cor = cleaned_masks[2, :, slice_y, :]

    # Sagittal slices
    bg_sag = volume[mod_idx, slice_x, :, :]
    wt_sag = cleaned_masks[0, slice_x, :, :]
    tc_sag = cleaned_masks[1, slice_x, :, :]
    et_sag = cleaned_masks[2, slice_x, :, :]

    if plane_view_mode == "Orthogonal Tri-Planar (All 3)":
        col_ax, col_cor, col_sag = st.columns(3)
        with col_ax:
            st.plotly_chart(render_slice_with_overlay(bg_ax, wt_ax, tc_ax, et_ax, f"Axial (Z={slice_z})"), use_container_width=True)
        with col_cor:
            st.plotly_chart(render_slice_with_overlay(bg_cor, wt_cor, tc_cor, et_cor, f"Coronal (Y={slice_y})"), use_container_width=True)
        with col_sag:
            st.plotly_chart(render_slice_with_overlay(bg_sag, wt_sag, tc_sag, et_sag, f"Sagittal (X={slice_x})"), use_container_width=True)
    else:
        if "Axial" in single_plane:
            st.plotly_chart(render_slice_with_overlay(bg_ax, wt_ax, tc_ax, et_ax, f"Axial Transverse (Z={slice_z})"), use_container_width=True)
        elif "Coronal" in single_plane:
            st.plotly_chart(render_slice_with_overlay(bg_cor, wt_cor, tc_cor, et_cor, f"Coronal Frontal (Y={slice_y})"), use_container_width=True)
        else:
            st.plotly_chart(render_slice_with_overlay(bg_sag, wt_sag, tc_sag, et_sag, f"Sagittal Profile (X={slice_x})"), use_container_width=True)

    st.caption("Sky: Whole Tumor (WT / Edema)  ·  Pink: Tumor Core (TC)  ·  Cream: Enhancing Active Rim (ET)")
    st.caption("Masks shown: " + ("SegResNet model prediction" if st.session_state.get("inference_performed")
              else "ground-truth annotation (use the Deep Learning Inference tab to view model output)"))


# ==============================================================================
# TAB 2: INTERACTIVE 3D SURFACE MESH RECONSTRUCTION
# ==============================================================================
with tab2:
    st.subheader("Interactive 3D Anatomical Surface Mesh Reconstruction")
    st.markdown("Real-time 3D isosurface mesh generation using Marching Cubes algorithm for stereotactic neurosurgical visualization.")
    
    c_3d_ctrl1, c_3d_ctrl2, c_3d_ctrl3 = st.columns(3)
    with c_3d_ctrl1:
        render_wt_3d = st.checkbox("Render Whole Tumor (Edema Shell)", value=True)
    with c_3d_ctrl2:
        render_tc_3d = st.checkbox("Render Tumor Core (TC)", value=True)
    with c_3d_ctrl3:
        render_et_3d = st.checkbox("Render Enhancing Margin (ET)", value=True)

    # Marching Cubes Mesh Extractor
    @st.cache_data
    def generate_3d_mesh_traces(masks_3ch):
        traces = []
        # Colors & opacities: WT (cyan), TC (gold), ET (red)
        configs = [
            ("Whole Tumor (WT)", 0, "#A7C6ED", 0.35),
            ("Tumor Core (TC)", 1, "#D9B3C4", 0.65),
            ("Enhancing Tumor (ET)", 2, "#F2E1D4", 0.90)
        ]

        for name, idx, color, opacity in configs:
            sub_mask = masks_3ch[idx]
            if np.sum(sub_mask) > 15:
                # Downsample slightly for ultra-smooth 60fps web browser 3D rotation
                step = 2 if sub_mask.shape[0] >= 128 else 1
                down_mask = sub_mask[::step, ::step, ::step]
                
                try:
                    verts, faces, normals, _ = measure.marching_cubes(down_mask, level=0.5, step_size=1)
                    verts = verts * step  # Scale back to original coordinates
                    
                    mesh_trace = go.Mesh3d(
                        x=verts[:, 0],
                        y=verts[:, 1],
                        z=verts[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        name=name,
                        color=color,
                        opacity=opacity,
                        lighting=dict(ambient=0.4, diffuse=0.8, specular=0.4, roughness=0.5),
                        lightposition=dict(x=100, y=200, z=150),
                        showlegend=True
                    )
                    traces.append(mesh_trace)
                except Exception:
                    pass
        return traces

    mesh_traces = generate_3d_mesh_traces(cleaned_masks)

    if mesh_traces:
        fig_3d = go.Figure()
        for tr in mesh_traces:
            if "Whole Tumor" in tr.name and not render_wt_3d:
                continue
            if "Tumor Core" in tr.name and not render_tc_3d:
                continue
            if "Enhancing" in tr.name and not render_et_3d:
                continue
            fig_3d.add_trace(tr)

        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(title="X (Sagittal mm)", backgroundcolor="rgba(242,225,212,0.04)", gridcolor="rgba(242,225,212,0.2)"),
                yaxis=dict(title="Y (Coronal mm)", backgroundcolor="rgba(242,225,212,0.04)", gridcolor="rgba(242,225,212,0.2)"),
                zaxis=dict(title="Z (Axial mm)", backgroundcolor="rgba(242,225,212,0.04)", gridcolor="rgba(242,225,212,0.2)"),
                aspectmode="data"
            ),
            margin=dict(l=0, r=0, b=0, t=20),
            height=580,
            **PLOT,
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(75,44,58,0.65)")
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.info("No tumor voxels detected to construct 3D surface mesh.")


# ==============================================================================
# TAB 3: QUANTITATIVE VOLUMETRICS & RANO CRITERIA
# ==============================================================================
with tab3:
    st.subheader("Quantitative Tumor Compartment Metrics & RANO Response Assessment")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    wt_ml = df_vol.loc[df_vol["Sub-Region"] == "Whole Tumor (WT)", "Volume (mL / cm³)"].values[0]
    tc_ml = df_vol.loc[df_vol["Sub-Region"] == "Tumor Core (TC)", "Volume (mL / cm³)"].values[0]
    et_ml = df_vol.loc[df_vol["Sub-Region"] == "Enhancing Tumor (ET)", "Volume (mL / cm³)"].values[0]

    m1.metric("Whole Tumor (WT)", f"{wt_ml:.2f} mL")
    m2.metric("Tumor Core (TC)", f"{tc_ml:.2f} mL")
    m3.metric("Enhancing Margin (ET)", f"{et_ml:.2f} mL")
    m4.metric("Peritumoral Edema", f"{edema_ml:.2f} mL")
    m5.metric("Necrotic Core", f"{necrosis_ml:.2f} mL")

    st.markdown("---")
    
    c_tbl, c_chart = st.columns([1.2, 1])
    with c_tbl:
        st.markdown("**Structured Volumetric Compartment Breakdown**")
        st.dataframe(df_vol, use_container_width=True)
        
        st.markdown("**Derived Compartment Metrics:**")
        st.write(f"- **Peritumoral Edema (WT - TC):** `{edema_ml:.3f} mL`")
        st.write(f"- **Necrotic / Cavitary Core (TC - ET):** `{necrosis_ml:.3f} mL`")

    with c_chart:
        st.markdown("**Sub-Region Volumetric Burden (mL)**")
        chart_df = pd.DataFrame({
            "Compartment": ["Whole Tumor", "Tumor Core", "Enhancing Margin", "Peritumoral Edema", "Necrotic Cavity"],
            "Volume (mL)": [wt_ml, tc_ml, et_ml, edema_ml, necrosis_ml]
        })
        fig_bar = px.bar(
            chart_df,
            x="Compartment",
            y="Volume (mL)",
            color="Compartment",
            color_discrete_sequence=["#A7C6ED", "#D9B3C4", "#F2E1D4", "#5B84CE", "#8E5F74"]
        )
        fig_bar.update_layout(showlegend=False, margin=dict(l=10, r=10, b=10, t=20), **PLOT)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("Clinical RANO (Response Assessment in Neuro-Oncology) Criteria Engine")
    
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Maximal Diameter (D1)", f"{rano.get('max_diameter_d1_mm', 0.0)} mm")
    rc2.metric("Perpendicular Diameter (D2)", f"{rano.get('perp_diameter_d2_mm', 0.0)} mm")
    rc3.metric("Sum of Products (SPD)", f"{rano.get('product_spd_mm2', 0.0)} mm²")
    rc4.metric("Measurement Slice", f"Axial #{rano.get('best_slice_idx', 0)}")

    st.info(f"**RANO Clinical Classification:** {rano.get('rano_category', 'Evaluated')} (Threshold: Measurable target lesion requires D1 ≥ 10mm and D2 ≥ 10mm on 1mm isotropic slice).")


# ==============================================================================
# TAB 4: DEEP LEARNING INFERENCE RUNNER
# ==============================================================================
with tab4:
    st.subheader("3D SegResNet Deep Learning Inference Engine")
    st.markdown("Run automated sliding-window inference with Gaussian blending directly on the active multi-parametric MRI scan.")
    
    c_inf1, c_inf2 = st.columns([2, 1])
    with c_inf1:
        st.write(f"**Active Scan:** `{selected_case['patient_id'] if selected_case else 'Synthetic Scan'}`")
        st.write(f"**Input Tensor Geometry:** `(4, {volume.shape[1]}, {volume.shape[2]}, {volume.shape[3]})` (T1, T1CE, T2, FLAIR)")
        st.write("**Inference Mode:** Overlapping 3D Gaussian Blended Sliding Window (`patch_size=(128, 128, 128)`, `overlap=0.25`)")

    with c_inf2:
        run_btn = st.button("Run SegResNet AI Segmentation", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Executing 3D SegResNet Sliding-Window Inference across volume..."):
            t_start = time.time()
            engine = BrainTumorInferenceEngine(
                checkpoint_path=os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "best_segresnet_metric.pth")
            )
            # The model was trained on z-score-normalized inputs (non-zero voxels, per channel,
            # i.e. NormalizeIntensityd(nonzero=True, channel_wise=True)). Raw intensities must be
            # normalized the same way or the predictions are meaningless.
            vol_norm = volume.astype(np.float32).copy()
            for _c in range(vol_norm.shape[0]):
                _m = vol_norm[_c] != 0
                if _m.any():
                    vol_norm[_c][_m] = (vol_norm[_c][_m] - vol_norm[_c][_m].mean()) / (vol_norm[_c][_m].std() + 1e-8)
            pred_results = engine.predict_full_volume(vol_norm, overlap=0.25)
            t_elapsed = time.time() - t_start

            st.session_state["pred_masks"] = pred_results["binary_3channel"]
            st.session_state["inference_performed"] = True
            st.session_state["ckpt_loaded"] = bool(getattr(engine, "checkpoint_loaded", False))
            
            st.success(f"Inference complete in {t_elapsed:.2f} seconds! Slices and volumetrics updated automatically.")
            st.rerun()

    if st.session_state.get("inference_performed"):
        st.info("Segmentation masks reflect current SegResNet model predictions.")
        if not st.session_state.get("ckpt_loaded", False):
            st.warning("No trained checkpoint was loaded - these predictions come from untrained weights and are not meaningful.")


# ==============================================================================
# TAB 5: CLINICAL PDF DIAGNOSTIC REPORT EXPORT
# ==============================================================================
with tab5:
    st.subheader("Automated Clinical PDF Diagnostic Summary Export")
    st.markdown("Generates an official neuro-oncology diagnostic report ready for tumor boards, neurosurgical planning, and PACS records.")
    
    patient_tag = selected_case["patient_id"] if selected_case else "BraTS2021_00001"
    
    # Generate in-memory PDF
    slice_data_for_pdf = {
        "bg_slice": volume[1, :, :, dim_z // 2],  # T1CE
        "masks": {
            "WT": cleaned_masks[0, :, :, dim_z // 2],
            "TC": cleaned_masks[1, :, :, dim_z // 2],
            "ET": cleaned_masks[2, :, :, dim_z // 2]
        },
        "slice_idx": dim_z // 2
    }
    
    pdf_bytes = generate_pdf_report(
        patient_id=patient_tag,
        volumetrics_dict=vol_metrics,
        slice_data=slice_data_for_pdf,
        mask_source=("SegResNet model prediction" if st.session_state.get("inference_performed")
                     else "Ground-truth annotation"),
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.download_button(
            label="Download Official Medical PDF Report",
            data=pdf_bytes,
            file_name=f"NeuroSeg_Diagnostic_Report_{patient_tag}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
    with col_btn2:
        csv_data = df_vol.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Structured CSV Metrics",
            data=csv_data,
            file_name=f"NeuroSeg_Metrics_{patient_tag}.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("**Clinical Report Text Preview:**")
    report_text = generate_clinical_text_report(patient_tag, vol_metrics)
    st.text_area("Preview:", report_text, height=220)


# ==============================================================================
# TAB 6: VALIDATION BENCHMARKS & MODEL QUALITY SCORECARD
# ==============================================================================
with tab6:
    st.subheader("Model Performance & BraTS Challenge Benchmark Validation")

    # Targets from the Module 3 specification. No Tumor Core target is specified there.
    DICE_TARGETS = {"WT": 0.88, "TC": None, "ET": 0.82}
    HD95_TARGET = 5.0
    history_path = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "training_history.json")

    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception as e:
            st.error(f"Could not read {history_path}: {e}")

    if not history:
        st.warning(
            "No validation record found. Run `python src/train.py` to train and validate; "
            "it writes outputs/checkpoints/training_history.json, which this tab reads."
        )
    else:
        best = max(history, key=lambda r: r["dice"]["mean"])
        st.markdown(
            f"Best validation epoch: **{best['epoch']}** of {len(history)} "
            f"· validation cases: **{best.get('val_cases', 'unknown')}**"
        )
        if not best.get("held_out", False):
            st.warning(
                "These scores were measured on cases that were also used for training. "
                "They are not a held-out evaluation and must not be presented as one."
            )

        wt_d, et_d = best["dice"]["WT"], best["dice"]["ET"]
        wt_h = best["hd95_mm"]["WT"]
        col_sc1, col_sc2, col_sc3 = st.columns(3)
        col_sc1.metric("Whole Tumor (WT) Dice", f"{wt_d:.3f}", delta=f"{wt_d - DICE_TARGETS['WT']:+.3f} (Target ≥ {DICE_TARGETS['WT']})")
        col_sc2.metric("Enhancing Tumor (ET) Dice", f"{et_d:.3f}", delta=f"{et_d - DICE_TARGETS['ET']:+.3f} (Target ≥ {DICE_TARGETS['ET']})")
        if wt_h is None:
            col_sc3.metric("WT 95% Hausdorff Distance", "n/a")
        else:
            col_sc3.metric("WT 95% Hausdorff Distance", f"{wt_h:.2f} mm",
                           delta=f"{wt_h - HD95_TARGET:+.2f} mm (Target ≤ {HD95_TARGET} mm)", delta_color="inverse")

        rows = []
        for key, name in [("WT", "Whole Tumor (WT)"), ("TC", "Tumor Core (TC)"), ("ET", "Enhancing Tumor (ET)")]:
            d, h, dt = best["dice"][key], best["hd95_mm"][key], DICE_TARGETS[key]
            if dt is None:
                verdict = "No target specified"
            else:
                verdict = "MET" if (d >= dt and h is not None and h <= HD95_TARGET) else "NOT MET"
            rows.append({
                "Sub-Region": name,
                "Dice (Model)": f"{d:.4f}",
                "Dice Target": "-" if dt is None else f"{dt:.2f}",
                "HD95 (mm)": "n/a" if h is None else f"{h:.2f}",
                "HD95 Target": f"{HD95_TARGET:.2f}",
                "Verdict": verdict,
            })
        st.table(pd.DataFrame(rows))

        if len(history) > 1:
            fig_curve = px.line(
                x=[r["epoch"] for r in history],
                y=[r["dice"]["mean"] for r in history],
                labels={"x": "Epoch", "y": "Mean validation Dice"},
            )
            fig_curve.update_layout(margin=dict(l=10, r=10, b=10, t=20), **PLOT)
            st.plotly_chart(fig_curve, use_container_width=True)

        st.caption(f"Metrics are read from {history_path}; verdicts are computed from them, not hardcoded.")