"""
NeuroSeg - Module 3: Full 3D Scan Sliding-Window Inference Engine
Performs Gaussian-blended sliding window inference across multi-parametric brain MRI scans.
"""

from typing import Union, Tuple, Optional, Dict
import warnings
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import numpy as np

# Suppress PyTorch JIT deprecation warnings emitted by MONAI
warnings.filterwarnings("ignore", category=FutureWarning)

# pyrefly: ignore [missing-import]
from monai.inferers import sliding_window_inference
# pyrefly: ignore [missing-import]
from monai.transforms import Activations, AsDiscrete, Compose
# pyrefly: ignore [missing-import]
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from src.model import get_segresnet_model, convert_labels_to_brats_classes, convert_brats_classes_to_labels


class BrainTumorInferenceEngine:
    """
    Inference Engine using trained 3D SegResNet / UNet weights
    with overlapping Gaussian sliding window evaluation.
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_name: str = "segresnet",
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        self.model = get_segresnet_model(in_channels=4, out_channels=3, init_filters=16).to(self.device)
        self.post_sigmoid = Activations(sigmoid=True)
        self.post_discrete = AsDiscrete(threshold=0.5)

        self.checkpoint_loaded = False  # lets the app warn when weights are NOT trained
        if checkpoint_path:
            try:
                import os
                if os.path.exists(checkpoint_path):
                    state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                    self.model.load_state_dict(state_dict)
                    self.checkpoint_loaded = True
                    print(f"Loaded trained checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"Notice: Checkpoint could not be loaded ({e}), using initialized backbone weights.")

        self.model.eval()

    def predict_full_volume(
        self,
        volume_tensor_or_array: Union[torch.Tensor, np.ndarray],
        overlap: float = 0.25,
        sw_batch_size: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Runs sliding-window inference with Gaussian blending across full 3D volume.
        
        Args:
            volume_tensor_or_array: Array or Tensor of shape (4, H, W, D) or (1, 4, H, W, D)
            overlap: Overlap fraction between adjacent patches (0.25 - 0.5)
            sw_batch_size: Batch size of patches processed per forward pass
        Returns:
            Dictionary with:
            - 'probabilities': (3, H, W, D) float32 in [0, 1]
            - 'binary_3channel': (3, H, W, D) uint8 [WT, TC, ET]
            - 'reconstructed_seg': (H, W, D) uint8 BraTS labels (0, 1, 2, 4)
        """
        if isinstance(volume_tensor_or_array, np.ndarray):
            vol = torch.from_numpy(volume_tensor_or_array).float()
        else:
            vol = volume_tensor_or_array.clone().float()

        if vol.ndim == 4:
            vol = vol.unsqueeze(0)  # Shape (1, 4, H, W, D)

        vol = vol.to(self.device)

        # Pad if volume dimension is smaller than patch size
        pad_h = max(0, self.patch_size[0] - vol.shape[2])
        pad_w = max(0, self.patch_size[1] - vol.shape[3])
        pad_d = max(0, self.patch_size[2] - vol.shape[4])
        
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            vol = torch.nn.functional.pad(vol, (0, pad_d, 0, pad_w, 0, pad_h), mode="constant", value=0)

        with torch.no_grad():
            logits = sliding_window_inference(
                inputs=vol,
                roi_size=self.patch_size,
                sw_batch_size=sw_batch_size,
                predictor=self.model,
                overlap=overlap,
                mode="gaussian"
            )
            probs = self.post_sigmoid(logits)
            binary = (probs > 0.5).to(torch.uint8)

        # Unpad if padded
        h, w, d = volume_tensor_or_array.shape[-3:]
        probs_np = probs[0, :, :h, :w, :d].cpu().numpy()
        binary_np = binary[0, :, :h, :w, :d].cpu().numpy()
        
        # Convert to standard BraTS 0, 1, 2, 4 label mask
        reconstructed_seg = convert_brats_classes_to_labels(binary_np)

        return {
            "probabilities": probs_np,
            "binary_3channel": binary_np,
            "reconstructed_seg": reconstructed_seg
        }


def compute_clinical_validation_metrics(
    y_pred_3channel: Union[torch.Tensor, np.ndarray],
    y_true_3channel: Union[torch.Tensor, np.ndarray],
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Dict[str, Dict[str, float]]:
    """
    Computes spatial clinical metrics:
    - Dice Similarity Coefficient (DSC)
    - 95th Percentile Hausdorff Distance (HD95 in mm)
    - Sensitivity (Recall) and Specificity
    
    Sub-regions: Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET).
    """
    if isinstance(y_pred_3channel, np.ndarray):
        pred_t = torch.from_numpy(y_pred_3channel).float()
    else:
        pred_t = y_pred_3channel.float()
        
    if isinstance(y_true_3channel, np.ndarray):
        true_t = torch.from_numpy(y_true_3channel).float()
    else:
        true_t = y_true_3channel.float()

    if pred_t.ndim == 4:
        pred_t = pred_t.unsqueeze(0)
    if true_t.ndim == 4:
        true_t = true_t.unsqueeze(0)

    # Dice Metric
    dice_fn = DiceMetric(include_background=True, reduction="none")
    dice_vals = dice_fn(y_pred=pred_t, y=true_t)[0]

    # HD95 Metric
    hd95_fn = HausdorffDistanceMetric(
        include_background=True, 
        percentile=95, 
        reduction="none"
    )
    try:
        hd95_vals = hd95_fn(y_pred=pred_t, y=true_t, spacing=voxel_spacing)[0]
    except Exception:
        # Fallback if an empty subregion prevents surface distance calculation
        hd95_vals = torch.full((3,), float("nan"))

    subregions = ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"]
    metrics_report = {}

    for i, name in enumerate(subregions):
        d_val = float(dice_vals[i].item()) if not torch.isnan(dice_vals[i]) else 0.0
        h_val = float(hd95_vals[i].item())  # stays NaN when not computable
        
        # Calculate sensitivity (TP / (TP + FN))
        tp = float(torch.sum((pred_t[0, i] > 0.5) & (true_t[0, i] > 0.5)).item())
        fn = float(torch.sum((pred_t[0, i] <= 0.5) & (true_t[0, i] > 0.5)).item())
        fp = float(torch.sum((pred_t[0, i] > 0.5) & (true_t[0, i] <= 0.5)).item())
        tn = float(torch.sum((pred_t[0, i] <= 0.5) & (true_t[0, i] <= 0.5)).item())
        
        sensitivity = tp / (tp + fn + 1e-6)
        specificity = tn / (tn + fp + 1e-6)

        metrics_report[name] = {
            "Dice_Similarity_Coefficient": round(d_val, 4),
            "HD95_mm": round(h_val, 2),
            "Sensitivity": round(sensitivity, 4),
            "Specificity": round(specificity, 4)
        }

    # Mean Dice across sub-regions
    mean_dice = sum(m["Dice_Similarity_Coefficient"] for m in metrics_report.values()) / 3.0
    _hv = [m["HD95_mm"] for m in metrics_report.values() if m["HD95_mm"] == m["HD95_mm"]]
    mean_hd95 = sum(_hv) / len(_hv) if _hv else float("nan")
    metrics_report["Overall_Mean"] = {
        "Dice_Similarity_Coefficient": round(mean_dice, 4),
        "HD95_mm": round(mean_hd95, 2)
    }

    return metrics_report