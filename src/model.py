"""
NeuroSeg - Module 2: 3D Deep Learning Model Architecture & Loss Setup
Implements:
- 3D SegResNet encoder-decoder segmentation backbone
- 3D nnU-Net style DynUNet / UNet backbone
- Custom Composite Dice-Focal Loss with sub-region weighting (WT, TC, ET)
- Forward sub-region remapping (BraTS integers -> WT, TC, ET channels)
- Inverse remapping (WT, TC, ET channels -> BraTS standard integer mask 0, 1, 2, 4)
- Model summary & parameter analysis
"""

import warnings
from typing import Union, Tuple, Dict
# pyrefly: ignore[missing-import]
import numpy as np
# pyrefly: ignore[missing-import]
import torch
# pyrefly: ignore[missing-import]
import torch.nn as nn

# Suppress PyTorch JIT deprecation warnings emitted by MONAI
warnings.filterwarnings("ignore", category=FutureWarning)

# pyrefly: ignore[missing-import]
from monai.networks.nets import SegResNet, UNet
# pyrefly: ignore[missing-import]
from monai.losses import DiceFocalLoss


def get_segresnet_model(
    in_channels: int = 4, 
    out_channels: int = 3, 
    init_filters: int = 16,
    dropout_prob: float = 0.2
) -> nn.Module:
    """
    Constructs a 3D SegResNet architecture for brain tumor segmentation.
    - in_channels=4: (T1, T1CE, T2, FLAIR)
    - out_channels=3: Independent multi-label channels (WT, TC, ET)
    - Residual blocks with group normalization and spatial dropout
    """
    model = SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=init_filters,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        dropout_prob=dropout_prob
    )
    return model


def get_nnunet_model(
    in_channels: int = 4, 
    out_channels: int = 3,
    init_filters: int = 16
) -> nn.Module:
    """
    Constructs a 3D nnU-Net style residual U-Net architecture.
    Features 5-level deep hierarchy with residual skip-connections.
    """
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(init_filters, init_filters * 2, init_filters * 4, init_filters * 8, init_filters * 16),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
        dropout=0.15
    )
    return model


def build_segmentation_model(
    model_name: str = "segresnet",
    in_channels: int = 4,
    out_channels: int = 3,
    init_filters: int = 16
) -> nn.Module:
    """Factory function to build specified 3D segmentation backbone."""
    name = model_name.lower()
    if "segresnet" in name:
        return get_segresnet_model(in_channels=in_channels, out_channels=out_channels, init_filters=init_filters)
    elif "unet" in name or "nnunet" in name:
        return get_nnunet_model(in_channels=in_channels, out_channels=out_channels, init_filters=init_filters)
    else:
        raise ValueError(f"Unknown architecture: {model_name}. Supported: 'segresnet', 'nnunet'")


class CompositeDiceFocalLoss(nn.Module):
    """
    Clinical Composite Loss Function:
    Combines Soft-Dice Loss (boundary & volumetric overlap) with 
    Focal Cross-Entropy (hard voxel mining & extreme foreground-background class imbalance).
    
    Includes sub-region weighting:
    Enhancing Tumor (ET) is typically tiny (<2% of brain voxels) and receives higher weighting
    to prevent gradient vanishing.
    """
    def __init__(
        self, 
        lambda_dice: float = 1.0, 
        lambda_focal: float = 1.0,
        subregion_weights: Tuple[float, float, float] = (1.0, 1.2, 1.5)
    ):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.register_buffer("subregion_weights", torch.tensor(subregion_weights, dtype=torch.float32))

        # Channel-wise computation per sub-region
        self.dice_focal = DiceFocalLoss(
            sigmoid=True,
            squared_pred=True,
            reduction="mean",
            lambda_dice=lambda_dice,
            lambda_focal=lambda_focal
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: (B, 3, H, W, D) logits
        # target: (B, 3, H, W, D) binary labels
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        num_channels = pred.shape[1]

        for c in range(num_channels):
            channel_loss = self.dice_focal(pred[:, c:c+1], target[:, c:c+1])
            total_loss = total_loss + self.subregion_weights[c] * channel_loss

        return total_loss / self.subregion_weights.sum()


def get_loss_function(subregion_weights: Tuple[float, float, float] = (1.0, 1.2, 1.5)) -> nn.Module:
    """Returns the configured composite loss function."""
    return CompositeDiceFocalLoss(subregion_weights=subregion_weights)


def convert_labels_to_brats_classes(label_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts raw BraTS label integers (1: Necrotic, 2: Edema, 4: Enhancing)
    into 3 multi-label binary evaluation channels:
    - Channel 0: Whole Tumor (WT)  -> Labels 1, 2, 4
    - Channel 1: Tumor Core (TC)   -> Labels 1, 4
    - Channel 2: Enhancing (ET)    -> Label 4
    
    Input shape:  (B, 1, H, W, D) or (1, H, W, D)
    Output shape: (B, 3, H, W, D) or (3, H, W, D)
    """
    wt = (label_tensor == 1) | (label_tensor == 2) | (label_tensor == 4)
    tc = (label_tensor == 1) | (label_tensor == 4)
    et = (label_tensor == 4)
    
    dim = 1 if label_tensor.ndim == 5 else 0
    return torch.cat([wt, tc, et], dim=dim).float()


def convert_brats_classes_to_labels(pred_3channel: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Inverse Mapping:
    Reconstructs standard BraTS integer labels (0, 1, 2, 4) from 3 binary channels [WT, TC, ET]:
    - 0: Background
    - 1: Necrotic Core (TC without ET)
    - 2: Peritumoral Edema (WT without TC)
    - 4: Enhancing Tumor (ET)
    
    Supports both PyTorch tensors and NumPy arrays.
    """
    if isinstance(pred_3channel, torch.Tensor):
        has_batch = (pred_3channel.ndim == 5)
        if not has_batch:
            pred_3channel = pred_3channel.unsqueeze(0)
            
        wt = pred_3channel[:, 0] > 0.5
        tc = pred_3channel[:, 1] > 0.5
        et = pred_3channel[:, 2] > 0.5
        
        reconstructed = torch.zeros_like(wt, dtype=torch.uint8)
        reconstructed[wt] = 2          # Edema
        reconstructed[tc] = 1          # Necrotic Core
        reconstructed[et] = 4          # Enhancing Margin
        
        return reconstructed if has_batch else reconstructed[0]
    else:
        wt = pred_3channel[0] > 0.5
        tc = pred_3channel[1] > 0.5
        et = pred_3channel[2] > 0.5
        
        reconstructed = np.zeros(wt.shape, dtype=np.uint8)
        reconstructed[wt] = 2
        reconstructed[tc] = 1
        reconstructed[et] = 4
        return reconstructed


def get_model_summary(model: nn.Module) -> Dict[str, Union[int, float, str]]:
    """Calculates parameter count and estimated memory footprint."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = (total_params * 4) / (1024 * 1024)
    
    return {
        "model_architecture": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": round(model_size_mb, 2)
    }