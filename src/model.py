# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from monai.networks.nets import SegResNet, UNet
# pyrefly: ignore [missing-import]
from monai.losses import DiceFocalLoss

def get_segresnet_model(in_channels=4, out_channels=3, init_filters=16):
    """
    3D SegResNet Architecture:
    - in_channels=4: (T1, T1CE, T2, FLAIR)
    - out_channels=3: Multi-label sigmoid channels (WT, TC, ET)
    """
    model = SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=init_filters,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        dropout_prob=0.2
    )
    return model

def get_loss_function():
    """
    Composite Loss: Dice Loss + Focal Loss
    Handles extreme voxel imbalance between tiny tumor foci and background brain volume.
    """
    loss_fn = DiceFocalLoss(
        sigmoid=True,          # Multi-label sub-regions use independent sigmoid activations
        squared_pred=True,
        reduction="mean",
        lambda_dice=1.0,
        lambda_focal=1.0
    )
    return loss_fn

def convert_labels_to_brats_classes(label_tensor):
    """
    Converts raw BraTS label integers (1, 2, 4) into 3 binary channels:
    Channel 0: Whole Tumor (WT)  -> Labels 1, 2, 4
    Channel 1: Tumor Core (TC)   -> Labels 1, 4
    Channel 2: Enhancing (ET)    -> Label 4
    Input shape:  (B, 1, H, W, D)
    Output shape: (B, 3, H, W, D)
    """
    wt = (label_tensor == 1) | (label_tensor == 2) | (label_tensor == 4)
    tc = (label_tensor == 1) | (label_tensor == 4)
    et = (label_tensor == 4)
    
    return torch.cat([wt, tc, et], dim=1).float()