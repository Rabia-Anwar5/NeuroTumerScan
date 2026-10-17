"""
NeuroSeg - Module 2 Verification Suite:
Tests 3D SegResNet and nnU-Net backbones, Composite Dice-Focal Loss,
Forward and Inverse Label Conversions, and Gradient Backpropagation.
"""

import os
import sys
import warnings
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from src.model import (
    get_segresnet_model,
    get_nnunet_model,
    build_segmentation_model,
    get_loss_function,
    convert_labels_to_brats_classes,
    convert_brats_classes_to_labels,
    get_model_summary
)


def test_module2():
    print("=" * 65)
    print("  VERIFYING MODULE 2: 3D MODEL ARCHITECTURES & LOSS SETUP")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Executing on target device: {device}")

    # 1. Test SegResNet backbone
    segresnet = build_segmentation_model("segresnet", in_channels=4, out_channels=3, init_filters=16).to(device)
    sr_summary = get_model_summary(segresnet)
    print(f" [1/5] SegResNet Backbone Initialized:")
    print(f"       Total Parameters: {sr_summary['total_parameters']:,} ({sr_summary['model_size_mb']} MB)")
    assert sr_summary['total_parameters'] > 1_000_000, "SegResNet parameter count unexpectedly small."

    # 2. Test nnU-Net / UNet backbone
    nnunet = build_segmentation_model("nnunet", in_channels=4, out_channels=3, init_filters=16).to(device)
    nn_summary = get_model_summary(nnunet)
    print(f" [2/5] nnU-Net 3D Backbone Initialized:")
    print(f"       Total Parameters: {nn_summary['total_parameters']:,} ({nn_summary['model_size_mb']} MB)")
    assert nn_summary['total_parameters'] > 1_000_000, "nnU-Net parameter count unexpectedly small."

    # 3. Test Forward Pass on 3D Patch: (B=1, C=4, H=64, W=64, D=64) (using 64 for fast CPU unit test)
    dummy_input = torch.randn(1, 4, 64, 64, 64, device=device)
    segresnet.eval()
    with torch.no_grad():
        output_logits = segresnet(dummy_input)
    print(f" [3/5] Forward Pass Output Logits Shape: {output_logits.shape}")
    assert output_logits.shape == (1, 3, 64, 64, 64), f"Unexpected output shape: {output_logits.shape}"

    # 4. Test Forward & Inverse Label Mapping
    dummy_raw_label = torch.zeros(1, 1, 64, 64, 64, dtype=torch.long, device=device)
    dummy_raw_label[:, :, 20:45, 20:45, 20:45] = 2  # Edema
    dummy_raw_label[:, :, 25:35, 25:35, 25:35] = 1  # Necrotic Core
    dummy_raw_label[:, :, 28:32, 28:32, 28:32] = 4  # Enhancing Margin

    # Forward mapping
    target_3ch = convert_labels_to_brats_classes(dummy_raw_label)
    assert target_3ch.shape == (1, 3, 64, 64, 64), f"Unexpected 3-channel shape: {target_3ch.shape}"
    
    # Inverse mapping
    reconstructed_label = convert_brats_classes_to_labels(target_3ch)
    assert reconstructed_label.shape == (1, 64, 64, 64)
    # Check that labels 1, 2, 4 are properly recovered
    recovered_vals = torch.unique(reconstructed_label).tolist()
    print(f" [4/5] Sub-Region Label Remapping Verified:")
    print(f"       Forward Target Channels: {target_3ch.shape} (WT, TC, ET)")
    print(f"       Inverse Recovered Classes: {recovered_vals} (Expected: [0, 1, 2, 4])")
    assert set(recovered_vals) == {0, 1, 2, 4}, f"Recovered classes mismatch: {recovered_vals}"

    # 5. Test Composite Loss & Gradient Backward Pass
    segresnet.train()
    optimizer = torch.optim.AdamW(segresnet.parameters(), lr=1e-4)
    loss_fn = get_loss_function(subregion_weights=(1.0, 1.2, 1.5))

    optimizer.zero_grad()
    train_out = segresnet(dummy_input)
    loss = loss_fn(train_out, target_3ch)
    loss.backward()
    optimizer.step()

    print(f" [5/5] Composite Dice-Focal Loss Computed: {loss.item():.4f}")
    print("       Gradient Backpropagation Successful!")
    assert not torch.isnan(loss), "Loss computed NaN."
    assert loss.item() > 0.0, "Loss must be positive."

    print(" >>> MODULE 2 VERIFICATION PASSED <<<\n")


if __name__ == "__main__":
    test_module2()