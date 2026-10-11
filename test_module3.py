import os
import sys
import warnings

# Suppress PyTorch JIT deprecation warnings emitted by MONAI
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure project root and src/ are in Python's search path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import nibabel as nib
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
from monai.inferers import sliding_window_inference
# pyrefly: ignore [missing-import]
from monai.transforms import Activations, AsDiscrete, Compose
# pyrefly: ignore [missing-import]
from monai.metrics import DiceMetric

# Safe dynamic import
try:
    from src.model import get_segresnet_model, get_loss_function, convert_labels_to_brats_classes
except ModuleNotFoundError:
    # pyrefly: ignore [missing-import]
    from model import get_segresnet_model, get_loss_function, convert_labels_to_brats_classes

try:
    from src.dataset_pipeline import load_brats_file_list, get_brats_dataloader
except ModuleNotFoundError:
    # pyrefly: ignore [missing-import]
    from dataset_pipeline import load_brats_file_list, get_brats_dataloader


def ensure_mock_sample_exists(sample_dir):
    """Ensures mock data is available for pipeline verification."""
    patient_dir = os.path.join(sample_dir, "BraTS2021_00001")
    os.makedirs(patient_dir, exist_ok=True)
    required_files = ["t1", "t1ce", "t2", "flair", "seg"]
    all_exist = all(
        len(os.listdir(patient_dir)) >= 5
        and all(os.path.getsize(os.path.join(patient_dir, f)) > 0 for f in os.listdir(patient_dir))
        for _ in [1]
    ) if os.path.exists(patient_dir) and os.listdir(patient_dir) else False

    if not all_exist:
        affine = np.eye(4)
        shape = (140, 140, 140)
        for seq in ["t1", "t1ce", "t2", "flair"]:
            vol = (np.random.randn(*shape) * 25 + 100).astype(np.float32)
            vol[vol < 0] = 0
            nib.save(nib.Nifti1Image(vol, affine), os.path.join(patient_dir, f"BraTS2021_00001_{seq}.nii.gz"))
        
        mask = np.zeros(shape, dtype=np.uint8)
        mask[50:85, 50:85, 50:85] = 1   # Necrotic core
        mask[40:95, 40:95, 40:95] = 2   # Edema
        mask[60:80, 60:80, 60:80] = 4   # Enhancing margin
        nib.save(nib.Nifti1Image(mask, affine), os.path.join(patient_dir, "BraTS2021_00001_seg.nii.gz"))
        print(f"Generated sample BraTS scan data at: {patient_dir}")


def test_module3_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Module 3 verification on device: {device}")

    sample_dir = os.path.join(PROJECT_ROOT, "data", "BraTS2021_Sample")
    ensure_mock_sample_exists(sample_dir)

    # 1. Dataset & DataLoader verification
    file_list = load_brats_file_list(sample_dir)
    print(f"Discovered {len(file_list)} valid patient scan sequence(s).")
    assert len(file_list) > 0, "No valid BraTS patient data detected."

    loader = get_brats_dataloader(file_list, batch_size=1, patch_size=(128, 128, 128), shuffle=False)
    batch = next(iter(loader))

    inputs = batch["image"].to(device)
    labels = batch["label"].to(device)
    targets = convert_labels_to_brats_classes(labels)

    print("\n--- Pipeline Tensor Shapes ---")
    print(f"Input Image Tensor Shape:  {inputs.shape}")
    print(f"Multi-label Target Shape:  {targets.shape}")

    # 2. Instantiate Model, Loss Function, and Optimizer
    model = get_segresnet_model(in_channels=4, out_channels=3, init_filters=16).to(device)
    loss_fn = get_loss_function()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)

    # 3. Simulate Training Step (Forward, Loss, Backward, Optimizer Step)
    print("\n--- Training Step Verification ---")
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    print(f"Training Step Dice-Focal Loss: {loss.item():.4f}")
    print("Backward pass & optimizer step succeeded.")

    # 4. Simulate Validation with Sliding Window Inference
    print("\n--- Sliding-Window Validation Verification ---")
    model.eval()
    post_trans = Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold=0.5)
    ])
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")

    with torch.no_grad():
        val_outputs = sliding_window_inference(
            inputs=inputs,
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
            mode="gaussian"
        )
        val_outputs_bin = post_trans(val_outputs)
        dice_metric(y_pred=val_outputs_bin, y=targets)

    dice_scores = dice_metric.aggregate()
    wt_dice = dice_scores[0].item()
    tc_dice = dice_scores[1].item()
    et_dice = dice_scores[2].item()
    mean_dice = (wt_dice + tc_dice + et_dice) / 3.0

    print(f"Validation Dice -> WT: {wt_dice:.4f} | TC: {tc_dice:.4f} | ET: {et_dice:.4f} | Mean: {mean_dice:.4f}")

    # 5. Checkpoint Saving Verification
    print("\n--- Checkpoint Verification ---")
    checkpoints_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoints_dir, "best_segresnet_metric.pth")
    torch.save(model.state_dict(), checkpoint_path)
    assert os.path.exists(checkpoint_path), "Checkpoint failed to save"
    print(f"Checkpoint successfully saved to: {checkpoint_path}")

    # 6. Save Verification Visualizations
    outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    fig_path = os.path.join(outputs_dir, "Module_3_Verification.png")

    slice_z = 64
    t1ce_slice = inputs[0, 1, :, :, slice_z].cpu().numpy()
    gt_wt = targets[0, 0, :, :, slice_z].cpu().numpy()
    pred_wt = val_outputs_bin[0, 0, :, :, slice_z].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(t1ce_slice, cmap="gray")
    axes[0].set_title("Input T1CE Anatomical", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(t1ce_slice, cmap="gray")
    axes[1].imshow(gt_wt, cmap="autumn", alpha=0.55)
    axes[1].set_title("Ground Truth Whole Tumor (WT)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(t1ce_slice, cmap="gray")
    axes[2].imshow(pred_wt, cmap="cool", alpha=0.55)
    axes[2].set_title("Validation Prediction (WT)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved verification plot to: {fig_path}")

    print("\n--- Module 3 Verification Completed Successfully! ---")


if __name__ == "__main__":
    test_module3_pipeline()