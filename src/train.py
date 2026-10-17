"""
NeuroSeg - Module 3: Model Optimization, Validation & Sliding-Window Training Engine
Implements:
- 3D SegResNet / UNet optimization with AdamW & Cosine Annealing LR
- Deep supervision and composite Dice-Focal Loss
- Sliding-window validation with Gaussian blending
- Multi-region spatial evaluation (Dice Similarity Coefficient & HD95)
- Model checkpointing & structured training history export
"""

import os
import sys
import json
import warnings
from typing import Optional, Dict, Any

# Suppress PyTorch JIT deprecation warnings emitted by MONAI
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from monai.inferers import sliding_window_inference
# pyrefly: ignore [missing-import]
from monai.metrics import DiceMetric, HausdorffDistanceMetric
# pyrefly: ignore [missing-import]
from monai.transforms import Activations, AsDiscrete, Compose

try:
    from src.model import get_segresnet_model, get_loss_function, convert_labels_to_brats_classes
    from src.dataset_pipeline import load_brats_file_list, get_brats_dataloader
    from src.inference import compute_clinical_validation_metrics
except ModuleNotFoundError:
    from model import get_segresnet_model, get_loss_function, convert_labels_to_brats_classes
    from dataset_pipeline import load_brats_file_list, get_brats_dataloader
    from inference import compute_clinical_validation_metrics


def train_and_validate(
    data_dir: str,
    num_epochs: int = 3,
    batch_size: int = 1,
    patch_size: tuple = (128, 128, 128),
    lr: float = 1e-4,
    checkpoint_dir: str = "./outputs/checkpoints"
) -> Dict[str, Any]:
    """
    Executes deep learning training sweeps and sliding-window validation.
    
    Args:
        data_dir: Path to BraTS patient directories
        num_epochs: Number of training epochs
        batch_size: Training patch batch size
        patch_size: 3D patch resolution (H, W, D)
        lr: Initial learning rate
        checkpoint_dir: Directory where best weights and logs are stored
    Returns:
        Dictionary summarizing best validation metrics and checkpoint path
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" NeuroSeg Deep Learning Engine Initialized on: {device}")
    print(f"=======================================================")

    # 1. Dataset & Dataloaders
    file_list = load_brats_file_list(data_dir)
    if len(file_list) == 0:
        raise ValueError(f"No valid BraTS patient data detected in {data_dir}")

    print(f"Loaded {len(file_list)} patient case(s) for training and validation.")
    # Hold out ~20% of patients for validation (fixed seed). With fewer than 5 cases a
    # split is impossible, so validation falls back to the training cases and the run is
    # flagged as NOT held-out (the app shows this warning next to the metrics).
    import random
    held_out = len(file_list) >= 5
    if held_out:
        shuffled = list(file_list)
        random.Random(42).shuffle(shuffled)
        n_val = max(1, int(0.2 * len(shuffled)))
        val_files, train_files = shuffled[:n_val], shuffled[n_val:]
    else:
        print("WARNING: fewer than 5 cases - validating on the training cases. "
              "These scores are NOT a held-out evaluation.")
        train_files = val_files = file_list
    print(f"Train cases: {len(train_files)} | Validation cases: {len(val_files)} | held-out: {held_out}")
    train_loader = get_brats_dataloader(train_files, batch_size=batch_size, patch_size=patch_size, shuffle=True)
    val_loader = get_brats_dataloader(val_files, batch_size=1, patch_size=patch_size, shuffle=False, training=False)

    # 2. Model, Loss, Optimizer, Scheduler
    model = get_segresnet_model(in_channels=4, out_channels=3, init_filters=16).to(device)
    loss_fn = get_loss_function(subregion_weights=(1.0, 1.2, 1.5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # 3. Post-processing transforms and validation metric calculators
    post_trans = Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold=0.5)
    ])
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_metric = -1.0
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_segresnet_metric.pth")
    history_log = []

    print("\n--- Commencing Optimization Sweeps ---")

    for epoch in range(1, num_epochs + 1):
        # ----------------- TRAINING PHASE -----------------
        model.train()
        epoch_loss = 0.0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            targets = convert_labels_to_brats_classes(labels)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            
            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= max(step, 1)
        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] - Loss: {epoch_loss:.4f} | LR: {current_lr:.6f}")

        # ----------------- VALIDATION PHASE -----------------
        model.eval()
        dice_metric.reset()
        hd95_metric.reset()

        with torch.no_grad():
            for val_data in val_loader:
                val_inputs = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)
                val_targets = convert_labels_to_brats_classes(val_labels)

                # Sliding window inference with Gaussian blending
                val_outputs = sliding_window_inference(
                    inputs=val_inputs,
                    roi_size=patch_size,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.25,
                    mode="gaussian"
                )

                val_outputs_bin = post_trans(val_outputs)
                dice_metric(y_pred=val_outputs_bin, y=val_targets)
                try:
                    hd95_metric(y_pred=val_outputs_bin, y=val_targets)
                except Exception:
                    pass

            # Aggregate validation scores
            dice_scores = dice_metric.aggregate()
            wt_dice = float(dice_scores[0].item()) if not torch.isnan(dice_scores[0]) else 0.0
            tc_dice = float(dice_scores[1].item()) if not torch.isnan(dice_scores[1]) else 0.0
            et_dice = float(dice_scores[2].item()) if not torch.isnan(dice_scores[2]) else 0.0
            mean_dice = (wt_dice + tc_dice + et_dice) / 3.0

            # HD95 is None when it could not be computed (e.g. an empty sub-region).
            # Never substitute a made-up number: report it as unavailable.
            def _hd(scores, i):
                try:
                    v = float(scores[i].item())
                    return None if (v != v or v == float("inf")) else v
                except Exception:
                    return None

            try:
                hd_scores = hd95_metric.aggregate()
            except Exception:
                hd_scores = None
            wt_hd95, tc_hd95, et_hd95 = [_hd(hd_scores, i) if hd_scores is not None else None for i in range(3)]
            _valid_hd = [v for v in (wt_hd95, tc_hd95, et_hd95) if v is not None]
            mean_hd95 = sum(_valid_hd) / len(_valid_hd) if _valid_hd else None

            print(f"    Val DSC  -> WT: {wt_dice:.4f} | TC: {tc_dice:.4f} | ET: {et_dice:.4f} | Mean: {mean_dice:.4f}")
            _f = lambda v: "n/a" if v is None else f"{v:.2f}mm"
            _r = lambda v: None if v is None else round(v, 2)
            print(f"    Val HD95 -> WT: {_f(wt_hd95)} | TC: {_f(tc_hd95)} | ET: {_f(et_hd95)} | Mean: {_f(mean_hd95)}")

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(epoch_loss, 4),
                "dice": {"WT": round(wt_dice, 4), "TC": round(tc_dice, 4), "ET": round(et_dice, 4), "mean": round(mean_dice, 4)},
                "hd95_mm": {"WT": _r(wt_hd95), "TC": _r(tc_hd95), "ET": _r(et_hd95), "mean": _r(mean_hd95)},
                "held_out": held_out,
                "val_cases": len(val_files),
                "learning_rate": current_lr
            }
            history_log.append(epoch_record)

            # Checkpoint management
            if mean_dice > best_metric:
                best_metric = mean_dice
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"    [BEST] New best validation score ({best_metric:.4f}). Checkpoint saved to: {best_checkpoint_path}")

    # Persist training history JSON
    history_file = os.path.join(checkpoint_dir, "training_history.json")
    with open(history_file, "w") as f:
        json.dump(history_log, f, indent=2)
    print(f"\nTraining complete. Full metrics history saved to: {history_file}")

    return {
        "best_mean_dice": best_metric,
        "best_checkpoint": best_checkpoint_path,
        "history_file": history_file
    }


if __name__ == "__main__":
    DATA_PATH = "./data/BraTS2021_Training_Data"
    if not os.path.exists(DATA_PATH) or not any(os.scandir(DATA_PATH)):
        DATA_PATH = "./data/BraTS2021_Sample"
    train_and_validate(DATA_PATH, num_epochs=3, batch_size=1, patch_size=(128, 128, 128))