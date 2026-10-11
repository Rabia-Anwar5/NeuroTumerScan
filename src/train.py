import os
import sys
import warnings

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
from monai.metrics import DiceMetric
# pyrefly: ignore [missing-import]
from monai.transforms import Activations, AsDiscrete, Compose

try:
    from src.model import get_segresnet_model, get_loss_function, convert_labels_to_brats_classes
    from src.dataset_pipeline import load_brats_file_list, get_brats_dataloader
except ModuleNotFoundError:
    # pyrefly: ignore [missing-import]
    from model import get_segresnet_model, get_loss_function, convert_labels_to_brats_classes
    # pyrefly: ignore [missing-import]
    from dataset_pipeline import load_brats_file_list, get_brats_dataloader


def train_and_validate(data_dir, num_epochs=3, batch_size=1, patch_size=(128, 128, 128), lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")

    # 1. Dataset & Dataloaders
    file_list = load_brats_file_list(data_dir)
    if len(file_list) == 0:
        raise ValueError(f"No valid BraTS patient data detected in {data_dir}")

    # For verification, use available cases
    train_files = file_list
    val_files = file_list

    train_loader = get_brats_dataloader(train_files, batch_size=batch_size, patch_size=patch_size, shuffle=True)
    val_loader = get_brats_dataloader(val_files, batch_size=1, patch_size=patch_size, shuffle=False)

    # 2. Model, Loss, Optimizer
    model = get_segresnet_model(in_channels=4, out_channels=3, init_filters=16).to(device)
    loss_fn = get_loss_function()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 3. Post-processing transforms for validation metric evaluation
    post_trans = Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold=0.5)
    ])
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")

    # 4. Training Loop
    print("\n--- Starting Training & Validation Run ---")
    best_metric = -1.0
    os.makedirs("./outputs/checkpoints", exist_ok=True)

    for epoch in range(1, num_epochs + 1):
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
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= max(step, 1)
        lr_scheduler.step()
        print(f"Epoch [{epoch}/{num_epochs}] - Training Loss: {epoch_loss:.4f}")

        # 5. Validation using Sliding-Window Inference
        model.eval()
        dice_metric.reset()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)
                val_targets = convert_labels_to_brats_classes(val_labels)

                # Sliding window inference evaluates the 3D volume using overlapping patches with Gaussian weighting
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

            # Compute Dice scores for WT, TC, and ET
            dice_scores = dice_metric.aggregate()
            wt_dice = dice_scores[0].item()
            tc_dice = dice_scores[1].item()
            et_dice = dice_scores[2].item()
            mean_dice = (wt_dice + tc_dice + et_dice) / 3.0

            print(f"Validation Dice -> WT: {wt_dice:.4f} | TC: {tc_dice:.4f} | ET: {et_dice:.4f} | Mean: {mean_dice:.4f}")

            # Save checkpoint
            if mean_dice > best_metric:
                best_metric = mean_dice
                checkpoint_path = "./outputs/checkpoints/best_segresnet_metric.pth"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"--> Saved best model checkpoint to {checkpoint_path}")

    print("\nTraining and validation complete.")
    return best_metric

if __name__ == "__main__":
    DATA_PATH = "./data/BraTS2021_Training_Data"
    # Fallback to local sample dataset if full training data has not been populated
    if not os.path.exists(DATA_PATH) or not any(os.scandir(DATA_PATH)):
        DATA_PATH = "./data/BraTS2021_Sample"
    train_and_validate(DATA_PATH, num_epochs=3, batch_size=1, patch_size=(128, 128, 128))