import os
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import nibabel as nib
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from dataset_pipeline import get_brats_data_dicts, build_module1_transforms
# pyrefly: ignore [missing-import]
from monai.data import DataLoader, Dataset


# 1. Setup local mock directory for verification
test_dir = "./data/BraTS2021_Sample/BraTS2021_00001"
os.makedirs(test_dir, exist_ok=True)
affine = np.eye(4)
shape = (140, 140, 140)

# Simulate 4 modalities
for seq in ["t1", "t1ce", "t2", "flair"]:
    vol = (np.random.randn(*shape) * 25 + 100).astype(np.float32)
    vol[vol < 0] = 0
    nib.save(nib.Nifti1Image(vol, affine), f"{test_dir}/BraTS2021_00001_{seq}.nii.gz")

# Simulate segmentation mask
mask = np.zeros(shape, dtype=np.uint8)
mask[50:85, 50:85, 50:85] = 1   # Necrotic core
mask[40:95, 40:95, 40:95] = 2   # Edema
mask[60:80, 60:80, 60:80] = 4   # Enhancing margin
nib.save(nib.Nifti1Image(mask, affine), f"{test_dir}/BraTS2021_00001_seg.nii.gz")

print("Local test scan files prepared.")

# 2. Run dataloader pipeline
file_list = get_brats_data_dicts("./data/BraTS2021_Sample")
transforms = build_module1_transforms(patch_size=(128, 128, 128))
ds = Dataset(data=file_list, transform=transforms)
loader = DataLoader(ds, batch_size=1, shuffle=True)

batch = next(iter(loader))
img_patch = batch["image"][0].numpy()
seg_patch = batch["label"][0, 0].numpy()

print("\n--- Local Pipeline Verification Successful ---")
print(f"Batch Image Tensor (C, H, W, D): {img_patch.shape}")
print(f"Batch Label Tensor (H, W, D):    {seg_patch.shape}")
print(f"Foreground Mean:                 {img_patch[img_patch != 0].mean():.4f}")
print(f"Foreground Std:                  {img_patch[img_patch != 0].std():.4f}")

# 3. Save verification image locally
slice_z = 64
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
titles = ["T1 (Anatomy)", "T1CE (Active Tumor)", "T2 (Edema)", "FLAIR (Infiltration)", "Ground Truth Overlay"]

for i in range(4):
    axes[i].imshow(img_patch[i, :, :, slice_z], cmap="gray")
    axes[i].set_title(titles[i], fontsize=11, fontweight="bold")
    axes[i].axis("off")

axes[4].imshow(img_patch[1, :, :, slice_z], cmap="gray")
axes[4].imshow(seg_patch[:, :, slice_z], cmap="tab10", alpha=0.55)
axes[4].set_title(titles[4], fontsize=11, fontweight="bold")
axes[4].axis("off")

os.makedirs("./outputs", exist_ok=True)
plt.tight_layout()
plt.savefig("./outputs/Module_1_Verification.png", dpi=300)
print("Saved plot to './outputs/Module_1_Verification.png'")