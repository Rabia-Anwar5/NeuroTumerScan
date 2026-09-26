import glob
import os
# pyrefly: ignore [missing-import]
from monai.data import DataLoader, Dataset
# pyrefly: ignore [missing-import]
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapLabelValued,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    Spacingd,
)


def get_brats_data_dicts(data_dir: str):
  """Discovers and structures multi-sequence patient scans from the dataset directory."""
  patient_dirs = sorted(
      [
          os.path.join(data_dir, d)
          for d in os.listdir(data_dir)
          if os.path.isdir(os.path.join(data_dir, d))
      ]
  )
  data_dicts = []

  for p_dir in patient_dirs:
    p_id = os.path.basename(p_dir)
    t1 = glob.glob(os.path.join(p_dir, "*t1.nii*"))
    t1ce = glob.glob(os.path.join(p_dir, "*t1ce.nii*")) or glob.glob(
        os.path.join(p_dir, "*t1gd.nii*")
    )
    t2 = glob.glob(os.path.join(p_dir, "*t2.nii*"))
    flair = glob.glob(os.path.join(p_dir, "*flair.nii*"))
    seg = glob.glob(os.path.join(p_dir, "*seg.nii*"))

    if t1 and t1ce and t2 and flair and seg:
      data_dicts.append({
          "image": [t1[0], t1ce[0], t2[0], flair[0]],  # 4-channel input
          "label": seg[0],
          "patient_id": p_id,
      })
  return data_dicts


def build_module1_transforms(patch_size=(128, 128, 128)):
  """Constructs the deterministic preprocessing and patch extraction pipeline."""
  train_transforms = Compose([
      # 1. Load multi-sequence 4D images and ground truth
      LoadImaged(keys=["image", "label"]),
      EnsureChannelFirstd(keys=["image", "label"]),
      # 2. Standardize spatial orientation and voxel spacing
      Orientationd(keys=["image", "label"], axcodes="RAS"),
      Spacingd(
          keys=["image", "label"],
          pixdim=(1.0, 1.0, 1.0),
          mode=("bilinear", "nearest"),
      ),
      # 3. Z-score intensity normalization per sequence (non-zero foreground only)
      NormalizeIntensityd(
          keys="image", nonzero=True, channel_wise=True
      ),  # 4. Patch extraction with 2:1 positive-to-negative sample ratio
      RandCropByPosNegLabeld(
          keys=["image", "label"],
          label_key="label",
          spatial_size=patch_size,
          pos=2,
          neg=1,
          num_samples=2,
          image_key="image",
          image_threshold=0,
      ),
      # 5. Spatial augmentation
      RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
      RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
      RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
      EnsureTyped(keys=["image", "label"], track_meta=False),
  ])
  return train_transforms


if __name__ == "__main__":
  DATA_DIR = "./data/BraTS2021_Training_Data"

  # Quick sanity test on mock or local data
  if os.path.exists(DATA_DIR):
    data_list = get_brats_data_dicts(DATA_DIR)
    print(f"Found {len(data_list)} valid multi-sequence cases.")

    transforms = build_module1_transforms()
    ds = Dataset(data=data_list[:2], transform=transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=True)

    sample = next(iter(loader))
    print("Preprocessing verification successful:")
    print(" - Input image batch shape (B, C, H, W, D):", sample["image"].shape)
    print(" - Label mask batch shape  (B, 1, H, W, D):", sample["label"].shape)
  else:
    print(
        f"Path '{DATA_DIR}' not detected. Ensure dataset folder is unzipped and properly linked."
    )