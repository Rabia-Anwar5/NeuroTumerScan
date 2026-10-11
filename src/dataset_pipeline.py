import os
import glob
import warnings

# Suppress PyTorch JIT deprecation warning from MONAI
warnings.filterwarnings("ignore", category=FutureWarning)

# pyrefly: ignore [missing-import]
from monai.data import Dataset, DataLoader
# pyrefly: ignore [missing-import]
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    EnsureTyped
)

def load_brats_file_list(data_dir: str):
    """
    Scans the directory for patient folders containing BraTS MRI sequences.
    """
    if not os.path.exists(data_dir):
        return []

    patient_dirs = sorted([
        os.path.join(data_dir, d) for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    data_list = []
    
    for p in patient_dirs:
        p_id = os.path.basename(p)
        t1 = glob.glob(os.path.join(p, "*t1.nii*"))
        t1ce = glob.glob(os.path.join(p, "*t1ce.nii*")) or glob.glob(os.path.join(p, "*t1gd.nii*"))
        t2 = glob.glob(os.path.join(p, "*t2.nii*"))
        flair = glob.glob(os.path.join(p, "*flair.nii*"))
        seg = glob.glob(os.path.join(p, "*seg.nii*"))
        
        if t1 and t1ce and t2 and flair and seg:
            data_list.append({
                "image": [t1[0], t1ce[0], t2[0], flair[0]],
                "label": seg[0],
                "patient_id": p_id
            })
            
    return data_list

def get_module1_transforms(patch_size=(128, 128, 128)):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest")
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=2,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        EnsureTyped(keys=["image", "label"], track_meta=False)
    ])

def get_brats_dataloader(data_list, batch_size=1, patch_size=(128, 128, 128), shuffle=True):
    transforms = get_module1_transforms(patch_size=patch_size)
    dataset = Dataset(data=data_list, transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Backward-compatibility aliases for Module 1
get_brats_data_dicts = load_brats_file_list
build_module1_transforms = get_module1_transforms