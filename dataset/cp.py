import csv
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import pandas as pd
from torchvision import transforms
from skimage.transform import resize
from nilearn import surface
import nibabel as nib
import argparse
import glob
import torchio as tio
from sklearn.model_selection import train_test_split
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    ResizeWithPadOrCropd,
    RandSpatialCropd,
    RandShiftIntensityd,
    RandRotated,
    ToTensord,
    CropForegroundd,
    Resized,
)
import torch
from torch.utils.data.dataset import Dataset

class CPDataset(Dataset):
    def __init__(self, tsv_path: str, augmentation: bool = False, sp_size: int = 128, d_size: int = 64):
        """
        Dataset for training with preprocessing and augmentation.

        Args:
            tsv_path (str): Path to the participants.tsv file.
            augmentation (bool): Whether to apply data augmentation.
            sp_size (int): Desired spatial size for the output images.
        """
        super().__init__()
        self.tsv_path = tsv_path
        self.augmentation = augmentation
        self.sp_size = sp_size
        self.d_size = d_size
        self.file_paths = self.get_data_files()

        print(f"Found {len(self.file_paths)} files in {self.tsv_path}")

    def get_data_files(self):
        """
        Load data files from participants.tsv.
        """
        participants = pd.read_csv(self.tsv_path, sep="\t")
        required_columns = ["sub_id_bids", "participant_id", "scan_id"]
        for col in required_columns:
            if col not in participants.columns:
                raise ValueError(f"Missing required column: {col}")
        
        unique_subjects = participants["sub_id_bids"].unique()
        train_subjects, _ = train_test_split(unique_subjects, test_size=0.2, random_state=42)

        data_files = []
        base_path = os.path.dirname(self.tsv_path)
        for _, row in participants.iterrows():
            if row["sub_id_bids"] in train_subjects:
                file_path = os.path.join(
                    base_path,
                    f"work_dir2/cbf2mni_wdir/{row['participant_id']}/{row['scan_id']}/wf/brainextraction/*_dtype.nii.gz"
                )
                matched_files = glob.glob(file_path)
                if len(matched_files) == 1:
                    data_files.append(matched_files[0])
                elif len(matched_files) > 1:
                    raise ValueError(f"Multiple files match pattern: {file_path}")
                else:
                    raise FileNotFoundError(f"No files found for: {file_path}")

        return data_files

    def roi_crop(self, image):
        """
        Crop the image to the region of interest (ROI) containing non-zero values.
        """
        # image = image[:, :, :, 0]  # Assume single-channel image
        mask = image > 0  # Binary mask of non-zero pixels
        coords = np.argwhere(mask)  # Coordinates of non-zero pixels

        # Compute bounding box
        x0, y0, z0 = coords.min(axis=0)
        x1, y1, z1 = coords.max(axis=0) + 1  # Bounding box dimensions

        # Crop to bounding box and pad to uniform size
        cropped = image[x0:x1, y0:y1, z0:z1]
        padded_crop = tio.CropOrPad(np.max(cropped.shape))(cropped.copy()[None])
        # print(cropped.shape, padded_crop.shape)
        padded_crop = np.transpose(padded_crop, (1, 2, 3, 0))  # Rearrange axes
        # print( padded_crop.shape)

        return padded_crop

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        """
        Preprocess and return a sample tensor.
        """
        # Load image
        image_path = self.file_paths[index]
        img = nib.load(image_path)
        # print(img.shape)
        img = np.swapaxes(img.get_data(), 0, 2)
        # print(img.shape)
        img = np.flip(img, 1)
        # print(img.shape)
        img = np.flip(img, 2)
        # print(img.shape)

        # Apply ROI cropping
        img = self.roi_crop(image=img)

        # Resize to uniform spatial size
        img = resize(img, (self.d_size, self.sp_size, self.sp_size), mode='constant')

        # Apply data augmentation
        if self.augmentation:
            random_n = torch.rand(1)
            random_i = 0.3 * torch.rand(1)[0] + 0.7
            if random_n[0] > 0.5:
                img = np.flip(img, 0)  # Random flip along the x-axis
            img = img * random_i.data.cpu().numpy()  # Random intensity scaling
        # Find min and max intensity for normalization
        a_min = img.min()
        a_max = img.max()

        # Normalize to [-1, 1]
        img = (img - a_min) / (a_max - a_min)  # Normalize to [0, 1]
        img = img * 2 - 1                      # Rescale to [-1, 1]
        # Normalize to [-1, 1]
        imageout = torch.from_numpy(img).float().view(1, self.d_size, self.sp_size, self.sp_size)
        # imageout = imageout * 2 - 1

        return {'data': imageout}
# def threshold_at_zero(x):
#     """
#     Custom threshold function for CropForeground.
#     Thresholds intensity values at 0 to differentiate foreground and background.
#     """
#     return x > 0

# class CPDataset(Dataset):
#     def __init__(self, tsv_path: str, is_VQGAN: bool = False):
#         """
#         Dataset for VQGAN or DDPM training with a single key ('data').
        
#         Args:
#             tsv_path (str): Path to the participants.tsv file.
#             is_VQGAN (bool): Whether to use VQGAN-specific transformations.
#         """
#         super().__init__()
#         self.tsv_path = tsv_path
#         self.is_VQGAN = is_VQGAN
#         self.file_paths = self.get_data_files()

#         print(f"Found {len(self.file_paths)} files in {self.tsv_path}")

#     def get_data_files(self):
#         """
#         Load data files from participants.tsv.
#         """
#         participants = pd.read_csv(self.tsv_path, sep="\t")
#         required_columns = ["sub_id_bids", "participant_id", "scan_id"]
#         for col in required_columns:
#             if col not in participants.columns:
#                 raise ValueError(f"Missing required column: {col}")
        
#         unique_subjects = participants["sub_id_bids"].unique()
#         train_subjects, test_subjects = train_test_split(unique_subjects, test_size=0.2, random_state=42)

#         data_files = []
#         base_path = os.path.dirname(self.tsv_path)
#         for _, row in participants.iterrows():
#             if row["sub_id_bids"] in train_subjects:
#                 file_path = os.path.join(
#                     base_path,
#                     f"work_dir2/cbf2mni_wdir/{row['participant_id']}/{row['scan_id']}/wf/brainextraction/*_dtype.nii.gz"
#                 )
#                 matched_files = glob.glob(file_path)
#                 if len(matched_files) == 1:
#                     data_files.append(matched_files[0])
#                 elif len(matched_files) > 1:
#                     raise ValueError(f"Multiple files match pattern: {file_path}")
#                 else:
#                     raise FileNotFoundError(f"No files found for: {file_path}")

#         return data_files

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx: int):
#         """
#         Get preprocessed sample tensors.
#         """
#         image_path = self.file_paths[idx]
#         a_min, a_max = self.get_image_min_max(image_path)

#         # Define transforms
#         # transforms = Compose([
#         #     LoadImaged(keys=["data"]),
#         #     EnsureChannelFirstd(keys=["data"]),
#         #     Spacingd(keys=["data"], pixdim=[1.0, 1.0, 1.0], mode="trilinear"),
#         #     Orientationd(keys=["data"], axcodes="RAS"),
#         #     ScaleIntensityRanged(keys=["data"], a_min=a_min, a_max=a_max, b_min=-1, b_max=1),
#         #     ResizeWithPadOrCropd(keys=["data"], spatial_size=[32, 256, 256], mode="replicate"),
#         #     RandSpatialCropd(keys=["data"], roi_size=[16, 256, 256], random_size=False),
#         #     RandShiftIntensityd(keys=["data"], offsets=0.1, prob=0.5),
#         #     RandRotated(keys=["data"], range_x=0.3, range_y=0.3, range_z=0.3, prob=0.5),
#         #     ToTensord(keys=["data"]),
#         # ])
#         transforms = Compose([
#             LoadImaged(keys=["data"]),
#             EnsureChannelFirstd(keys=["data"]),
#             Spacingd(keys=["data"], pixdim=[1.0, 1.0, 1.0], mode="trilinear"),  # Use bspline for smoother interpolation
#             Orientationd(keys=["data"], axcodes="RAS"),
#             # CropForegroundd(keys=["data"], source_key="data", select_fn=threshold_at_zero, margin=0, allow_smaller=True),  # Define a threshold for cropping
#             ScaleIntensityRanged(keys=["data"], a_min=a_min, a_max=a_max, b_min=-1, b_max=1),
#             ResizeWithPadOrCropd(keys=["data"], spatial_size=[16, 128, 128], mode="replicate"),  # Use constant padding
#             # RandSpatialCropd(keys=["data"], roi_size=[16, 128, 128], random_size=False),
#             RandShiftIntensityd(keys=["data"], offsets=0.05, prob=0.5),  # Reduce intensity shift
#             RandRotated(keys=["data"], range_x=0.1, range_y=0.1, range_z=0.1, prob=0.5, mode="trilinear"),  # Reduce rotation range
#             ToTensord(keys=["data"]),
#         ])

#         return transforms({"data": image_path})
    
#     @staticmethod
#     def get_image_min_max(image_path):
#         """
#         Compute min and max intensity values for scaling.
        
#         Args:
#             image_path (str): Path to the image file.
        
#         Returns:
#             Tuple[float, float]: Minimum and maximum intensity values.
#         """
#         img = LoadImaged(keys=["data"])({"data": image_path})["data"]
#         return img.min().item(), img.max().item()
