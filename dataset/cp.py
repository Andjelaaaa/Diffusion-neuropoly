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
    ScaleIntensityRangePercentilesd,
    Resized,
)
import torch
from torch.utils.data.dataset import Dataset

def create_bcp_records(tsv_path: str):
    """
    Parse the participants.tsv to collect NIfTI file paths plus subject/session info.
    Returns a list of dicts, each with:
      {
        "image": "/path/to/some_stripped_n4.nii.gz",
        "subject_id": "sub-XXX",
        "session_id": "ses-YYY",
      }
    """
    participants = pd.read_csv(tsv_path, sep="\t")

    data_dicts = []
    base_dir = os.path.dirname(tsv_path)
    for _, row in participants.iterrows():
        subject_id = row["participant_id"]
        # Handle multiple sessions if the cell is comma-separated
        sessions = row["sessions"].split(",")

        for session in sessions:
            session_id = session.strip()
            anat_dir = os.path.join(
                base_dir,
                f"n4_bias_correction/{subject_id}/{session_id}/anat/"
            )
            if not os.path.exists(anat_dir):
                print(f"Missing directory: {anat_dir}")
                continue

            # Collect all stripped T1 files
            stripped_files = glob.glob(os.path.join(anat_dir, "*_T1w_stripped_n4.nii.gz"))
            if not stripped_files:
                print(f"No stripped files found in {anat_dir}")
                continue

            # Choose the latest run (if run is used in naming)
            latest_file = sorted(
                stripped_files,
                key=lambda x: int(x.split("_run-")[1].split("_")[0])
                if "_run-" in x else 0
            )[-1]

            data_dicts.append({
                "data": latest_file,
                "subject_id": subject_id,
                "session_id": session_id
            })

    print(f"Found {len(data_dicts)} valid entries in {tsv_path}")
    return data_dicts

def threshold_at_zero(x):
    """
    Function used by CropForeground(d) to decide which voxels are 'foreground'.
    Here, we define foreground as values strictly greater than zero.
    """
    return x > 0

def get_default_monai_transforms(spatial_size=(64, 128, 128)):
    """
    Returns a Compose of MONAI dictionary-based transforms:
      1. LoadImaged
      2. EnsureChannelFirstd
      3. Orientationd
      4. Spacingd (to 1mm isotropic)
      5. CropForegroundd (foreground = x>0)
      6. ScaleIntensityRangePercentilesd (to [-1, 1])
      7. Resized to (64, 128, 128)
    """
    return Compose([
        LoadImaged(keys="data"),
        EnsureChannelFirstd(keys="data"),
        Orientationd(keys="data", axcodes="RPI"),
        # 1) Resample to 1mm isotropic
        Spacingd(
            keys="data",
            pixdim=(1.0, 1.0, 1.0),
            mode="bilinear"
        ),
        # 2) Crop out only the non-zero region
        CropForegroundd(
            keys="data",
            source_key="data",      # Use the same image to compute bounding box
            select_fn=threshold_at_zero,
            margin=0
        ),
        
        # 4) Resize final shape to (64, 128, 128) with interpolation
        Resized(
            keys="data",
            spatial_size=spatial_size,
            mode=["area"],  # 'area' is often good for down/upsampling grayscale
        ),
        # 3) Scale intensity to [-1, 1], based on image min/max
        ScaleIntensityRangePercentilesd(
            keys="data",
            lower=0.0,
            upper=100.0,
            b_min=-1.0,
            b_max=1.0,
            clip=True
        ),
    ])

class CPDataset(Dataset):
    """
    PyTorch Dataset that uses MONAI transforms for loading and preprocessing
    T1w-stripped images from BIDS derivatives.

    Args:
        tsv_path (str): path to participants.tsv
        base_dir (str): root path containing BIDS derivatives
        transform (Compose): optional MONAI transform; if None, uses default
    """
    def __init__(
        self,
        tsv_path: str,
        transform=None
    ):
        super().__init__()
        # Create list of dicts: [{"image": ..., "subject_id": ..., "session_id": ...}, ...]
        self.data_dicts = create_bcp_records(tsv_path)

        # If no transform is passed, use the default pipeline
        if transform is None:
            transform = get_default_monai_transforms()
        self.transform = transform

    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, idx: int):
        """
        Loads and transforms the data at index `idx`.
        Returns a dictionary with:
         - "data": preprocessed [1, 64, 128, 128] Tensor
         - "subject_id": subject ID string
         - "session_id": session ID string
         - plus any additional transform metadata
        """
        data_item = self.data_dicts[idx]
        # The transform expects a dictionary. We pass it directly.
        # The transform pipeline will modify "data", but "subject_id" / "session_id" will remain unaltered.
        # That means the final output dictionary from self.transform(...) will have both the "data" and the IDs.
        output = self.transform(data_item)

        # Check for NaNs in the output tensor
        assert not torch.isnan(output["data"]).any(), f"NaN values found in 'data' for subject {data_item['subject_id']}"

        return output

# class CPDataset(Dataset):
#     def __init__(self, tsv_path: str, augmentation: bool = False, sp_size: int = 128, d_size: int = 64):
#         """
#         Dataset for training with preprocessing and augmentation.

#         Args:
#             tsv_path (str): Path to the participants.tsv file.
#             augmentation (bool): Whether to apply data augmentation.
#             sp_size (int): Desired spatial size for the output images.
#         """
#         super().__init__()
#         self.tsv_path = tsv_path
#         self.augmentation = augmentation
#         self.sp_size = sp_size
#         self.d_size = d_size
#         self.file_paths, self.scan_ids, self.sub_ids = self.get_data_files()

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
        
#         # unique_subjects = participants["sub_id_bids"].unique()
#         # train_subjects, _ = train_test_split(unique_subjects, test_size=0.2, random_state=42)

#         data_files = []
#         scan_ids = []
#         sub_ids = []
#         base_path = os.path.dirname(self.tsv_path)
#         for _, row in participants.iterrows():
#             # if row["sub_id_bids"] in train_subjects:
#             file_path = os.path.join(
#                 base_path,
#                 f"work_dir2/cbf2mni_wdir/{row['participant_id']}/{row['scan_id']}/wf/brainextraction/*_dtype.nii.gz"
#             )
#             matched_files = glob.glob(file_path)
            
#             if len(matched_files) == 1:
#                 data_files.append(matched_files[0])
#                 scan_ids.append(row['scan_id'])
#                 sub_ids.append(row['sub_id_bids'])
#             elif len(matched_files) > 1:
#                 raise ValueError(f"Multiple files match pattern: {file_path}")
#             else:
#                 raise FileNotFoundError(f"No files found for: {file_path}")

#         return data_files, scan_ids, sub_ids

#     def roi_crop(self, image):
#         """
#         Crop the image to the region of interest (ROI) containing non-zero values.
#         """
#         # image = image[:, :, :, 0]  # Assume single-channel image
#         mask = image > 0  # Binary mask of non-zero pixels
#         coords = np.argwhere(mask)  # Coordinates of non-zero pixels

#         # Compute bounding box
#         x0, y0, z0 = coords.min(axis=0)
#         x1, y1, z1 = coords.max(axis=0) + 1  # Bounding box dimensions

#         # Crop to bounding box and pad to uniform size
#         cropped = image[x0:x1, y0:y1, z0:z1]
#         padded_crop = tio.CropOrPad(np.max(cropped.shape))(cropped.copy()[None])
#         # print(cropped.shape, padded_crop.shape)
#         padded_crop = np.transpose(padded_crop, (1, 2, 3, 0))  # Rearrange axes
#         # print( padded_crop.shape)

#         return padded_crop, (x0, x1, y0, y1, z0, z1)

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, index):
#         """
#         Preprocess and return a sample tensor.
#         """
#         # Load image
#         image_path = self.file_paths[index]
#         scan_id = self.scan_ids[index]
#         sub_id = self.sub_ids[index]
#         img = nib.load(image_path)
#         affine = img.affine
        
#         img = np.swapaxes(img.get_fdata(), 0, 2)
#         # print(img.shape)
#         img = np.flip(img, 1)
#         # print(img.shape)
#         img = np.flip(img, 2)
#         # print(img.shape)

#         # Apply ROI cropping
#         # img = self.roi_crop(image=img)
#         cropped_img, bbox = self.roi_crop(img)

#         # Resize to uniform spatial size
#         # img = resize(img, (self.d_size, self.sp_size, self.sp_size), mode='constant')
#         resized_img = resize(cropped_img, (self.d_size, self.sp_size, self.sp_size), mode='constant')
#         # resized_img = resize(img, (self.d_size, self.sp_size, self.sp_size), mode='constant')
#         # img = img.get_fdata()
#         # resized_img = img

#         # Update affine for resized image
#         x0, x1, y0, y1, z0, z1 = bbox
#         scale_factors = np.array([
#             (x1 - x0) / self.d_size,
#             (y1 - y0) / self.sp_size,
#             (z1 - z0) / self.sp_size
#         ])
#         updated_affine = affine.copy()
#         updated_affine[:3, :3] *= scale_factors[:, None]  # Adjust scaling
#         updated_affine[:3, 3] += [x0, y0, z0]  # Adjust translation
#         # updated_affine = 1

#         # Apply data augmentation
#         if self.augmentation:
#             print('AUGMENTATION')
#             random_n = torch.rand(1)
#             random_i = 0.3 * torch.rand(1)[0] + 0.7
#             if random_n[0] > 0.5:
#                 img = np.flip(img, 0)  # Random flip along the x-axis
#             img = img * random_i.data.cpu().numpy()  # Random intensity scaling
            
#         # Normalize to [-1, 1]
#         a_min, a_max = resized_img.min(), resized_img.max()
#         normalized_img = (resized_img - a_min) / (a_max - a_min) * 2 - 1
#         # print('NORMALIZED:', normalized_img.shape)

#         # Reorder axes from (512, 512, 210) to (210, 512, 512)
#         # reordered_img = np.transpose(normalized_img, (2, 0, 1))  # Move axis 2 to the front
#         # print('REORDERED:', reordered_img.shape)

#         # Convert to PyTorch tensor and add batch dimension
#         # imageout = torch.from_numpy(reordered_img).float().unsqueeze(0) 
#         # print('IMAGEOUT:', imageout.shape)
#         # imageout = torch.from_numpy(normalized_img).float().view(1, 210, 512, 512)
#         imageout = torch.from_numpy(normalized_img).float().view(1, self.d_size, self.sp_size, self.sp_size)


#         return {'data': imageout,
#                 'scan_id': scan_id,
#                 'sub_id': sub_id,
#                 'affine': affine,
#                 'upt_affine': updated_affine}
    
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
