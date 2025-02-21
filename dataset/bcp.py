import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset

# MONAI
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRangePercentilesd,
    ResizeWithPadOrCropd,
    CropForegroundd,
    Resized,
)
import nibabel as nib


def create_bcp_records(tsv_path: str):
    """
    Parse the participants.tsv to collect NIfTI file paths plus subject/session info.
    Returns a list of dicts, each with:
      {
        "image": "/path/to/some_stripped.nii.gz",
        "subject_id": "sub-XXX",
        "session_id": "ses-YYY",
      }
    """
    participants = pd.read_csv(tsv_path, sep="\t")

    data_dicts = []
    base_dir = os.path.dirname(tsv_path)
    runs_to_include_pth = os.path.join(base_dir, 'code', 'runs_to_include.csv')
    
    runs_to_include = pd.read_csv(runs_to_include_pth)

    for _, row in participants.iterrows():
        subject_id = row["participant_id"]
        # Handle multiple sessions if the cell is comma-separated
        sessions = row["sessions"].split(",")
        # print(subject_id)
        # print(sessions)
        if subject_id in runs_to_include['participant_id'].to_list():
            for session in sessions:
                session_id = session.strip()
                if session_id in runs_to_include[runs_to_include['participant_id']==subject_id]['session_id'].to_list():
                    anat_dir = os.path.join(
                        base_dir,
                        f"derivatives/n4_bias_correction/{subject_id}/{session_id}/anat/"
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

class BCPDataset(Dataset):
    """
    PyTorch Dataset that uses MONAI transforms for loading and preprocessing
    T1w-stripped images from BIDS derivatives.

    Args:
        tsv_path (str): path to participants.tsv
        transform (Compose): optional MONAI transform; if None, uses default
    """
    def __init__(
        self,
        tsv_path: str,
        transform=None
    ):
        super().__init__()
        # Create list of dicts: [{"data": ..., "subject_id": ..., "session_id": ...}, ...]
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
        assert not torch.isnan(output["data"]).any(), f"NaN values found in 'data' for subject {data_item["subject_id"]}"

        return output


# if __name__ == "__main__":
#     # Example usage
#     tsv_path = "/path/to/participants.tsv"
#     base_dir = "/path/to/base_bids_directory"

#     # Initialize dataset with default transforms
#     dataset = BCPDataset(tsv_path, base_dir)

#     print("Dataset size:", len(dataset))

#     # Retrieve one sample
#     sample = dataset[0]
#     image_tensor = sample["image"]
#     subject_id = sample["subject_id"]
#     session_id = sample["session_id"]

#     print("Image shape:", image_tensor.shape)  # [1, 64, 128, 128]
#     print("Subject:", subject_id, "| Session:", session_id)
#     # e.g., use in a DataLoader for training
#     # from torch.utils.data import DataLoader
#     # loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
