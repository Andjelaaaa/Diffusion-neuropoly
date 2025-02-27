"Adapted from https://github.com/SongweiGe/TATS"

# add the main folder to the path so the modules can be imported without errors
import os
import sys
import nibabel as nib
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from torch.utils.data import DataLoader
from ddpm.diffusion import default
from vq_gan_3d.model import VQGAN
from train.callbacks import ImageLogger, VideoLogger
from train.get_dataset import get_dataset
import hydra
from omegaconf import DictConfig, open_dict

import os
import matplotlib.pyplot as plt
import numpy as np
from dataset import CPDataset
from dataset import BCPDataset  
from monai.transforms import (
    Compose,
    Invertd,
    SaveImaged
)
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

import torch
import torchvision
import numpy as np
from PIL import Image


def get_inversion_and_save_transforms(
    forward_transforms,
    output_dir,
    output_postfix="_orig",
):
    """
    Return a Compose that:
     1. Invertd(...) the "image" key using the same forward_transforms.
     2. SaveImaged(...) to disk with the original orientation, spacing, etc.
    """
    return Compose([
        # 1) invert
        Invertd(
            keys=["image"],                        # which key(s) to invert
            transform=forward_transforms,          # the same pipeline used in the dataset
            orig_keys=["image"],                   # same as "image" in the forward pass
            meta_keys=["image_meta_dict"],         # forward transforms store info here
            nearest_interp=False,                  # or True for label data
            to_tensor=True,
        ),
        # 2) save
        SaveImaged(
            keys=["image"],
            meta_keys=["image_meta_dict"],
            output_dir=output_dir,
            output_postfix=output_postfix,
            resample=False,    # no extra resampling needed; we want original orientation/spacing
            separate_folder=False,
            print_log=True
        ),
    ])
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
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes="RPI"),
        # 1) Resample to 1mm isotropic
        Spacingd(
            keys="image",
            pixdim=(1.0, 1.0, 1.0),
            mode="bilinear"
        ),
        # 2) Crop out only the non-zero region
        CropForegroundd(
            keys="image",
            source_key="image",      # Use the same image to compute bounding box
            select_fn=threshold_at_zero,
            margin=0
        ),
        
        # 4) Resize final shape to (64, 128, 128) with interpolation
        Resized(
            keys="image",
            spatial_size=spatial_size,
            mode=["area"],  # 'area' is often good for down/upsampling grayscale
        ),
        # 3) Scale intensity to [-1, 1], based on image min/max
        ScaleIntensityRangePercentilesd(
            keys="image",
            lower=0.0,
            upper=100.0,
            b_min=-1.0,
            b_max=1.0,
            clip=True
        ),
    ])
def visualize_and_save_bcpdataset(tsv_path, results_folder, num_samples=2):
    """
    Demonstrate loading from BCPDataset, visualizing a slice,
    then inverting the transforms and saving as a NIfTI.
    """
    # 1) Instantiate dataset with known forward transforms
    forward_transforms = get_default_monai_transforms()
    dataset = BCPDataset(tsv_path=tsv_path, base_dir=os.path.dirname(tsv_path),
                         transform=forward_transforms)

    # 2) Prepare a post-transform pipeline for inverting & saving
    inversion_pipeline = get_inversion_and_save_transforms(
        forward_transforms=forward_transforms,
        output_dir=results_folder,
        output_postfix="_orig"
    )

    # Create results folder if it doesn't exist
    os.makedirs(results_folder, exist_ok=True)

    print(f"Dataset size: {len(dataset)} samples")
    num_samples = min(num_samples, len(dataset))

    for i in range(num_samples):
        sample = dataset[i]  # This is already preprocessed
        image_tensor = sample["image"]  # shape: [1, 64, 128, 128] if 3D
        subject_id = sample["subject_id"]
        session_id = sample["session_id"]
        print(sample.keys())
        print(sample['foreground_start_coord'])

        print(f"Sample {i+1}: Subject={subject_id}, Session={session_id}")
        print("Image shape (preprocessed):", image_tensor.shape)

        # 3) Visualize the middle slice of the *preprocessed* data
        img_data = image_tensor.squeeze().cpu().numpy()  # [64, 128, 128]

        processed_nifti = nib.Nifti1Image(img_data, affine=np.eye(4))
        processed_nifti_path = os.path.join(results_folder, f"{subject_id}_{session_id}_processed.nii.gz")
        nib.save(processed_nifti, processed_nifti_path)


        middle_idx = img_data.shape[0] // 2
        middle_slice = img_data[middle_idx]
        
        plt.figure()
        plt.imshow(middle_slice, cmap="gray")
        plt.title(f"Preprocessed Middle Slice - {subject_id}, {session_id}")
        plt.axis("off")

        png_save_path = os.path.join(
            results_folder, f"{subject_id}_{session_id}_slice.png"
        )
        plt.savefig(png_save_path, bbox_inches="tight")
        plt.close()
        print(f"Saved slice visualization to {png_save_path}")

        # 4) Invert + save the full volume as a NIfTI
        #    We pass the entire dictionary `sample` to the inversion pipeline.
        #    That pipeline will produce + save a .nii.gz in the specified folder.
        inverted_output = inversion_pipeline(sample)
        # The inverted result is also returned; e.g. inverted_output["image"] is the original shape/tensor.

        # The file naming from SaveImaged uses the "filename_or_obj" key in meta dict
        # by default, or you can specify `output_filename="..."` in SaveImaged(...).
        # So you'll see something like: subjectID_sessionID_T1w_stripped_orig.nii.gz
        print("Done inverting and saving the original-space image.\n")

def visualize_and_save_bcpdataset_prev(tsv_path, results_folder, num_samples=2):
    """
    Visualize and save the middle slice of preprocessed images from BCPDataset.

    Args:
        tsv_path (str): Path to the participants.tsv file.
        results_folder (str): Path to the folder where plots will be saved.
        num_samples (int): Number of samples to visualize.
    """
    # Initialize the dataset
    dataset = BCPDataset(tsv_path=tsv_path, base_dir=os.path.dirname(tsv_path), sp_size=128, d_size=64, augmentation=False)

    # Create results folder if it doesn't exist
    os.makedirs(results_folder, exist_ok=True)

    print(f"Dataset size: {len(dataset)} samples")
    num_samples = min(num_samples, len(dataset))

    for i in range(num_samples):
        sample = dataset[i]  # Get a single data sample
        affine = sample["affine"]

        print(f"Processing Sample {i + 1}: {sample['sub_id']}, {sample['scan_id']}")
        print("SHAPE BEFORE:", sample["data"].shape)

        img_data = sample["data"].squeeze().numpy()  # Remove unnecessary dimensions
        print("SHAPE:", img_data.shape)
        print("MAX:", np.max(img_data), "MIN:", np.min(img_data))

        # Ensure it's a 3D volume and select the middle slice
        if img_data.ndim == 3:
            middle_idx = img_data.shape[0] // 2  # Calculate middle slice index
            middle_slice = img_data[middle_idx, :, :]
        else:
            raise ValueError(f"Expected 3D data, but got {img_data.ndim}D data for sample {i + 1}")

        # Save the image as a NIfTI file with correct affine transformation
        # input_nifti = nib.Nifti1Image(np.transpose(img_data, (2, 1, 0)), affine=affine)
        # print("NEW SHAPE:", np.transpose(img_data, (2, 1, 0)).shape)

        input_nifti = nib.Nifti1Image(img_data, affine=affine)

        # File naming
        sub_id = sample["sub_id"]
        scan_id = sample["scan_id"]
        nifti_save_path = os.path.join(results_folder, f"{sub_id}_{scan_id}_affine.nii.gz")

        nib.save(input_nifti, nifti_save_path)
        print(f"Saved: {nifti_save_path}")

        # Plot and save the middle slice as PNG
        plt.figure()
        plt.imshow(middle_slice, cmap="gray")
        plt.axis("off")
        plt.title(f"Sample {i + 1}, Middle Slice {middle_idx + 1}")

        # Save the plot
        png_save_path = os.path.join(results_folder, f"{sub_id}_{scan_id}_middle_slice.png")
        plt.savefig(png_save_path, bbox_inches="tight")
        plt.close()

        print(f"Saved middle slice visualization: {png_save_path}")

    print(f"Saved {num_samples} images and visualizations to {results_folder}")

def visualize_and_save_cpdataset(tsv_path, results_folder, is_VQGAN=False, num_samples=2):
    """
    Visualize and save the middle slice of preprocessed images from CPDataset.
    
    Args:
        tsv_path (str): Path to the participants.tsv file.
        results_folder (str): Path to the folder where plots will be saved.
        is_VQGAN (bool): Whether to use VQGAN-specific transformations.
        num_samples (int): Number of samples to visualize.
    """
    # Initialize the dataset
    # dataset = CPDataset(tsv_path=tsv_path, sp_size=256, augmentation=True)
    dataset = CPDataset(tsv_path=tsv_path, sp_size=128, d_size=64, augmentation=False)

    # Create results folder if it doesn't exist
    os.makedirs(results_folder, exist_ok=True)

    print(f"Dataset size: {len(dataset)} samples")
    num_samples = min(num_samples, len(dataset))

    for i in range(num_samples):
        sample = dataset[i]  # Get a single data sample
        upt_affine = sample['upt_affine']
        affine = sample['affine']
        print('SHAPE BEFORE', sample['data'].shape)
        img_data = sample['data'].squeeze().numpy()  # Remove unnecessary dimensions
        print('SHAPE', img_data.shape)
        print('MAX', np.max(img_data), 'MIN', np.min(img_data))
        # Ensure it's a 3D volume and select the middle slice
        if img_data.ndim == 3:
            middle_idx = img_data.shape[0] // 2  # Calculate middle slice index
            middle_slice = img_data[middle_idx, :, :]
        else:
            raise ValueError(f"Expected 3D data, but got {img_data.ndim}D data for sample {i + 1}")

        # Save the input and reconstruction as .nii files
        input_nifti = nib.Nifti1Image(np.transpose(img_data, (2, 1, 0)), affine=sample['affine']) 
        print('NEW SHAPE', np.transpose(img_data, (2, 1, 0)).shape)
        # input_diff_affine = nib.Nifti1Image(img_data, affine=sample['upt_affine'])
        sub_id = sample['sub_id']
        scan_id = sample['scan_id']
        
        nib.save(input_nifti, os.path.join(results_folder, f"{sub_id}_{scan_id}_affine.nii.gz"))
        # nib.save(input_diff_affine, os.path.join(results_folder, f"{sub_id}_{scan_id}_upt_affine.nii.gz"))

        # Plot and save the middle slice
        # plt.figure()
        # plt.imshow(middle_slice, cmap='gray')
        # plt.axis('off')
        # plt.title(f"Sample {i + 1}, Middle Slice {middle_idx + 1}")
        
        # # Save the plot
        # save_path = os.path.join(results_folder, f"sample_{i + 1}_middle_slice_64.png")
        # plt.savefig(save_path, bbox_inches='tight')
        # plt.close()
    
    print(f"Saved volumes to {results_folder}")


def create_3d_image_grid(
    images: torch.Tensor, 
    slices=None, 
    **kwargs
) -> Image.Image:
    """
    Creates a grid of selected slices from a 5D tensor of shape (B, C, D, H, W).
    Returns a PIL image of the final grid.
    
    Args:
        images (torch.Tensor): A 5D tensor of shape (B, C, D, H, W).
        slices (list[int] or None): Which slice indices to extract. 
                                    If None, a single middle slice is used.
        **kwargs: Additional kwargs for torchvision.utils.make_grid, e.g.:
                  - 'nrow' for how many images per row
                  - 'normalize', 'range', etc.
    
    Returns:
        PIL.Image: A PIL image containing the grid.
    """
    # Make sure we're on CPU for easier PIL/numpy ops
    images = images.cpu()
    B, C, D, H, W = images.shape
    
    # Default slice is the middle if none specified
    if slices is None:
        slices = [D // 2]  # single slice in the middle

    # Collect slices in a list of Tensors
    all_slice_imgs = []
    for slice_idx in slices:
        slice_idx = min(slice_idx, D - 1)  # clamp in case slice_idx >= D
        slice_img = images[:, :, slice_idx, :, :]  # shape: (B, C, H, W)
        all_slice_imgs.append(slice_img)
    
    # Concatenate along batch dimension => shape: (B * num_slices, C, H, W)
    slice_images = torch.cat(all_slice_imgs, dim=0)

    # Create a grid (Tensor) => shape: (3, grid_height, grid_width) or (1, ...)
    grid = torchvision.utils.make_grid(slice_images, **kwargs)

    # Convert the grid to a PIL image
    # grid shape is (C, H, W); move channel last => (H, W, C)
    ndarr = grid.permute(1, 2, 0).numpy()

    # If you used `normalize=True`, your range is [0, 1], so multiply by 255
    # Or if not normalized, you might need to clamp instead
    ndarr = (ndarr * 255).astype(np.uint8)
    
    return Image.fromarray(ndarr)

class LogInputImagesCallback(Callback):
    """
    Logs input images to W&B during training at a given frequency.
    """

    def __init__(self, log_every_n_steps=500, max_images=4):
        """
        Args:
            log_every_n_steps (int): Frequency (in steps) to log images.
            max_images (int): How many images from the batch to log.
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.max_images = max_images

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """
        This hook is called at the end of every training batch.
        We check if it's time to log, and if so, send a few images to W&B.
        """
        # Only log every `log_every_n_steps` steps
        global_step = trainer.global_step
        if global_step % self.log_every_n_steps == 0 and global_step > 0:
            # `batch` is typically a dict if you used a custom dataset; adjust as needed
            # Assuming your dataset __getitem__ returns {"data": tensor, "some_label": ...}
            images = batch["data"]  # shape: [B, C, H, W, ...] depending on your dataset
            subject_id = batch["subject_id"]
            images = images[: self.max_images]  # take a few images

            B, C, D, H, W = images.shape
            quarter_slice = max(D // 4, 0)
            middle_slice = max(D // 2, 0)
            three_quarter_slice = max((3 * D) // 4, 0)

            # Create a grid from these 3 slices
            grid_img = create_3d_image_grid(
                images, 
                slices=[quarter_slice, middle_slice, three_quarter_slice],
                nrow=self.max_images, 
                normalize=True,      # Example: turn on normalization
                range=(0, 1),        # Example: scale intensities from [0,1]
                pad_value=255        # Example: white padding
            )

            # Convert tensor to PIL image if needed
            if isinstance(grid_img, torch.Tensor):
                grid_img = torchvision.transforms.ToPILImage()(grid_img)

            # Now log the PIL image to W&B
            trainer.logger.experiment.log({
                f"3D_volume_slices_grid_{subject_id}": wandb.Image(grid_img, caption=f"Subject {subject_id}"),
                "global_step": global_step
            })

class EarlyStoppingAfter400Epochs(Callback):
    def __init__(self, patience=20, min_epochs=400):
        """
        Args:
            patience (int): Number of epochs to wait before stopping training
                            if validation loss keeps declining and training loss increases.
            min_epochs (int): Minimum number of epochs to complete before considering early stopping.
        """
        self.patience = patience
        self.min_epochs = min_epochs
        self.val_losses = []
        self.train_losses = []
    
    def on_validation_end(self, trainer, pl_module):
        """
        Called at the end of validation. Check if the condition for stopping is met.
        """
        # Get current validation and training loss
        current_epoch = trainer.current_epoch
        current_val_loss = trainer.callback_metrics.get("val/recon_loss")
        current_train_loss = trainer.callback_metrics.get("train/recon_loss")
        
        # Append the losses to track trends
        if current_val_loss is not None:
            self.val_losses.append(current_val_loss.item())
        if current_train_loss is not None:
            self.train_losses.append(current_train_loss.item())

        # Ensure the minimum epoch count is reached before checking for early stopping
        if current_epoch >= self.min_epochs and len(self.val_losses) > self.patience:
            recent_val_losses = self.val_losses[-self.patience:]
            recent_train_losses = self.train_losses[-self.patience:]
            
            # Check if validation loss is declining and training loss is increasing
            if all(x < y for x, y in zip(recent_val_losses[1:], recent_val_losses[:-1])) and \
               all(x > y for x, y in zip(recent_train_losses[1:], recent_train_losses[:-1])):
                trainer.should_stop = True
                print(f"Early stopping triggered: Validation loss decreased for {self.patience} epochs after {self.min_epochs} epochs.")
    
    def on_train_end(self, trainer, pl_module):
        """Reset the losses for the next fold."""
        self.val_losses = []
        self.train_losses = []

@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    pl.seed_everything(cfg.model.seed)

    num_folds = 5  # Number of folds for cross-validation

    # Ensure saving checkpoints in /scratch
    scratch_dir = os.environ.get("SCRATCH", "/scratch")  # Use $SCRATCH environment variable
    original_root_dir = os.path.join(scratch_dir, cfg.dataset.name, 'exp_combined_5fold')

    for fold_idx in range(num_folds):
        print(f"Training Fold {fold_idx + 1}/{num_folds}")
        
        # Get datasets for the current fold
        train_dataset, val_dataset, sampler = get_dataset(cfg, fold_idx=fold_idx, num_folds=num_folds, d_size=64)
        
        # Initialize DataLoaders
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.model.batch_size,
                                      num_workers=cfg.model.num_workers, sampler=sampler)
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.model.batch_size,
                                    shuffle=False, num_workers=cfg.model.num_workers)

        # Adjust learning rate dynamically
        bs, base_lr, ngpu, accumulate = cfg.model.batch_size, cfg.model.lr, cfg.model.gpus, cfg.model.accumulate_grad_batches
        with open_dict(cfg):
            cfg.model.lr = accumulate * (ngpu / 8.) * (bs / 4.) * base_lr

        # Use a separate variable for the fold-specific directory
        fold_specific_dir = os.path.join(original_root_dir, f"fold_{fold_idx}")
        os.makedirs(fold_specific_dir, exist_ok=True)  # Ensure the directory exists
        print(f"Fold {fold_idx + 1}: Learning Rate = {cfg.model.lr:.2e}")
        print(f"Saving results to: {fold_specific_dir}")

        # Initialize the model
        model = VQGAN(cfg)

        # Define callbacks
        callbacks = [
            ModelCheckpoint(
            monitor='val/recon_loss',
            save_top_k=3,
            mode='min',
            filename=f'latest_checkpoint_fold_{fold_idx}-val_loss={{val/recon_loss:.4f}}'),
            EarlyStoppingAfter400Epochs(patience=20, min_epochs=400),
            # ModelCheckpoint(monitor='val/recon_loss', save_top_k=3, mode='min', filename=f'latest_checkpoint_fold_{fold_idx}'),
            ModelCheckpoint(every_n_train_steps=3000, save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'),
            ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1, filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'), 
            LogInputImagesCallback(log_every_n_steps=3000, max_images=1)     
        ]

        wandb_dir = os.path.join(fold_specific_dir, "wandb_logs")
        os.makedirs(wandb_dir, exist_ok=True)

        wandb_logger = WandbLogger(project="VQ-GAN", save_dir=wandb_dir, mode="offline")
        # Trainer setup
        trainer = pl.Trainer(
            logger=wandb_logger,
            gpus=cfg.model.gpus,
            accumulate_grad_batches=cfg.model.accumulate_grad_batches,
            # default_root_dir=cfg.model.default_root_dir,
            default_root_dir=fold_specific_dir,
            callbacks=callbacks,
            max_steps=cfg.model.max_steps,
            max_epochs=cfg.model.max_epochs,
            precision=cfg.model.precision,
            gradient_clip_val=cfg.model.gradient_clip_val,
        )
        
        # Start training for the current fold
        trainer.fit(model, train_dataloader, val_dataloader)

        print(f"Fold {fold_idx + 1} completed!")

if __name__ == '__main__':
    # tsv_path = "/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/all-participants.tsv"  # Replace with the actual path
    # results_folder = "./results/"  # Replace with the desired folder
    # visualize_and_save_cpdataset(tsv_path=tsv_path, results_folder=results_folder, is_VQGAN=False, num_samples=1)
    
    # tsv_path = "/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/hc-bcp/participants.tsv"  
    # results_folder = "/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/results/"  # Replace with the desired folder
    # visualize_and_save_bcpdataset(tsv_path=tsv_path, results_folder=results_folder)
    
    # To run all folds
    run()
