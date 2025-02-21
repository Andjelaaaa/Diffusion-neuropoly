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
# def visualize_and_save_bcpdataset(tsv_path, results_folder, num_samples=2):
#     """
#     Visualize and save the middle slice of preprocessed images from BCPDataset.

#     Args:
#         tsv_path (str): Path to the participants.tsv file.
#         results_folder (str): Path to the folder where plots will be saved.
#         num_samples (int): Number of samples to visualize.
#     """
#     # Initialize the dataset
#     dataset = BCPDataset(tsv_path=tsv_path, base_dir=os.path.dirname(tsv_path))

#     # Create results folder if it doesn't exist
#     os.makedirs(results_folder, exist_ok=True)

#     print(f"Dataset size: {len(dataset)} samples")
#     num_samples = min(num_samples, len(dataset))

#     for i in range(num_samples):
#         sample = dataset[i]  # Get a single data sample
#         image_tensor = sample["image"]

#         subject_id = sample["subject_id"]
#         session_id = sample["session_id"]

#         print("Image shape:", image_tensor.shape)  # [1, 64, 128, 128]
#         print("Subject:", subject_id, "| Session:", session_id)

#         img_data = sample["image"].squeeze().numpy()  # Remove unnecessary dimensions
#         print("SHAPE:", img_data.shape)
#         print("MAX:", np.max(img_data), "MIN:", np.min(img_data))

#         # Ensure it's a 3D volume and select the middle slice
#         if img_data.ndim == 3:
#             middle_idx = img_data.shape[0] // 2  # Calculate middle slice index
#             middle_slice = img_data[middle_idx, :, :]
#         else:
#             raise ValueError(f"Expected 3D data, but got {img_data.ndim}D data for sample {i + 1}")

#         # Save the image as a NIfTI file with correct affine transformation
#         # input_nifti = nib.Nifti1Image(np.transpose(img_data, (2, 1, 0)), affine=affine)
#         # print("NEW SHAPE:", np.transpose(img_data, (2, 1, 0)).shape)

#         # input_nifti = nib.Nifti1Image(img_data, affine=affine)

#         # # File naming
#         # sub_id = sample["sub_id"]
#         # scan_id = sample["scan_id"]
#         # nifti_save_path = os.path.join(results_folder, f"{sub_id}_{scan_id}_affine.nii.gz")

#         # nib.save(input_nifti, nifti_save_path)
#         # print(f"Saved: {nifti_save_path}")

#         # Plot and save the middle slice as PNG
#         plt.figure()
#         plt.imshow(middle_slice, cmap="gray")
#         plt.axis("off")
#         plt.title(f"Sample {i + 1}, Middle Slice {middle_idx + 1}")

#         # Save the plot
#         png_save_path = os.path.join(results_folder, f"{subject_id}_{session_id}_middle_slice.png")
#         plt.savefig(png_save_path, bbox_inches="tight")
#         plt.close()

#         print(f"Saved middle slice visualization: {png_save_path}")

#     print(f"Saved {num_samples} images and visualizations to {results_folder}")

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

# @hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
# def run(cfg: DictConfig):
#     # Set the GPU to use
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.model.gpu_index)
    
#     pl.seed_everything(cfg.model.seed)

#     train_dataset, val_dataset, sampler = get_dataset(cfg)
#     train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.model.batch_size,
#                                   num_workers=cfg.model.num_workers, sampler=sampler)
#     val_dataloader = DataLoader(val_dataset, batch_size=cfg.model.batch_size,
#                                 shuffle=False, num_workers=cfg.model.num_workers)

#     # Automatically adjust learning rate
#     bs, base_lr, ngpu, accumulate = cfg.model.batch_size, cfg.model.lr, cfg.model.gpus, cfg.model.accumulate_grad_batches

#     with open_dict(cfg):
#         cfg.model.lr = accumulate * (ngpu / 8.) * (bs / 4.) * base_lr
#         cfg.model.default_root_dir = os.path.join(
#             cfg.model.default_root_dir, cfg.dataset.name, cfg.model.default_root_dir_postfix)
#     print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus/8) * {} (batchsize/4) * {:.2e} (base_lr)".format(
#         cfg.model.lr, accumulate, ngpu / 8, bs / 4, base_lr))

#     model = VQGAN(cfg)

#     callbacks = [
#         ModelCheckpoint(monitor='val/recon_loss', save_top_k=3, mode='min', filename='latest_checkpoint'),
#         ModelCheckpoint(every_n_train_steps=3000, save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'),
#         ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1, filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'),
#         ImageLogger(batch_frequency=750, max_images=4, clamp=True),
#         VideoLogger(batch_frequency=1500, max_videos=4, clamp=True)
#     ]

#     # Load the most recent checkpoint file
#     base_dir = os.path.join(cfg.model.default_root_dir, 'lightning_logs')
#     if os.path.exists(base_dir):
#         log_folder = ckpt_file = ''
#         version_id_used = step_used = 0
#         for folder in os.listdir(base_dir):
#             version_id = int(folder.split('_')[1])
#             if version_id > version_id_used:
#                 version_id_used = version_id
#                 log_folder = folder
#         if len(log_folder) > 0:
#             ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
#             for fn in os.listdir(ckpt_folder):
#                 if fn == 'latest_checkpoint.ckpt':
#                     ckpt_file = 'latest_checkpoint_prev.ckpt'
#                     os.rename(os.path.join(ckpt_folder, fn),
#                               os.path.join(ckpt_folder, ckpt_file))
#             if len(ckpt_file) > 0:
#                 cfg.model.resume_from_checkpoint = os.path.join(
#                     ckpt_folder, ckpt_file)
#                 print('will start from the recent ckpt %s' %
#                       cfg.model.resume_from_checkpoint)
        
#     accelerator = None
#     if cfg.model.gpus > 1:
#         accelerator = 'ddp'

#     trainer = pl.Trainer(
#         gpus=cfg.model.gpus,
#         accumulate_grad_batches=cfg.model.accumulate_grad_batches,
#         default_root_dir=cfg.model.default_root_dir,
#         resume_from_checkpoint=cfg.model.resume_from_checkpoint,
#         callbacks=callbacks,
#         max_steps=cfg.model.max_steps,
#         max_epochs=cfg.model.max_epochs,
#         precision=cfg.model.precision,
#         gradient_clip_val=cfg.model.gradient_clip_val,
#         accelerator=accelerator,
#     )
    
#     trainer.fit(model, train_dataloader, val_dataloader)

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
    # Set the GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.model.gpu_index)
    
    pl.seed_everything(cfg.model.seed)

    num_folds = 3  # Number of folds for cross-validation

    # original_root_dir = os.path.join(cfg.model.default_root_dir, cfg.dataset.name)  # Preserve the original root directory
    original_root_dir = os.path.join(cfg.model.default_root_dir, cfg.dataset.name, 'exp_combined_data_3fold')

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
            # ImageLogger(batch_frequency=3000, max_images=4, clamp=True),
            # VideoLogger(batch_frequency=1500, max_videos=4, clamp=True)
        ]

        wandb_logger = WandbLogger(project="VQ-GAN")
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

@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def rerun_entry(cfg: DictConfig):
    """
    Entry point for rerunning a specific fold.
    """
    # Define fold index, checkpoint path, and folder name
    # fold_idx = 2  # Replace with the desired fold index
    # checkpoint_path = "/home/GRAMES.POLYMTL.CA/andim/Diffusion-neuropoly/checkpoints/vq_gan/CP/exp_3_folds/fold_2/lightning_logs/version_0/checkpoints/epoch=104-step=39000-train/recon_loss=3.54.ckpt"  # Replace with the actual checkpoint path
    # folder_name = "exp_3_folds"  # Replace if different
    # fold_idx = 0  # Replace with the desired fold index
    # checkpoint_path = "/home/GRAMES.POLYMTL.CA/andim/Diffusion-neuropoly/checkpoints/vq_gan/COMBINED/exp_combined_data_3fold/fold_0/lightning_logs/version_2/checkpoints/epoch=208-step=327000-train/recon_loss=0.12.ckpt"  # Replace with the actual checkpoint path
    # folder_name = "exp_combined_data_3fold"  # Replace if different

    fold_idx = 0  # Replace with the desired fold index
    checkpoint_path = "/home/GRAMES.POLYMTL.CA/andim/Diffusion-neuropoly/checkpoints/vq_gan/COMBINED/exp_combined_data_3fold/fold_0/lightning_logs/version_3/checkpoints/epoch=240-step=378000-train/recon_loss=0.10.ckpt"  # Replace with the actual checkpoint path
    folder_name = "exp_combined_data_3fold"  # Replace if different

    rerun_specific_fold(cfg, fold_idx, checkpoint_path, folder_name)


def rerun_specific_fold(cfg: DictConfig, fold_idx: int, checkpoint_path: str, folder_name: str):
    """
    Retrain a specific fold from a given checkpoint.

    Args:
        cfg (DictConfig): Hydra configuration object.
        fold_idx (int): Index of the fold to retrain.
        checkpoint_path (str): Path to the checkpoint for resuming.
        folder_name (str): Subfolder name for saving results.
    """
    # Set the GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.model.gpu_index)
    
    pl.seed_everything(cfg.model.seed)

    original_root_dir = os.path.join(cfg.model.default_root_dir, cfg.dataset.name, folder_name)

    print(f"Retraining Fold {fold_idx + 1}")

    # Train the specified fold
    train_single_fold(cfg, fold_idx, original_root_dir, checkpoint_path=checkpoint_path)


def train_single_fold(cfg, fold_idx, original_root_dir, checkpoint_path=None):
    """
    Train a single fold.

    Args:
        cfg (DictConfig): Hydra configuration object.
        fold_idx (int): Index of the fold to train.
        original_root_dir (str): Root directory for saving results.
        checkpoint_path (str): Path to the checkpoint for resuming training. Default is None.
    """
    # Get datasets for the specified fold
    train_dataset, val_dataset, sampler = get_dataset(cfg, fold_idx=fold_idx, num_folds=cfg.get('num_folds', 3), d_size=64)
    
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
        ModelCheckpoint(monitor='val/recon_loss', save_top_k=3, mode='min', filename=f'latest_checkpoint_fold_{fold_idx}'),
        ModelCheckpoint(every_n_train_steps=3000, save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'),
        ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1, filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'),      
    ]

    # Trainer setup
    trainer = pl.Trainer(
        gpus=cfg.model.gpus,
        accumulate_grad_batches=cfg.model.accumulate_grad_batches,
        default_root_dir=fold_specific_dir,
        resume_from_checkpoint=checkpoint_path,  # Resume from checkpoint if provided
        callbacks=callbacks,
        max_steps=cfg.model.max_steps,
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        gradient_clip_val=cfg.model.gradient_clip_val,
    )
    
    # Start training
    trainer.fit(model, train_dataloader, val_dataloader)
    print(f"Training for Fold {fold_idx + 1} completed!")

if __name__ == '__main__':
    # tsv_path = "/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/all-participants.tsv"  # Replace with the actual path
    # results_folder = "./results/"  # Replace with the desired folder
    # visualize_and_save_cpdataset(tsv_path=tsv_path, results_folder=results_folder, is_VQGAN=False, num_samples=1)
    
    # tsv_path = "/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/hc-bcp/participants.tsv"  
    # results_folder = "/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/results/"  # Replace with the desired folder
    # visualize_and_save_bcpdataset(tsv_path=tsv_path, results_folder=results_folder)
    
    # To run all folds
    run()
    # To retrain a specific fold (uncomment the following lines for specific use case)
    # rerun_entry()
