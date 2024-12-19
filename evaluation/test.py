
import os
GPU_ID = 3 
os.environ["CUDA_VISIBLE_DEVICES"]=f"{GPU_ID}"
device = f'cuda:0'
import sys
from re import I
import re
sys.path.append('../')
import importlib
import datetime

# Add the root directory of the project to sys.path
project_root = os.path.abspath("/home/GRAMES.POLYMTL.CA/andim/Diffusion-neuropoly")
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the module dynamically
module_name = "vq_gan_3d.model.vqgan"
vqgan_module = importlib.import_module(module_name)

# Access the VQGAN class
VQGAN = vqgan_module.VQGAN
import torch
from ddpm import Unet3D, GaussianDiffusion, Trainer
from train.get_dataset import get_dataset
from hydra import initialize, compose
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import nibabel as nib
from tqdm import tqdm

def find_best_ckpt(base_folder):
    

        # Regular expression to extract recon_loss
        pattern = re.compile(r'recon_loss=(\d+\.\d+)\.ckpt')

        # List to store parsed data
        checkpoints = []

        # Walk through all subdirectories and files
        for root, dirs, files in os.walk(base_folder):
            for filename in files:
                if filename.endswith(".ckpt"):  # Ensure we're working with .ckpt files
                    match = pattern.search(filename)  # Search for recon_loss in the filename
                    if match:
                        recon_loss = float(match.group(1))
                        full_path = os.path.join(root, filename)
                        print(f"Found file: {filename}, Recon_loss: {recon_loss}")
                        checkpoints.append((full_path, recon_loss))

        # Find the checkpoint(s) with the minimum recon_loss
        if checkpoints:
            min_loss = min(checkpoints, key=lambda x: x[1])[1]  # Find the minimum recon_loss
            best_checkpoints = [ckpt for ckpt in checkpoints if ckpt[1] == min_loss]
            
            print(f"Lowest recon_loss: {min_loss}")
            print("Best checkpoints:")
            for ckpt in best_checkpoints:
                print(f"Path: {ckpt[0]}, Recon_loss: {ckpt[1]}")
                return ckpt[0]
        else:
            print("No valid checkpoints found.")

def save_reconstructions(save_path, vqgan, val_dataset):
    """
    Save the reconstructions from the VQGAN model for the validation set.

    Args:
        save_path (str): Directory where the reconstructions will be saved.
        vqgan (torch.nn.Module): The trained VQGAN model.
        val_dataset (torch.utils.data.Dataset): Validation dataset.
    """
    # Ensure the save path exists
    os.makedirs(save_path, exist_ok=True)

    # Put the model in evaluation mode
    vqgan.eval()

    # Iterate over the validation dataset
    for idx in tqdm(range(len(val_dataset)), desc="Saving Reconstructions"):
        # Get the data and scan_id
        sample = val_dataset[idx]
        input_data = torch.tensor(sample['data'][None]).to(device)
        # input_data = sample['data'].unsqueeze(0).to(vqgan.device)  # Add batch dimension
        sub_id = sample['sub_id']
        scan_id = sample['scan_id']
        
        with torch.no_grad():
            # Encode and decode to get the reconstruction
            latent_z = vqgan.pre_vq_conv(vqgan.encoder(input_data))
            reconstructed = vqgan.decode(latent_z)

        # Convert to numpy
        input_numpy = input_data[0,0,:,:,:].cpu().numpy() #input: 1,1,64,128,128
        # reconstructed_numpy = reconstructed.squeeze(0).cpu().numpy()
        reconstructed_numpy = reconstructed[0,0,:,:,:].cpu().numpy()

        # Save the input and reconstruction as .nii files
        input_nifti = nib.Nifti1Image(input_numpy, affine=np.eye(4))
        reconstructed_nifti = nib.Nifti1Image(reconstructed_numpy, affine=np.eye(4))


        nib.save(reconstructed_nifti, os.path.join(save_path, f"{sub_id}_{scan_id}_reconstructed.nii.gz"))
        nib.save(input_nifti, os.path.join(save_path, f"{sub_id}_{scan_id}_input.nii.gz"))

def get_training_duration(base_folder):
    ckpt_files = []

    # Walk through all subdirectories and files to collect checkpoint files
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".ckpt"):  # Look for checkpoint files
                ckpt_files.append(os.path.join(root, file))
    
    # Ensure there are checkpoints to process
    if not ckpt_files:
        print("No checkpoint files found.")
        return None

    # Sort checkpoint files by modification time
    ckpt_files.sort(key=lambda x: os.path.getmtime(x))

    # Get the earliest and latest modification times
    first_ckpt_time = os.path.getmtime(ckpt_files[0])
    last_ckpt_time = os.path.getmtime(ckpt_files[-1])

    # Calculate duration in seconds
    duration_seconds = last_ckpt_time - first_ckpt_time

    # Convert to a human-readable format
    duration = str(datetime.timedelta(seconds=duration_seconds))
    print(f"Training started at: {datetime.datetime.fromtimestamp(first_ckpt_time)}")
    print(f"Training ended at: {datetime.datetime.fromtimestamp(last_ckpt_time)}")
    print(f"Total training duration: {duration}")

    return duration

# Function to extract recon_loss and epochs
def extract_ckpt_data(base_folder):
    ckpt_data = []

    # Walk through all subdirectories and files
    for root, dirs, file in os.walk(base_folder):
        # Regular expression to extract the number after "epoch="
        match = re.search(r'epoch=(\d+)', root)
        epoch = int(match.group(1)) if match else None
        
        match = re.search(r'recon_loss=(\d+\.\d+)', file[0])
        recon_loss = float(match.group(1)) if match else None
        # Append only if both epoch and recon_loss are valid
        if epoch is not None and recon_loss is not None:
            ckpt_data.append((epoch, recon_loss))
    
    return sorted(ckpt_data, key=lambda x: x[0])

def plot_recon_loss_per_epoch(base_folder, title_details):
    ckpt_data = extract_ckpt_data(base_folder)
    
    # If data is available, plot it
    if ckpt_data:
        epochs, recon_losses = zip(*ckpt_data)

        # Plotting recon_loss vs epoch
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, recon_losses, marker='o', label="Recon Loss")
        plt.title("Reconstruction Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Reconstruction Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'vq_gan_training_{title_details}.png')
    else:
        print("No valid checkpoints found.")

def plot_generated_samples(vqgan, train_dataset, device, epoch, sample_indices):
    """
    Plot and save input and output samples for multiple indices.

    Args:
        train_dataset: The dataset to retrieve samples from.
        device: The device (e.g., GPU or CPU) to use for computations.
        sample_indices: List of sample indices to visualize.
    """
    num_samples = len(sample_indices)
    rows = 2
    cols = (num_samples + 1) // 2  # Ensures grid layout fits all samples

    fig, axes = plt.subplots(rows, cols * 2, figsize=(12, 6))  # Create subplots grid

    for i, sample_idx in enumerate(sample_indices):
        # Get the input and pass it through the model
        input_ = torch.tensor(train_dataset[sample_idx]['data'][None]).to(device)
        with torch.no_grad():
            z = vqgan.pre_vq_conv(vqgan.encoder(input_))
            output = vqgan.decode(z)

        # Calculate subplot indices
        row, col = divmod(i, cols)

        # Plot the input middle slice
        axes[row, col * 2].imshow(input_[0, 0, input_.shape[2] // 2, :, :].cpu().numpy(), cmap='gray')
        axes[row, col * 2].set_title(f"Sample {sample_idx} - Input")
        axes[row, col * 2].axis('off')

        # Plot the output middle slice
        axes[row, col * 2 + 1].imshow(output[0, 0, output.shape[2] // 2, :, :].cpu().numpy(), cmap='gray')
        axes[row, col * 2 + 1].set_title(f"Sample {sample_idx} - Output")
        axes[row, col * 2 + 1].axis('off')

    # Remove empty subplots if the number of samples is odd
    for ax in axes.flat[num_samples * 2:]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'generated_vq_gan_outputs_epoch_{epoch}.png')
    plt.close(fig)



if __name__ == '__main__':
    tsv_path = '/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/all-participants.tsv'
    dataset_name = "cp"
    
    USE_DATASET = 'CP'
    

    # if USE_DATASET == 'CP':
    #     # VQGAN_CHECKPOINT = "../checkpoints/vq_gan/CP/lightning_logs/version_1/checkpoints/epoch=474-step=204000-train/recon_loss=0.08.ckpt"
    #     # VQGAN_CHECKPOINT = "../checkpoints/vq_gan/CP/lightning_logs/version_1/checkpoints/epoch=481-step=207000-train/recon_loss=0.12.ckpt"
    #     # VQGAN_CHECKPOINT = "../checkpoints/vq_gan/CP/lightning_logs/version_1/checkpoints/epoch=302-step=130000-10000-train/recon_loss=0.07.ckpt"
    #     VQGAN_CHECKPOINT = "../checkpoints/vq_gan/CP/lightning_logs/version_1/checkpoints/epoch=325-step=140000-10000-train/recon_loss=0.04.ckpt"

    #     # Specify the version_base explicitly
    #     with initialize(config_path="../config/", version_base="1.1"):
    #             cfg = compose(config_name="base_cfg.yaml", overrides=[
    #                 "model=ddpm",
    #                 f"dataset.tsv_path={tsv_path}",
    #                 #f"dataset={dataset_name}",
    #                 f"model.vqgan_ckpt='{VQGAN_CHECKPOINT}'",
    #                 "model.diffusion_img_size=32",
    #                 "model.diffusion_depth_size=32",
    #                 "model.diffusion_num_channels=8",
    #                 "model.dim_mults=[1,2,4,8]",
    #                 "model.batch_size=2",
    #                 "++model.gpus=1",
    #             ])
    #             print("yahoo")
    # else:
    #     print('Not implemented dataset') 

    # # train_dataset, _, _ = get_dataset(cfg)

    # vqgan = VQGAN.load_from_checkpoint(VQGAN_CHECKPOINT)
    # vqgan.decoding_diviser = 3  # An odd integer for VRAM optimization
    # vqgan = vqgan.to(device)
    # vqgan.eval()

    # # Split the path and extract the epoch part
    # for part in VQGAN_CHECKPOINT.split('/'):
    #     if 'epoch=' in part:
    #         epoch = int(part.split('=')[1].split('-')[0])  # Extract and convert to integer
    
    # # Plot generated samples for a specific epoch
    # plot_generated_samples(train_dataset, device, epoch, [0,1,2,3])
    # plot_generated_samples(vqgan, train_dataset, device, epoch, [0,1,2,3])

    # Base folder to start the search
    # base_folder = "../checkpoints/vq_gan/CP/lightning_logs"
    # find_best_ckpt(base_folder)

    # Plot recon loss
    # base_folder = "../checkpoints/vq_gan/CP/lightning_logs/version_1/checkpoints"
    # plot_recon_loss_per_epoch(base_folder)
    # get_training_duration(base_folder)

    #################################################################################
    ########################VQ-GAN Testing on reconstructions per subject############

    # import torch
    # print(torch.cuda.device_count())  # Number of available GPUs
    # for i in range(torch.cuda.device_count()):
    #     print(f"Device {i}: {torch.cuda.get_device_name(i)}")

    num_folds = 5
    paths_for_fold = []
    for fold_idx in range(num_folds):
        # base_folder = f"../checkpoints/vq_gan/CP/fold_{fold_idx}/lightning_logs"
        # path = find_best_ckpt(base_folder)
        # paths_for_fold.append(path)

        # Plot recon loss
        base_folder = f"../checkpoints/vq_gan/CP/fold_{fold_idx}/lightning_logs/version_0/checkpoints"
        plot_recon_loss_per_epoch(base_folder, f'fold_{fold_idx}')
        get_training_duration(base_folder)



        # if USE_DATASET == 'CP':
        #     VQGAN_CHECKPOINT = path

        #     # Specify the version_base explicitly
        #     with initialize(config_path="../config/", version_base="1.1"):
        #             cfg = compose(config_name="base_cfg.yaml", overrides=[
        #                 "model=ddpm",
        #                 f"dataset.tsv_path={tsv_path}",
        #                 #f"dataset={dataset_name}",
        #                 f"model.vqgan_ckpt='{VQGAN_CHECKPOINT}'",
        #                 "model.diffusion_img_size=32",
        #                 "model.diffusion_depth_size=32",
        #                 "model.diffusion_num_channels=8",
        #                 "model.dim_mults=[1,2,4,8]",
        #                 "model.batch_size=2",
        #                 "++model.gpus=1",
        #             ])
        #             print("yahoo")
        # else:
        #     print('Not implemented dataset') 

    

        # vqgan = VQGAN.load_from_checkpoint(VQGAN_CHECKPOINT)
        # vqgan.decoding_diviser = 3  # An odd integer for VRAM optimization
        # vqgan = vqgan.to(device)
        # print(f"Processing Fold {fold_idx + 1}/{num_folds}")
        
        # # Get datasets for the current fold
        # _, val_dataset, _ = get_dataset(cfg, fold_idx=fold_idx, num_folds=num_folds)
        
        # # Save reconstructions for this fold
        # path = '/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/VQ-GAN/'
        # fold_save_path = os.path.join(path, f"fold_{fold_idx}_reconstructions")
        # save_reconstructions(save_path=fold_save_path, vqgan=vqgan, val_dataset=val_dataset)
        
