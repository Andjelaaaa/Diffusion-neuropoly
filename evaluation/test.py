
import os
# GPU_ID = 1 
# os.environ["CUDA_VISIBLE_DEVICES"]=f"{GPU_ID}"
# device = f'cuda:0'
import sys
from re import I
import re
import csv
sys.path.append('../')
import importlib
import datetime
import torch
import torch.nn.functional as F

# Add the root directory of the project to sys.path
project_root = os.path.abspath("/home/andim/Diffusion-neuropoly")
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

def calculate_and_store_recon_loss(csv_path, sub_id, scan_id, x_recon, x, l1_weight):
    """
    Calculate the reconstruction loss and store it in a CSV file.

    Args:
        csv_path (str): Path to the CSV file where results will be stored.
        sub_id (str): Subject ID.
        scan_id (str): Scan ID.
        x_recon (torch.Tensor): Reconstructed data.
        x (torch.Tensor): Original input data.
        l1_weight (float): Weight for the L1 loss.
    """
    # Calculate reconstruction loss
    recon_loss = F.l1_loss(x_recon, x) * l1_weight

    # Ensure the CSV file exists and write headers if not
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            # Write header if the file is being created
            writer.writerow(["Subject ID", "Scan ID", "Reconstruction Loss"])
        
        # Write the loss data
        writer.writerow([sub_id, scan_id, recon_loss.item()])


def save_reconstructions(save_path, vqgan, val_dataset, l1_weight=4.0):
    """
    Save the reconstructions from the VQGAN model for the validation set and calculate reconstruction loss.

    Args:
        save_path (str): Directory where the reconstructions will be saved.
        vqgan (torch.nn.Module): The trained VQGAN model.
        val_dataset (torch.utils.data.Dataset): Validation dataset.
        l1_weight (float): Weight for the L1 loss.
    """
    # Ensure the save path exists
    os.makedirs(save_path, exist_ok=True)

    # Set the CSV path dynamically based on save_path
    csv_path = os.path.join(save_path, "reconstruction_losses.csv")

    # Put the model in evaluation mode
    vqgan.eval()

    # Iterate over the validation dataset
    for idx in tqdm(range(len(val_dataset)), desc="Saving Reconstructions"):
        # Get the data and scan_id
        sample = val_dataset[idx]
        input_data = torch.tensor(sample['data'][None]).to(device)  # Add batch dimension
        sub_id = sample['sub_id']
        scan_id = sample['scan_id']
        
        with torch.no_grad():
            # Encode and decode to get the reconstruction
            latent_z = vqgan.pre_vq_conv(vqgan.encoder(input_data))
            reconstructed = vqgan.decode(latent_z)

        # Convert input and reconstruction to numpy for saving
        input_numpy = input_data[0, 0, :, :, :].cpu().numpy()  # Input shape: 1,1,64,128,128
        reconstructed_numpy = reconstructed[0, 0, :, :, :].cpu().numpy()

        # Save the input and reconstruction as .nii files 
        input_nifti = nib.Nifti1Image(np.transpose(input_numpy, (2, 1, 0)), affine=sample['upt_affine'])  # Use original affine
        reconstructed_nifti = nib.Nifti1Image(np.transpose(reconstructed_numpy, (2, 1, 0)), affine=sample['upt_affine'])

        nib.save(reconstructed_nifti, os.path.join(save_path, f"{sub_id}_{scan_id}_reconstructed.nii.gz"))
        nib.save(input_nifti, os.path.join(save_path, f"{sub_id}_{scan_id}_input.nii.gz"))

        # Calculate and store reconstruction loss
        calculate_and_store_recon_loss(csv_path, sub_id, scan_id, reconstructed, input_data, l1_weight)


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

def save_reconstruction_pngs(save_path, checkpoints, fold_idx, device, num_fold):
    """
    Save a side-by-side PNG image showing the input and reconstructions for a single sample.

    Args:
        save_path (str): Directory where the PNGs will be saved.
        checkpoints (list): List of checkpoint paths for the fold.
        fold_idx (int): Fold index for naming and organization.
        device (torch.device): The device (CPU/GPU) for processing.
    """

    os.makedirs(save_path, exist_ok=True)

    # Load models from checkpoints
    models = []
    recon_losses = []

    for checkpoint in checkpoints:
        model = VQGAN.load_from_checkpoint(checkpoint).to(device)
        model.eval()
        models.append(model)

    # Get the validation dataset for this fold
    val_dataset = get_val_dataset(models[0], checkpoints[0], 'CP', fold_idx, num_fold)

    # Use the first sample from the validation set
    sample = val_dataset[0]
    input_data = torch.tensor(sample['data'][None]).to(device)  # Add batch dimension
    sub_id = sample['sub_id']
    scan_id = sample['scan_id']

    # Generate reconstructions for each model and calculate reconstruction losses
    reconstructions = []
    for model in models:
        with torch.no_grad():
            vq_output = model.encoder(input_data)
            latent_z = model.pre_vq_conv(vq_output)
            reconstructed = model.decode(latent_z)
            reconstructions.append(reconstructed[0, 0, :, :, :].cpu().numpy())
            
            print(f"Encoder output shape: {vq_output.shape}") #[1, 32, 32, 64, 64]
            
            print(f"pre_vq_conv output shape: {latent_z.shape}") #[1, 8, 32, 64, 64]
            
            print(f"Decoder input shape: {latent_z.shape}, output shape: {reconstructed.shape}")
            # Decoder input shape: torch.Size([1, 8, 32, 64, 64]), output shape: torch.Size([1, 1, 64, 128, 128])
            
            # Compute reconstruction loss
            # x_recon = model.decoder(model.pre_vq_conv(vq_output))
            x_recon = model.decoder(vq_output)
            recon_loss = F.l1_loss(x_recon, input_data) * model.l1_weight
            recon_losses.append(recon_loss.item())

    # Convert input data to numpy
    input_numpy = input_data[0, 0, :, :, :].cpu().numpy()

    # Plot the input and reconstructions
    fig, axes = plt.subplots(1, len(models) + 1, figsize=(16, 4))
    
    # Plot the input
    middle_idx = input_numpy.shape[0] // 2  # Middle slice
    axes[0].imshow(input_numpy[middle_idx], cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    # Plot reconstructions
    for i, (reconstruction, loss) in enumerate(zip(reconstructions, recon_losses)):
        axes[i + 1].imshow(reconstruction[middle_idx], cmap="gray")
        title = f"Recon {i + 1}\nLoss: {loss:.4f}"
        axes[i + 1].set_title(title)
        axes[i + 1].axis("off")

    # Save the plot
    png_save_path = os.path.join(
        save_path, f"fold_{fold_idx}_sample_{sub_id}_{scan_id}.png"
    )
    plt.tight_layout()
    plt.savefig(png_save_path, bbox_inches="tight")
    plt.close()

    print(f"Saved reconstruction visualization for fold {fold_idx} to {png_save_path}")

def plot_generated_samples(vqgan, train_dataset, device, epoch, sample_indices, exp_name):
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
        # axes[row, col * 2].imshow(input_[0, 0, input_.shape[2] // 2, :, :].cpu().numpy(), cmap='gray')
        axes[row, col * 2].imshow(input_[0, 0, :, input_.shape[3] // 2, :].cpu().numpy(), cmap='gray')
        axes[row, col * 2].set_title(f"Sample {sample_idx} - Input")
        axes[row, col * 2].axis('off')

        # Plot the output middle slice
        # axes[row, col * 2 + 1].imshow(output[0, 0, output.shape[2] // 2, :, :].cpu().numpy(), cmap='gray')
        axes[row, col * 2].imshow(output[0, 0, :, output.shape[3] // 2, :].cpu().numpy(), cmap='gray')
        axes[row, col * 2 + 1].set_title(f"Sample {sample_idx} - Output")
        axes[row, col * 2 + 1].axis('off')

    # Remove empty subplots if the number of samples is odd
    for ax in axes.flat[num_samples * 2:]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'generated_vq_gan_outputs_epoch_{epoch}_{exp_name}.png')
    plt.close(fig)

def extract_scan_id(filename):
    # Use a regex to match the desired scan ID pattern
    match = re.search(r'_([A-Za-z0-9_\-]+)_(?:input|reconstructed)\.nii\.gz$', filename)
    if match:
        return match.group(1)  # Extract the matched group
    return None

def explore_data_dist_per_fold(tsv_path, fold_path, fold, exp_name):
    data = pd.read_csv(tsv_path, sep='\t')

    # Extract relevant columns: scan_id and age
    scan_age_data = data[['scan_id', 'age']]

    # Directory containing NIfTI files (simulate here with filenames)
    scan_ids = []
    for file in os.listdir(fold_path):
        if file.endswith('.nii.gz'):
            scan_ids.append(extract_scan_id(file))

    # Filter the data for relevant scan IDs
    matched_data = scan_age_data[scan_age_data['scan_id'].isin(scan_ids)]
    # print(matched_data.head())

    # Plot the age distribution
    plt.figure(figsize=(10, 6))
    plt.hist(matched_data['age'], bins=20, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Age Distribution of Matched Data')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f'age_dist_{exp_name}_fold_{fold}.png')

def explore_data_dist_all_folds(tsv_path, num_folds, exp_name):
    # Load the TSV file
    data = pd.read_csv(tsv_path, sep='\t')

    # Extract relevant columns: scan_id and age
    scan_age_data = data[['scan_id', 'age', 'sex', 'sub_id_bids']]

    plt.figure(figsize=(10, 6))  # Create a single figure for overlaying plots

    # Iterate through each fold
    for fold_idx in range(num_folds):
        # Generate the fold path
        fold_path = f'/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/VQ-GAN/{exp_name}/fold_{fold_idx}_reconstructions'

        scan_ids = []
        for file in os.listdir(fold_path):
            if file.endswith('.nii.gz'):
                scan_ids.append(extract_scan_id(file))

        # Filter the data for relevant scan IDs
        matched_data = scan_age_data[scan_age_data['scan_id'].isin(scan_ids)]
        print(matched_data.describe())
        # Ensure each scan_id is unique
        unique_matched_data = matched_data.drop_duplicates(subset='sub_id_bids')
        print(unique_matched_data.describe())
        # Keep only the sex column
        sex_data = unique_matched_data[['sex']]
        # Print the counts
        print("Number of 0s:", sex_data.value_counts().get(0, 0))
        print("Number of 1s:", sex_data.value_counts().get(1, 0))

        # Plot the age distribution for this fold
        plt.hist(
            matched_data['age'],
            bins=20,
            alpha=0.5,  # Make the bars semi-transparent for overlaying
            label=f'Fold {fold_idx}'
        )

    # Add title, labels, and legend
    plt.title('Age Distribution Across Folds')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.legend()
    plt.savefig(f'age_distribution_overlaid_{exp_name}.png') 

def get_val_dataset(model, checkpoint_path, dataset_name, fold_idx, num_fold):
    """
    Get the validation dataset for the specified fold.

    Args:
        model (VQGAN): The model instance.
        checkpoint_path (str): Path to the VQGAN checkpoint.
        dataset_name (str): The name of the dataset (e.g., 'CP').
        fold_idx (int): Fold index for cross-validation.
    
    Returns:
        torch.utils.data.Dataset: Validation dataset for the fold.
    """
    if dataset_name == 'CP':
        with initialize(config_path="../config/", version_base="1.1"):
            cfg = compose(config_name="base_cfg.yaml", overrides=[
                "model=ddpm",
                f"dataset.tsv_path={tsv_path}",
                f"model.vqgan_ckpt='{checkpoint_path}'",
                "model.diffusion_img_size=32",
                "model.diffusion_depth_size=32",
                "model.diffusion_num_channels=8",
                "model.dim_mults=[1,2,4,8]",
                "model.batch_size=2",
                "++model.gpus=1",
            ])
            print("Configuration loaded successfully.")
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented.")

    # Get the validation dataset
    _, val_dataset, _ = get_dataset(cfg, fold_idx=fold_idx, num_folds=num_fold)  # Adjust num_folds if necessary
    return val_dataset

def reconstruct_combined_dataset(exp_name, dataset_name, fold_idx, checkpoint_path, scratch_dir, dataset_base_path, device="cuda:0"):
    """
    Reconstructs validation set images from the combined dataset (CP + BCP) using VQGAN.
    Saves both inputs & reconstructions as seen by the network (before Invertd).

    Args:
        exp_name (str): Experiment name.
        fold_idx (int): Fold index (e.g., 0, 1, 2, etc.).
        checkpoint_path (str): Path to the VQGAN model checkpoint.
        scratch_dir (str): Base directory where outputs will be saved.
        dataset_base_path (str): Path to the base directory containing datasets.
        device (str): The device to run computations on (default: "cuda:0").
    """

    # Set up the evaluation directory
    save_path = os.path.join(scratch_dir, "evaluation", exp_name, f"fold_{fold_idx}_reconstructions")
    os.makedirs(save_path, exist_ok=True)

    # Load the trained VQGAN model
    print(f"Loading model from: {checkpoint_path}")
    vqgan = VQGAN.load_from_checkpoint(checkpoint_path).to(device)
    vqgan.eval()

    with initialize(config_path="../config/", version_base="1.1"):
        cfg = compose(config_name="base_cfg.yaml", overrides=[
            f"dataset={dataset_name}",
            f"dataset.tsv_path={dataset_base_path}",
            f"+model.vqgan_ckpt='{checkpoint_path}'",
            "+model.diffusion_img_size=32",
            "+model.diffusion_depth_size=32",
            "+model.diffusion_num_channels=8",
            "+model.dim_mults=[1,2,4,8]",
            "model.batch_size=2",
            "++model.gpus=1",
        ])

    # Load validation dataset using get_dataset (which internally handles COMBINED)
    print(f"Loading validation dataset for fold {fold_idx}...")
    _, val_dataset, _ = get_dataset(cfg, fold_idx=fold_idx, num_folds=5)  # Adjust `num_folds` if needed

    print(f"Fold {fold_idx}: Validation dataset size = {len(val_dataset)}")

    # Iterate through the validation dataset and generate reconstructions
    print(f"Generating and saving network inputs and outputs for fold {fold_idx}...")
    for idx in tqdm(range(len(val_dataset)), desc="Processing images"):
        # Load sample
        sample = val_dataset[idx]
        input_data = torch.tensor(sample['data'][None]).to(device)  # Add batch dimension
        sub_id = sample['subject_id']
        scan_id = sample['session_id']  # Ensure session_id is correctly used

        with torch.no_grad():
            # Encode and decode
            latent_z = vqgan.pre_vq_conv(vqgan.encoder(input_data))
            reconstructed = vqgan.decode(latent_z)

        # Convert tensors to numpy (before inverting transforms!)
        input_numpy = input_data[0, 0, :, :, :].cpu().numpy()
        reconstructed_numpy = reconstructed[0, 0, :, :, :].cpu().numpy()

        # Save as NIfTI files (network-preprocessed version)
        input_nifti = nib.Nifti1Image(input_numpy, affine=np.eye(4))  # Identity affine (not original)
        reconstructed_nifti = nib.Nifti1Image(reconstructed_numpy, affine=np.eye(4))

        input_path = os.path.join(save_path, f"{sub_id}_{scan_id}_input_network.nii.gz")
        recon_path = os.path.join(save_path, f"{sub_id}_{scan_id}_reconstructed_network.nii.gz")

        nib.save(input_nifti, input_path)
        nib.save(reconstructed_nifti, recon_path)

        print(f"Saved: {input_path} and {recon_path}")

    print(f"All network-space reconstructions saved in: {save_path}")

if __name__ == '__main__':
    exp_name = "exp_combined_5fold"
    fold_idx = 0
    checkpoint_path = "/home/andim/scratch/COMBINED/exp_combined_5fold/fold_0/wandb_logs/VQ-GAN/3ctf1vvp/checkpoints/latest_checkpoint_fold_0-val_loss=val/recon_loss=0.0970.ckpt"
    scratch_dir = "/home/andim/scratch"
    dataset_base_path = "/home/andim/projects/def-bedelb/andim/"
    dataset_name = 'combined'

    reconstruct_combined_dataset(exp_name, dataset_name, fold_idx, checkpoint_path, scratch_dir, dataset_base_path)


    # tsv_path = '/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/all-participants.tsv'
    # dataset_name = "cp"
    
    # USE_DATASET = 'CP'
    

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

    # train_dataset = get_dataset(cfg)

    # vqgan = VQGAN.load_from_checkpoint(VQGAN_CHECKPOINT)
    # vqgan.decoding_diviser = 3  # An odd integer for VRAM optimization
    # vqgan = vqgan.to(device)
    # vqgan.eval()

    # # Split the path and extract the epoch part
    # for part in VQGAN_CHECKPOINT.split('/'):
    #     if 'epoch=' in part:
    #         epoch = int(part.split('=')[1].split('-')[0])  # Extract and convert to integer
    
    # # # Plot generated samples for a specific epoch
    # # plot_generated_samples(train_dataset, device, epoch, [0,1,2,3])
    # # plot_generated_samples(vqgan, train_dataset, device, epoch, [0,1,2,3])
    # plot_generated_samples(vqgan, train_dataset, device, epoch, [0,1,2,3], 'first')

    # # Base folder to start the search
    # # base_folder = "../checkpoints/vq_gan/CP/lightning_logs"
    # # find_best_ckpt(base_folder)

    # # Plot recon loss
    # # base_folder = "../checkpoints/vq_gan/CP/lightning_logs/version_1/checkpoints"
    # # plot_recon_loss_per_epoch(base_folder)
    # # get_training_duration(base_folder)

    # #################################################################################
    # ########################VQ-GAN Testing on reconstructions per subject############

    # # import torch
    # # print(torch.cuda.device_count())  # Number of available GPUs
    # # for i in range(torch.cuda.device_count()):
    # #     print(f"Device {i}: {torch.cuda.get_device_name(i)}")

    # # num_folds = 5
    # num_folds = 3
    # paths_for_fold = []
    # # exp_name = f'exp_{num_folds}_folds'
    # exp_name = f'exp_{num_folds}_fold'
    
    # # Plot the age distribution for all folds in one graph
    # # explore_data_dist_all_folds(tsv_path, num_folds, exp_name)

    # # for fold_idx in range(num_folds):
    #     # base_folder = f"../checkpoints/vq_gan/CP/fold_{fold_idx}/lightning_logs"
    #     # base_folder = f"../checkpoints/vq_gan/CP/{exp_name}/fold_{fold_idx}/lightning_logs"
    #     # path = find_best_ckpt(base_folder)
    #     # paths_for_fold.append(path)

    #     # Plot recon loss
    #     # base_folder = f"../checkpoints/vq_gan/CP/fold_{fold_idx}/lightning_logs/version_0/checkpoints"
    #     # if fold_idx == 2 and '3' in exp_name:
    #     #     base_folder = f"../checkpoints/vq_gan/CP/fold_{fold_idx}/lightning_logs/version_1/checkpoints"
    #     # # plot_recon_loss_per_epoch(base_folder, f'fold_{fold_idx}')
    #     # get_training_duration(base_folder)

    #     # Analyze age dist
    #     # fold_path = f'/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/VQ-GAN/{exp_name}/fold_{fold_idx}_reconstructions'
    #     # explore_data_dist_per_fold(tsv_path, fold_path, fold_idx, exp_name)

        
    #     # if USE_DATASET == 'CP':
    #     #     # VQGAN_CHECKPOINT = path
    #     #     # VQGAN_CHECKPOINT = f"../checkpoints/vq_gan/CP/{exp_name}/fold_{fold_idx}/lightning_logs/version_0/checkpoints/latest_checkpoint_fold_{fold_idx}-v2.ckpt"
    #     #     # if fold_idx == 2:
    #     #     #     VQGAN_CHECKPOINT = f"../checkpoints/vq_gan/CP/{exp_name}/fold_{fold_idx}/lightning_logs/version_1/checkpoints/latest_checkpoint_fold_{fold_idx}-v2.ckpt"

    #     #     if fold_idx == 0:
    #     #         VQGAN_CHECKPOINT = f"../checkpoints/vq_gan/CP/{exp_name}/fold_{fold_idx}/lightning_logs/version_0/checkpoints/latest_checkpoint_fold_{fold_idx}-val_loss=val/recon_loss=0.0784.ckpt"
    #     #     elif fold_idx == 1:
    #     #         VQGAN_CHECKPOINT = f"../checkpoints/vq_gan/CP/{exp_name}/fold_{fold_idx}/lightning_logs/version_0/checkpoints/latest_checkpoint_fold_{fold_idx}-val_loss=val/recon_loss=0.1210.ckpt"
    #     #     else:
    #     #         VQGAN_CHECKPOINT = f"../checkpoints/vq_gan/CP/{exp_name}/fold_{fold_idx}/lightning_logs/version_0/checkpoints/latest_checkpoint_fold_{fold_idx}-val_loss=val/recon_loss=0.2947.ckpt"
    #     #     # Specify the version_base explicitly
    #     #     with initialize(config_path="../config/", version_base="1.1"):
    #     #             cfg = compose(config_name="base_cfg.yaml", overrides=[
    #     #                 "model=ddpm",
    #     #                 f"dataset.tsv_path={tsv_path}",
    #     #                 #f"dataset={dataset_name}",
    #     #                 f"model.vqgan_ckpt='{VQGAN_CHECKPOINT}'",
    #     #                 "model.diffusion_img_size=32",
    #     #                 "model.diffusion_depth_size=32",
    #     #                 "model.diffusion_num_channels=8",
    #     #                 "model.dim_mults=[1,2,4,8]",
    #     #                 "model.batch_size=2",
    #     #                 "++model.gpus=1",
    #     #             ])
    #     #             print("yahoo")
    #     # else:
    #     #     print('Not implemented dataset') 

    

    #     # vqgan = VQGAN.load_from_checkpoint(VQGAN_CHECKPOINT)
    #     # vqgan.decoding_diviser = 3  # An odd integer for VRAM optimization
    #     # vqgan = vqgan.to(device)
    #     # print(f"Processing Fold {fold_idx + 1}/{num_folds}")
        
    #     # # Get datasets for the current fold
    #     # _, val_dataset, _ = get_dataset(cfg, fold_idx=fold_idx, num_folds=num_folds)
        
    #     # # Save reconstructions for this fold
    #     # path = f'/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/VQ-GAN/{exp_name}/'
    #     # # path = f'/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/VQ-GAN/{exp_name}/latest_checkpoint/'
    #     # fold_save_path = os.path.join(path, f"fold_{fold_idx}_reconstructions")
    #     # save_reconstructions(save_path=fold_save_path, vqgan=vqgan, val_dataset=val_dataset)

    #     # # # Save reconstructions from latest checkpoints # # #
    #     # checkpoints = [
    #     # f"../checkpoints/vq_gan/CP/fold_{fold_idx}/lightning_logs/version_0/checkpoints/latest_checkpoint_fold_{fold_idx}.ckpt",
    #     # f"../checkpoints/vq_gan/CP/fold_{fold_idx}/lightning_logs/version_0/checkpoints/latest_checkpoint_fold_{fold_idx}-v1.ckpt",
    #     # f"../checkpoints/vq_gan/CP/fold_{fold_idx}/lightning_logs/version_0/checkpoints/latest_checkpoint_fold_{fold_idx}-v2.ckpt",
    #     # ]
    #     # save_path = f'/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/VQ-GAN/exp_5_fold/fold_{fold_idx}_reconstructions'
        
    #     # path = '/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/VQ-GAN/exp_5_fold/'
    #     # path = '/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/VQ-GAN/exp_3_fold/'
    #     # fold_save_path = os.path.join(path, f"fold_{fold_idx}_reconstructions")
        
    #     # save_reconstruction_pngs(save_path, checkpoints, fold_idx, device, num_folds)
        
