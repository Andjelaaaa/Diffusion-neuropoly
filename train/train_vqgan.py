"Adapted from https://github.com/SongweiGe/TATS"

# add the main folder to the path so the modules can be imported without errors
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
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

def visualize_and_save_cpdataset(tsv_path, results_folder, is_VQGAN=False, num_samples=5):
    """
    Visualize and save the middle slice of preprocessed images from CPDataset.
    
    Args:
        tsv_path (str): Path to the participants.tsv file.
        results_folder (str): Path to the folder where plots will be saved.
        is_VQGAN (bool): Whether to use VQGAN-specific transformations.
        num_samples (int): Number of samples to visualize.
    """
    # Initialize the dataset
    dataset = CPDataset(tsv_path=tsv_path, sp_size=256, augmentation=True)

    # Create results folder if it doesn't exist
    os.makedirs(results_folder, exist_ok=True)

    print(f"Dataset size: {len(dataset)} samples")
    num_samples = min(num_samples, len(dataset))

    for i in range(num_samples):
        sample = dataset[i]  # Get a single data sample
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

        # Plot and save the middle slice
        plt.figure()
        plt.imshow(middle_slice, cmap='gray')
        plt.axis('off')
        plt.title(f"Sample {i + 1}, Middle Slice {middle_idx + 1}")
        
        # Save the plot
        save_path = os.path.join(results_folder, f"sample_{i + 1}_middle_slice.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    print(f"Saved middle slice plots to {results_folder}")

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
@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    # Set the GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.model.gpu_index)
    
    pl.seed_everything(cfg.model.seed)

    num_folds = 5  # Number of folds for cross-validation

    original_root_dir = os.path.join(cfg.model.default_root_dir, cfg.dataset.name)  # Preserve the original root directory

    for fold_idx in range(num_folds):
        print(f"Training Fold {fold_idx + 1}/{num_folds}")
        
        # Get datasets for the current fold
        train_dataset, val_dataset, sampler = get_dataset(cfg, fold_idx=fold_idx, num_folds=num_folds)
        
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
            # ImageLogger(batch_frequency=3000, max_images=4, clamp=True),
            # VideoLogger(batch_frequency=1500, max_videos=4, clamp=True)
        ]

        # Trainer setup
        trainer = pl.Trainer(
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
    run()
    # tsv_path = "/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/all-participants.tsv"  # Replace with the actual path
    # results_folder = "./results/"  # Replace with the desired folder
    # visualize_and_save_cpdataset(tsv_path=tsv_path, results_folder=results_folder, is_VQGAN=False, num_samples=5)
