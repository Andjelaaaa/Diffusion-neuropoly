from dataset import MRNetDataset, BRATSDataset, ADNIDataset, DUKEDataset, LIDCDataset, DEFAULTDataset, BIDSDataset, CPDataset, BCPDataset, CombinedDataset
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import GroupKFold
import torch


def get_dataset(cfg, fold_idx=None, num_folds=5, d_size=64):
    if cfg.dataset.name == 'BCP':
        dataset = BCPDataset(tsv_path=cfg.dataset.tsv_path, d_size=d_size)
        return dataset
    if cfg.dataset.name == 'COMBINED':  # If using a combined dataset
        # dataset1 = CPDataset(tsv_path=cfg.dataset.tsv_path, d_size=d_size)
        # dataset2 = BCPDataset(root_dir=cfg.dataset.root_dir, is_VQGAN=cfg.dataset.is_VQGAN)
        dataset1 = CPDataset(tsv_path='/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/hc-calgary-preschool/participants.tsv')
        dataset2 = BCPDataset(tsv_path='/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/hc-bcp/participants.tsv')
        
        combined_dataset = CombinedDataset(dataset1, dataset2)

        print(type(dataset1.data_dicts[0].keys()))

        # print('HIIIII', [data["subject_id"] for data in dataset1.data_dicts])

        # Perform cross-validation splits
        if fold_idx is not None:
            # Merge subject IDs for cross-validation
            sub_ids = [data["subject_id"] for data in dataset1.data_dicts] + [data["subject_id"] for data in dataset2.data_dicts]  
            assert len(sub_ids) == len(combined_dataset), "Mismatch between subject IDs and dataset length"
            print(f'There are {len(sub_ids)} subjects of which CP: {len([data["subject_id"] for data in dataset1.data_dicts])} and BCP: {len([data["subject_id"] for data in dataset2.data_dicts])}')
            
            gkf = GroupKFold(n_splits=num_folds)
            indices = list(range(len(combined_dataset)))
            train_indices, val_indices = list(gkf.split(indices, groups=sub_ids))[fold_idx]
            
            train_dataset = torch.utils.data.Subset(combined_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(combined_dataset, val_indices)
            sampler = None  # Adjust sampler if required
            return train_dataset, val_dataset, sampler
        
        return combined_dataset, None, None  # Return dataset without splitting if fold is not specified
    if cfg.dataset.name == 'MRNet':
        train_dataset = MRNetDataset(
            root_dir=cfg.dataset.root_dir, task=cfg.dataset.task, plane=cfg.dataset.plane, split='train')
        val_dataset = MRNetDataset(root_dir=cfg.dataset.root_dir,
                                   task=cfg.dataset.task, plane=cfg.dataset.plane, split='valid')
        sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weight, num_samples=len(train_dataset.sample_weight))
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'BRATS':
        train_dataset = BRATSDataset(
            root_dir=cfg.dataset.root_dir, imgtype=cfg.dataset.imgtype, train=True, severity=cfg.dataset.severity, resize=cfg.dataset.resize)
        val_dataset = BRATSDataset(
            root_dir=cfg.dataset.root_dir, imgtype=cfg.dataset.imgtype, train=True, severity=cfg.dataset.severity, resize=cfg.dataset.resize)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'ADNI':
        train_dataset = ADNIDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        val_dataset = ADNIDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'DUKE':
        train_dataset = DUKEDataset(
            root_dir=cfg.dataset.root_dir)
        val_dataset = DUKEDataset(
            root_dir=cfg.dataset.root_dir)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'LIDC':
        train_dataset = LIDCDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        val_dataset = LIDCDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'DEFAULT':
        train_dataset = DEFAULTDataset(
            root_dir=cfg.dataset.root_dir, is_VQGAN=cfg.dataset.is_VQGAN)
        val_dataset = DEFAULTDataset(
            root_dir=cfg.dataset.root_dir, is_VQGAN=cfg.dataset.is_VQGAN)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'BIDS':
        train_dataset = BIDSDataset(
            root_dir=cfg.dataset.root_dir, is_VQGAN=cfg.dataset.is_VQGAN, contrasts=cfg.dataset.contrasts, derivatives=cfg.dataset.derivatives, mandatory_derivatives=cfg.dataset.mandatory_derivatives)
        val_dataset = BIDSDataset(
            root_dir=cfg.dataset.root_dir, is_VQGAN=cfg.dataset.is_VQGAN, contrasts=cfg.dataset.contrasts, derivatives=cfg.dataset.derivatives, mandatory_derivatives=cfg.dataset.mandatory_derivatives)
        sampler = None
        return train_dataset, val_dataset, sampler
    # if cfg.dataset.name == 'CP':
    #     dataset = CPDataset(tsv_path=cfg.dataset.tsv_path, d_size=d_size)
    #     return dataset
    # # Perform cross-validation splits
    # if fold_idx is not None:
    #     # Extract the scan IDs (or subject IDs)
    #     sub_ids = dataset.sub_ids  # Ensure this corresponds to subject IDs
    #     assert len(sub_ids) == len(dataset), "Mismatch between subject IDs and dataset length"
    #     # Create GroupKFold object
    #     gkf = GroupKFold(n_splits=num_folds)
        
    #     # Split the dataset by groups
    #     indices = list(range(len(dataset)))
    #     train_indices, val_indices = list(gkf.split(indices, groups=sub_ids))[fold_idx]
        
    #     train_dataset = torch.utils.data.Subset(dataset, train_indices)
    #     val_dataset = torch.utils.data.Subset(dataset, val_indices)
    #     sampler = None  # Adjust sampler if required
    #     return train_dataset, val_dataset, sampler
    raise ValueError(f'{cfg.dataset.name} Dataset is not available')
    
    
