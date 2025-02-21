from torch.utils.data import Dataset, ConcatDataset

class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.len1 = len(dataset1)
        self.len2 = len(dataset2)

    def __len__(self):
        return self.len1 + self.len2  # Total size of combined dataset

    def __getitem__(self, index):
        if index < self.len1:
            return self.dataset1[index]  # Get sample from dataset1
        else:
            return self.dataset2[index - self.len1]  # Get sample from dataset2
