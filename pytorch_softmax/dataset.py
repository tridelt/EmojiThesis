from torch.utils.data import Dataset

class LoadedDataSet(Dataset):
    def __init__(self, pairs):
        self.data = pairs
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        context, target = self.data[idx]
        return context, target
