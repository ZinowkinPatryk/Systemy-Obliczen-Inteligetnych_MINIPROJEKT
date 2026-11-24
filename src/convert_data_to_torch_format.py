import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet18_Weights
transforms = ResNet18_Weights.DEFAULT.transforms()


class HFImageDataset(Dataset):
    def __init__(self, hf_split, transform=None):
        self.dataset = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)