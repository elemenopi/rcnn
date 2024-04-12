import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
class CustomDataset(Dataset):
    def __init__(self,X,Y,transform = None):
        self.X = X
        self.Y = Y
        self.transform = transform
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        image = self.X[index]
        label = self.Y[index]
        if self.transform:
            image = self.transform(image)
        return image,label