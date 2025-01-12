import os
from PIL import Image
from torch.utils.data import Dataset

class FilterDataset(Dataset):
    def __init__(self, original_dir, filtered_dir, transform=None):
        self.original_dir = original_dir
        self.filtered_dir = filtered_dir
        self.transform = transform
        self.image_names = os.listdir(original_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        original_path = os.path.join(self.original_dir, self.image_names[idx])
        filtered_path = os.path.join(self.filtered_dir, self.image_names[idx])

        original_image = Image.open(original_path).convert("RGB")
        filtered_image = Image.open(filtered_path).convert("RGB")

        if self.transform:
            original_image = self.transform(original_image)
            filtered_image = self.transform(filtered_image)

        return original_image, filtered_image
