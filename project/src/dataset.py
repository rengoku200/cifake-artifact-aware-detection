import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class CIFAKEDataset(Dataset):
    def __init__(self, root_dir="../data", split="train", transform=None):
        """
        root_dir: path to the data folder that contains 'train/' and 'test/'
        split: 'train' or 'test'
        """
        base_dir = os.path.join(root_dir, split)
        self.real_dir = os.path.join(base_dir, "REAL")
        self.fake_dir = os.path.join(base_dir, "FAKE")

        #pairs: REAL = 0, FAKE = 1
        self.samples = []

        for fname in os.listdir(self.real_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                self.samples.append((os.path.join(self.real_dir, fname), 0))

        for fname in os.listdir(self.fake_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                self.samples.append((os.path.join(self.fake_dir, fname), 1))

        # simple transform 
        self.transform = transform or T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label
