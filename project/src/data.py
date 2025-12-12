from torch.utils.data import DataLoader
from dataset import CIFAKEDataset


def test_loader():
    dataset = CIFAKEDataset(root_dir="../data", split="train")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for imgs, labels in loader:
        print("Batch image tensor shape:", imgs.shape)  
        print("Batch labels:", labels)                 
        break


if __name__ == "__main__":
    test_loader()