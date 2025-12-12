import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import os 
from tqdm import tqdm


from dataset import CIFAKEDataset
from models.artifact_aware_cnn import ArtifactAwareCNN
from models.baseline_cnn import BaselineCNN
from models.cnn import CIFAKECNN
from models.vgg16 import VGG16CIFAKE



def get_dataloaders(root_dir="../data", batch_size=64, model_type = "vgg16"):
    """
    Creates pytorch dataloaders for traning and testing datasets.
    Parameters:
      root_dir: path to data folder containing 'train/' and 'test/' subfolders.
     batch_size: number of samples per batch.

    Returns:
      train_loader, test_loader: DataLoader objects for training and testing datasets.

    """
    
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
    ])

    """
    VGG16 expects 224*224 RGB and ImageNet normalization so we need a different transform
    """
    vgg_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ])
    
    if model_type=="vgg16":

        train_dataset = CIFAKEDataset(root_dir=root_dir, split="train", transform=vgg_transform)
        test_dataset = CIFAKEDataset(root_dir=root_dir, split="test", transform=vgg_transform)
    else:
        train_dataset = CIFAKEDataset(root_dir=root_dir, split="train", transform=transform)
        test_dataset = CIFAKEDataset(root_dir=root_dir, split="test", transform=transform)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, device):
    """
    Trains the model over one epoch
    Parameters:
      model: the neural network model to train.
      train_loader: DataLoader for training data.
      criterion: loss function.
      optimizer: optimization algorithm.
      device: computation device (CPU or GPU).

      Returns:
      avg_loss: average training loss over the epoch.
      accuracy: training accuracy over the epoch.
    """

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0


    for imgs, labels in tqdm(train_loader, desc="Training", leave=False):
        imgs = imgs.to(device)
        labels =labels.to(device)


        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        total_loss += loss.item() * imgs.size(0)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluates the model on the test dataset.
    Parameters:
      model: the neural network model to evaluate.
      test_loader: DataLoader for test data.
      criterion: loss function.
      device: computation device (CPU or GPU).

    Returns:
      avg_loss: average test loss.
      accuracy: test accuracy.
    """

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main(model_type="baseline"):
    """
    model_type: "baseline", "paper", "vgg", or "artifact"
    """
    if torch.backends.mps.is_available():
      device = torch.device("mps")
      print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print("Device:", device)



    train_loader, test_loader = get_dataloaders(root_dir="../data", batch_size=64, model_type=model_type)

    if model_type == "baseline":
        print("Training BaselineCNN...")
        model = BaselineCNN().to(device)
        ckpt_name = "baseline_cnn.pt"

    elif model_type == "paper":
        print("Training paper CNN (CIFAKECNN)...")
        model = CIFAKECNN().to(device)
        ckpt_name = "paper_cnn.pt"

    elif model_type=="vgg16":
        print("Training VGG16CIFAKE...")
        model = VGG16CIFAKE(pretrained=True, freeze_backbone=True).to(device)
        ckpt_name = "vgg16.pt"

    else:
        print("Training ArtifactAwareCNN...")
        model = ArtifactAwareCNN().to(device)
        ckpt_name = "artifact_aware_cnn.pt"

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if model_type=="vgg16":
        num_epochs = 5
    else:
        num_epochs = 20

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")
        
    #Saving trained models for interpretability later
    os.makedirs("../checkpoints", exist_ok=True)
    ckpt_path = os.path.join("../checkpoints", ckpt_name)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved {model_type} model to {ckpt_path}")


if __name__ == "__main__":
    # "baseline", "paper", "vgg16", or "artifact"
    main(model_type="vgg16")





