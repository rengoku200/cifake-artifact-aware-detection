import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm


from dataset import CIFAKEDataset
from models.baseline_cnn import BaselineCNN 



def get_dataloaders(root_dir="../data", batch_size=64):
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
        imgs.to(device)
        labels.to(device)


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
            imgs.to(device)
            labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    """Main training loop."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(root_dir="../data", batch_size=64)

    model = BaselineCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 2
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")


if __name__ == "__main__":
    main()




