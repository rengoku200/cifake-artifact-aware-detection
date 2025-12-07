import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import torchvision.transforms as T

from captum.attr import IntegratedGradients, Occlusion

from dataset import CIFAKEDataset
from models.baseline_cnn import BaselineCNN
from models.cnn import CIFAKECNN       
from models.artifact_aware_cnn import ArtifactAwareCNN


def load_model(model_name, device=None):
    """
    Loads any trained model by name:
    - 'baseline'
    - 'paper'
    - 'artifact'
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "baseline":
        model = BaselineCNN().to(device)
        ckpt = "../checkpoints/baseline_cnn.pt"

    elif model_name == "paper":
        model = CIFAKECNN().to(device)
        ckpt = "../checkpoints/paper_cnn.pt"

    elif model_name == "artifact":
        model = ArtifactAwareCNN().to(device)
        ckpt = "../checkpoints/artifact_aware_cnn.pt"

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model: {model_name} ✔️")

    return model


def get_sample(batch_size=1):
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
    ])

    test_ds = CIFAKEDataset(root_dir="../data", split="test", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    return next(iter(test_loader))


def visualize(img, ig_attr, occ_attr, save_path=None):
    img_np = img.permute(1, 2, 0).cpu().numpy()
    ig_map = ig_attr.mean(dim=0).cpu().numpy()
    occ_map = occ_attr.mean(dim=0).cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))

    axs[0].imshow(img_np)
    axs[0].set_title("Original")
    axs[0].axis("off")

    im1 = axs[1].imshow(ig_map, cmap="hot")
    axs[1].set_title("Integrated Gradients")
    axs[1].axis("off")
    fig.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(occ_map, cmap="hot")
    axs[2].set_title("Occlusion")
    axs[2].axis("off")
    fig.colorbar(im2, ax=axs[2])

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved heatmap to {save_path}")
    else:
        plt.show()


def run_ig(model, img, label, device):
    img, label = img.to(device), label.to(device)

    def forward_fn(x):
        logits = model(x)
        return logits[:, label.item()]  # logit for class

    ig = IntegratedGradients(forward_fn)

    attributions, _ = ig.attribute(
        img,
        baselines=torch.zeros_like(img),
        return_convergence_delta=True
    )

    return attributions



def run_occlusion(model, img, label, device):
    img, label = img.to(device), label.to(device)

    def forward_fn(x):
        probs = F.softmax(model(x), dim=1)
        return probs[:, label.item()]

    occ = Occlusion(forward_fn)

    attributions = occ.attribute(
        img,
        strides=(1, 8, 8),
        sliding_window_shapes=(3, 16, 16),
        baselines=0,
    )

    return attributions


def frequency_ablation(model, img, device):
    img = img.to(device)

    with torch.no_grad():
        base_prob = F.softmax(model(img), dim=1)[0, 1].item()

    print(f"Baseline fake probability: {base_prob:.4f}")

    # Prepare FFT
    freq = torch.fft.fft2(img)
    freq = torch.fft.fftshift(freq, dim=(-2, -1))

    b, c, h, w = freq.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 4

    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    dist = ((Y - cy)**2 + (X - cx)**2).sqrt().to(device)

    for band in ["low", "high"]:
        mask = torch.ones_like(freq)

        if band == "low":
            mask[..., dist <= radius] = 0
        else:
            mask[..., dist > radius] = 0

        masked = freq * mask
        masked = torch.fft.ifftshift(masked, dim=(-2, -1))
        x_masked = torch.fft.ifft2(masked).real.clamp(0, 1)

        with torch.no_grad():
            p = F.softmax(model(x_masked), dim=1)[0, 1].item()

        print(f"[{band.upper()} frequencies removed] fake_prob = {p:.4f}")




def main(model_name="artifact"):
    if torch.backends.mps.is_available():
      device = torch.device("mps")
      print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print("Device:", device)

    model = load_model(model_name, device)
    imgs, labels = get_sample(batch_size=1)
    img = imgs[0:1]
    label = labels[0]

    # Generate attributions
    ig_attr = run_ig(model, img, label, device)
    occ_attr = run_occlusion(model, img, label, device)

    # Save visualization
    save_path = f"../interpret_{model_name}.png"
    visualize(img[0], ig_attr[0], occ_attr[0], save_path)

    # Frequency ablation
    frequency_ablation(model, img, device)


if __name__ == "__main__":
    # change to "baseline", "paper", or "artifact"
    main(model_name="artifact")