import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import os

# Import components from our first file
from baseline_setup import get_dataloaders, DEVICE, MODEL_PATH_FP32_EDGE
from baseline_setup import get_model, split_and_verify_model

# --- Configuration ---
NUM_EPOCHS_ATTACKER = 5
LR_ATTACKER = 0.001
ATTACKER_MODEL_PATH = "models/decoder_attacker.pth"

# --- 1. Define the Decoder (Attacker) ---

class Decoder(nn.Module):
    """
    This is the attacker's model. It tries to reconstruct the original
    image from the intermediate features.
    """
    def __init__(self):
        super(Decoder, self).__init__()
        # Input: [Batch, 576, 7, 7] -> Output: [Batch, 3, 224, 224]
        # CORRECTED ORDER: Conv -> BatchNorm -> ReLU
        self.layers = nn.Sequential(
            # 7 -> 14
            nn.ConvTranspose2d(576, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 14 -> 28
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 28 -> 56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 56 -> 112
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 112 -> 224
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)

# Helper function to de-normalize images for viewing
def unnormalize(tensor):
    return (tensor * 0.5) + 0.5

# --- 2. Train the Decoder ---

def train_attacker(edge_model, trainloader):
    print(f"Training the attacker (decoder) on {DEVICE}...")
    decoder = Decoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(decoder.parameters(), lr=LR_ATTACKER)

    edge_model.eval()

    for epoch in range(NUM_EPOCHS_ATTACKER):
        decoder.train()
        for images, _ in tqdm(trainloader, desc=f"Training Decoder Epoch {epoch+1}"):
            images = images.to(DEVICE)
            optimizer.zero_grad()

            with torch.no_grad():
                features = edge_model(images)

            reconstructions = decoder(features)
            loss = criterion(reconstructions, images)
            loss.backward()
            optimizer.step()

    print("Decoder training complete.")
    torch.save(decoder.state_dict(), ATTACKER_MODEL_PATH)
    print(f"Decoder model saved to {ATTACKER_MODEL_PATH}")
    return decoder

# --- 3. Evaluate the Baseline (FP32) Attack ---

def evaluate_attack(edge_model, decoder, testloader, filename="fp32_reconstruction.png", title_prefix="FP32"):
    print(f"Evaluating baseline ({title_prefix}) privacy leak...")
    edge_model.eval()
    decoder.eval()

    total_psnr = 0
    total_images = 0

    # Get one batch for visualization
    images, _ = next(iter(testloader))
    images = images.to(DEVICE)

    with torch.no_grad():
        features = edge_model(images)
        reconstructions = decoder(features)

        for i in range(images.shape[0]):
            # Move to CPU for numpy conversion
            original_img = unnormalize(images[i]).cpu().permute(1, 2, 0).numpy()
            reconstructed_img = unnormalize(reconstructions[i]).cpu().permute(1, 2, 0).numpy()

            original_img = original_img.clip(0, 1)
            reconstructed_img = reconstructed_img.clip(0, 1)

            total_psnr += psnr(original_img, reconstructed_img, data_range=1.0)
            total_images += 1

    avg_psnr = total_psnr / total_images
    print(f"Baseline ({title_prefix}) Attack PSNR: {avg_psnr:.2f} dB")

    # --- Save a comparison image ---
    fig, axes = plt.subplots(2, 8, figsize=(20, 6))
    for i in range(8):
        axes[0, i].imshow(unnormalize(images[i]).cpu().permute(1, 2, 0))
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(unnormalize(reconstructions[i]).cpu().permute(1, 2, 0))
        axes[1, i].set_title(f"{title_prefix} Reconstructed")
        axes[1, i].axis("off")

    plt.suptitle(f"{title_prefix} Attack Reconstruction (PSNR: {avg_psnr:.2f} dB)", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {title_prefix} comparison image to {filename}")

    return avg_psnr

# --- Main execution ---
def main():
    full_model = get_model().to(DEVICE)
    edge_model, _ = split_and_verify_model(full_model)
    edge_model.load_state_dict(torch.load(MODEL_PATH_FP32_EDGE, map_location=DEVICE))
    edge_model.eval()

    trainloader, testloader = get_dataloaders()

    if not os.path.exists(ATTACKER_MODEL_PATH):
        decoder = train_attacker(edge_model, trainloader)
    else:
        print(f"Loading existing decoder from {ATTACKER_MODEL_PATH}...")
        decoder = Decoder().to(DEVICE)
        decoder.load_state_dict(torch.load(ATTACKER_MODEL_PATH, map_location=DEVICE))

    evaluate_attack(edge_model, decoder, testloader)

if __name__ == "__main__":
    main()
