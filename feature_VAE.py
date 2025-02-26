import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import timm
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

# Configurations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LATENT_DIM = 128  # You can change this to 256
FEATURE_DIM = 1024
EPOCHS = 50
LEARNING_RATE = 1e-3
IMAGE_SIZE = 224  # Swin-base expects 224x224 images
NUM_WORKERS = 4

# Feature Extraction Model (Swin-Base)
# You can modify to load from local directory
class FeatureExtractor(nn.Module):
    def __init__(self, model_name="swin_base_patch4_window7_224", feature_dim=FEATURE_DIM):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)  # Removes classification head
        self.feature_dim = feature_dim
    
    def forward(self, x):
        return self.model(x)  # Returns 1024D feature vector

# ImageNet Dataset with Feature Extraction
class ImageNetFeatureDataset(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = datasets.ImageFolder(root, transform=transform)
        self.feature_extractor = FeatureExtractor().to(DEVICE).eval()
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]  # Ignore label
        img = img.to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            feature = self.feature_extractor(img).cpu().squeeze()
        return feature

# Define VAE Model
class VAE(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, latent_dim=LATENT_DIM):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.fc4 = nn.Linear(latent_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return self.fc6(h)  # No activation since it's feature reconstruction

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss Function (VAE)
def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")  # MSE loss for reconstruction
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence loss
    return recon_loss + kl_loss

# Training Function
def train_vae(vae, train_loader, optimizer, epochs=EPOCHS):
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(batch)
            loss = loss_function(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # Save trained VAE
    torch.save(vae.state_dict(), "vae_swin_base.pth")
    print("VAE model saved!")

# Main Execution
if __name__ == "__main__":
    # Load ImageNet Features Dataset
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    imagenet_train = ImageNetFeatureDataset(root="/path/to/imagenet/train", transform=transform)
    train_loader = DataLoader(imagenet_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # Initialize VAE
    vae = VAE(input_dim=FEATURE_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)

    # Train VAE
    train_vae(vae, train_loader, optimizer, epochs=EPOCHS)
