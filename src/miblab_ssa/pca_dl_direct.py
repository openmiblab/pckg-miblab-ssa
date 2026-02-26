import os
import logging
import numpy as np
import zarr

# PyTorch Core
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Data & Optimization
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(x.size(0), *self.shape)

class HierarchicalOrderedAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # Encoder: 1x128x128x128 -> latent_dim
        self.encoder = nn.Sequential(
            self._block(1, 16),   # 64^3
            self._block(16, 32),  # 32^3
            self._block(32, 64),  # 16^3
            self._block(64, 128), # 8^3
            nn.Flatten(),
            nn.Linear(128 * 8 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        
        # Decoder: latent_dim -> 1x128x128x128
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128 * 8 * 8 * 8),
            nn.ReLU(),
            Reshape(128, 8, 8, 8),
            self._up_block(128, 64), # 16^3
            self._up_block(64, 32),  # 32^3
            self._up_block(32, 16),  # 64^3
            nn.ConvTranspose3d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # 128^3
            nn.Sigmoid()
        )

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU()
        )

    def _up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose3d(in_c, out_c, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU()
        )

    def forward(self, x, mask_dim=None):
        z = self.encoder(x)
        
        # Nested Dropout: Forces low indices to learn coarse features
        if self.training and mask_dim is not None:
            mask = torch.zeros_like(z)
            mask[:, :mask_dim] = 1.0
            z = z * mask
            
        recon = self.decoder(z)
        return recon, z
    

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, predict, target):
        predict = predict.view(-1)
        target = target.view(-1)
        intersection = (predict * target).sum()
        return 1 - ((2. * intersection + self.smooth) / (predict.sum() + target.sum() + self.smooth))

class ZarrMaskDataset(Dataset):
    def __init__(self, zarr_path):
        self.data = zarr.open(zarr_path, mode='r')['masks'] # (N, 301, 301, 301)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Load and add channel dim: (1, 301, 301, 301)
        return torch.from_numpy(self.data[idx].astype(np.float32)).unsqueeze(0)

def preprocess_mask(x):
    # Crop 301 to 256
    c = (301 - 256) // 2
    x = x[:, :, c:c+256, c:c+256, c:c+256]
    # Downsample to 128
    return F.interpolate(x, size=(128, 128, 128), mode='trilinear', align_corners=False)




def deep_pca_from_mask(
    zarr_path, 
    model_pth_path,
    n_components=128, 
    epochs=100, 
    batch_size=1 # 3D volumes are heavy
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ZarrMaskDataset(zarr_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    model = HierarchicalOrderedAutoencoder(latent_dim=n_components).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = DiceLoss()
    scaler = GradScaler()

    logging.info(f"Training Hierarchical AE on {device}...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for x in loader:
            x = x.to(device)
            
            # 1. Downsample for 128^3 training
            with torch.no_grad():
                x_128 = preprocess_mask(x)
            
            # 2. Randomly select dimensionality for this step
            k = np.random.randint(1, n_components + 1)
            
            # 3. Forward pass with mixed precision
            with autocast():
                recon, _ = model(x_128, mask_dim=k)
                loss = criterion(recon, x_128)

            # 4. Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()

        logging.info(f"Epoch {epoch+1}/{epochs} | Dice Loss: {epoch_loss/len(loader):.6f}")

    # Save model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'latent_dim': n_components,
        'original_shape': dataset.root.attrs.get('original_shape'),
    }
    torch.save(checkpoint, model_pth_path)


def add_deep_pca_from_mask_metrics(
    zarr_path=None, 
    model_pth_path=None,
    batch_size=2  # Keep low for 3D volumes
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Data Loading
    logging.info(f"Connecting to {zarr_path} for metrics calculation...")
    dataset = ZarrMaskDataset(zarr_path)
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 2. Load Model Checkpoint
    checkpoint = torch.load(model_pth_path, map_location=device, weights_only=False)
    latent_dim = checkpoint['latent_dim']
    
    model = HierarchicalOrderedAutoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. Extraction: Run all masks through the encoder
    logging.info("Extracting hierarchical latent features...")
    all_scores = []
    
    with torch.no_grad():
        for x in eval_loader:
            x = x.to(device)
            # Preprocess 301^3 -> 128^3
            x_128 = preprocess_mask(x)
            
            # Forward pass (we only need z)
            # Pass mask_dim=None to ensure we use all features for metrics
            _, scores = model(x_128, mask_dim=None)
            all_scores.append(scores.cpu())

    # 4. Statistical Calculations
    scores_final = torch.cat(all_scores, dim=0).numpy() # Shape: (N, latent_dim)
    latent_std = np.std(scores_final, axis=0) + 1e-6
    latent_mean = np.mean(scores_final, axis=0)

    # 5. Update Checkpoint
    logging.info("Updating checkpoint with latent statistics...")
    checkpoint['latent_mean'] = latent_mean.astype(np.float32)
    checkpoint['latent_std'] = latent_std.astype(np.float32)
    torch.save(checkpoint, model_pth_path)

    logging.info("Metrics successfully added to checkpoint.")