import torch
import torch.nn as nn
import zarr
import numpy as np
import dask.array as da
from torch.utils.data import DataLoader, TensorDataset
import logging
from tqdm import tqdm

class OrderedAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Encoder: Compressing polynomial features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        # Decoder: Reconstructing the Signed Distance Transform features
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x, mask_dim=None):
        z = self.encoder(x)
        # To force ordering, we can zero out dimensions during training
        # This is a 'Nested Dropout' approach
        if self.training and mask_dim is not None:
            mask = torch.zeros_like(z)
            mask[:, :mask_dim] = 1.0
            z = z * mask
        
        return self.decoder(z), z


def deep_pca_from_features_zarr(
    features_zarr_path: str, 
    output_zarr_path: str,
    model_save_path: str,
    n_components=10, 
    epochs=100,
    batch_size=64
):
    # 1. Load and Preprocess Data
    logging.info(f"Loading features from {features_zarr_path}...")
    feat_root = zarr.open(features_zarr_path, mode='r')
    data_np = feat_root['features'][:].astype(np.float32)
    
    # Standardize Features (Crucial for Neural Networks)
    train_mean = np.mean(data_np, axis=0)
    train_std = np.std(data_np, axis=0) + 1e-6
    data_normalized = (data_np - train_mean) / train_std
    
    data_tensor = torch.from_numpy(data_normalized).float()
    n_samples, n_features = data_tensor.shape
    
    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Initialize Model
    model = OrderedAutoencoder(n_features, n_components)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 3. Training Loop with Nested Dropout
    logging.info(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in loader:
            x = batch[0]
            # Randomly select a sub-dimension to focus the 'energy' on early components
            # This forces the network to make the first components the most important
            k = np.random.randint(1, n_components + 1)
            
            recon, _ = model(x, mask_dim=k)
            loss = criterion(recon, x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(loader):.6f}")

    # 4. Post-Training: Calculate Metrics and Latent Stats
    model.eval()
    with torch.no_grad():
        # Get all scores (latent coordinates)
        full_recon, scores = model(data_tensor)
        
        # Calculate Variance per component (for Normalization in scoring)
        latent_std = torch.std(scores, dim=0).numpy()
        
        # Calculate Explained Variance equivalent
        total_var = torch.var(data_tensor).item()
        variances = []
        for k in range(1, n_components + 1):
            # Reconstruct using only top K components
            mask = torch.zeros_like(scores)
            mask[:, :k] = 1.0
            z_limited = scores * mask
            recon_k = model.decoder(z_limited)
            
            mse_k = criterion(recon_k, data_tensor).item()
            # Ratio of variance explained by the first k components
            variances.append(1 - (mse_k / total_var))

    exp_var_ratio = np.diff([0] + variances)

    # 5. Save Model and Metadata (for Reconstruction/Scoring)
    logging.info(f"Saving model to {model_save_path}...")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'train_mean': train_mean,
        'train_std': train_std,
        'latent_std': latent_std,
        'input_dim': n_features,
        'latent_dim': n_components,
        'original_shape': feat_root.attrs.get('original_shape'),
        'kwargs': feat_root.attrs.get('kwargs')
    }
    torch.save(checkpoint, model_save_path)

    # 6. Save Scores and Ratios to Zarr
    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    root.create_dataset('variance_ratio', data=exp_var_ratio.astype(np.float32))
    root.create_dataset('scores', data=scores.numpy().astype(np.float32))
    root.create_dataset('labels', data=feat_root['labels'][:])
    
    # Transfer attributes
    root.attrs.update(feat_root.attrs)

    logging.info("Deep PCA Training Complete.")
    return exp_var_ratio




def deep_scores_from_features_zarr(
    features_zarr_path, 
    model_pth_path, 
    output_zarr_path, 
    chunk_size=100
):
    """
    Inference version of the Non-Linear PCA. 
    Passes unseen features through the trained Encoder to get 'scores'.
    """
    logging.info("Loading model and connecting to Zarr...")

    # 1. Load the trained model architecture and weights
    # We assume the model dimensions were saved in the pth or known
    checkpoint = torch.load(model_pth_path)
    input_dim = checkpoint['input_dim']
    latent_dim = checkpoint['latent_dim']
    
    model = OrderedAutoencoder(input_dim, latent_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Open Unseen Data
    feat_root = zarr.open(features_zarr_path, mode='r')
    features_da = da.from_zarr(feat_root['features']).rechunk({0: chunk_size, 1: -1})
    labels = feat_root['labels'][:]
    
    # Load scaling parameters (mean/std) calculated during training
    # Deep models perform best when input features are z-scored
    train_mu = torch.from_numpy(checkpoint['train_mean']).float()
    train_std = torch.from_numpy(checkpoint['train_std']).float()
    latent_std = torch.from_numpy(checkpoint['latent_std']).float()

    # 3. Prepare Output Zarr
    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    
    # Initialize datasets for the scores
    n_samples = features_da.shape[0]
    scores_ds = root.create_dataset('scores', 
                                    shape=(n_samples, latent_dim), 
                                    chunks=(chunk_size, latent_dim), 
                                    dtype='f4', compressor=compressor)
    
    norm_scores_ds = root.create_dataset('normalized_scores', 
                                         shape=(n_samples, latent_dim), 
                                         chunks=(chunk_size, latent_dim), 
                                         dtype='f4', compressor=compressor)

    # 4. Stream Inference through the Encoder
    logging.info("Streaming features through Encoder...")
    
    with torch.no_grad():
        # Process in chunks to stay within memory limits
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            
            # Get chunk and convert to torch
            chunk_np = features_da[start:end].compute()
            x = torch.from_numpy(chunk_np).float()
            
            # Input Normalization (Standard for Deep Learning)
            x_normalized = (x - train_mu) / (train_std + 1e-6)
            
            # Pass through Encoder only to get latent scores
            z = model.encoder(x_normalized)
            
            # Calculate Normalized Scores (equivalent to PCA scores / sqrt(var))
            z_norm = z / (latent_std + 1e-6)
            
            # Save back to Zarr
            scores_ds[start:end] = z.numpy()
            norm_scores_ds[start:end] = z_norm.numpy()

    root.create_dataset('labels', data=labels)
    logging.info(f"Finished. Scores saved to {output_zarr_path}")
    return True



def deep_masks_from_scores_zarr(
    mask_from_features,  # Your callback to invert polynomial -> 3D mask
    model_pth_path: str,
    scores_zarr_path: str,
    output_zarr_path: str,
    target_labels: list = None,
    num_components: int = None, # Equivalent to 'components' list in PCA
    chunk_size: int = 10
):
    """
    Non-linear reconstruction:
    Scores -> Decoder -> Polynomial Features -> 3D Mask
    """
    # 1. Load Model and Metadata
    checkpoint = torch.load(model_pth_path)
    model = OrderedAutoencoder(checkpoint['input_dim'], checkpoint['latent_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load training stats for inverse scaling
    train_mu = torch.from_numpy(checkpoint['train_mean']).float()
    train_std = torch.from_numpy(checkpoint['train_std']).float()

    z_scores = zarr.open(scores_zarr_path, mode='r')
    shape = tuple(checkpoint['original_shape'])
    kwargs = checkpoint.get('kwargs', {})

    # Handle labels and indexing
    all_labels = [str(l) for l in z_scores['labels'][:]]
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    
    if target_labels is not None:
        valid_indices = [label_to_idx[l] for l in target_labels if l in label_to_idx]
        found_labels = [l for l in target_labels if l in label_to_idx]
    else:
        valid_indices = list(range(len(all_labels)))
        found_labels = all_labels

    # 2. Setup Output Store
    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    z_recons = root.create_dataset(
        'reconstructed_masks',
        shape=(len(valid_indices),) + shape,
        chunks=(1,) + shape,
        dtype=bool,
        compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    )
    root.create_dataset('labels', data=found_labels)

    # 3. Reconstruction Loop
    logging.info(f"Reconstructing {len(valid_indices)} masks using {num_components or 'all'} components...")

    with torch.no_grad():
        for i, idx in enumerate(valid_indices):
            # A. Get the latent score
            score_np = z_scores['scores'][idx, :]
            score_tensor = torch.from_numpy(score_np).float().unsqueeze(0) # (1, latent_dim)

            # B. Apply "Low-Rank" Approximation
            # We zero out all components beyond the requested number
            if num_components is not None:
                mask = torch.zeros_like(score_tensor)
                mask[:, :num_components] = 1.0
                score_tensor = score_tensor * mask

            # C. Pass through Decoder (Non-linear projection)
            # The decoder outputs normalized features
            recon_features = model.decoder(score_tensor)

            # D. Inverse Scaling (De-normalize to original polynomial scale)
            # feat_vec = (normalized * std) + mu
            feat_vec = (recon_features * train_std) + train_mu
            feat_vec_np = feat_vec.squeeze().numpy()

            # E. Generate 3D volume via your SDT-inverse function
            mask_3d = mask_from_features(feat_vec_np, shape, **kwargs)
            z_recons[i, ...] = mask_3d.astype(bool)

            if (i + 1) % chunk_size == 0:
                logging.info(f"Progress: {i+1}/{len(valid_indices)} completed.")

    logging.info("Deep reconstruction finished.")
    return True



def deep_modes_from_pca_zarr(
    mask_from_features, 
    model_pth_path: str, 
    modes_zarr_path: str, 
    n_components=8, 
    n_coeffs=11, 
    max_coeff=3  # Typically 2 or 3 standard deviations
):
    """
    Generates 3D shape modes by traversing the latent space of the Autoencoder.
    Output: 5D Zarr (Coeff_Steps, Mode_Index, Depth, Height, Width)
    """
    # 1. Load Model and Stats
    checkpoint = torch.load(model_pth_path)
    model = OrderedAutoencoder(checkpoint['input_dim'], checkpoint['latent_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Feature scaling parameters
    train_mu = torch.from_numpy(checkpoint['train_mean']).float()
    train_std = torch.from_numpy(checkpoint['train_std']).float()
    
    # Latent scaling (Standard Deviation of the scores)
    # This is equivalent to 'sdev' in your linear PCA version
    latent_sdev = torch.from_numpy(checkpoint['latent_std']).float()

    shape = tuple(checkpoint['original_shape'])
    kwargs = checkpoint.get('kwargs', {})
    limit_k = min(n_components, checkpoint['latent_dim'])

    # 2. Setup Step Coefficients (e.g., -3 sigma to +3 sigma)
    coeffs_range = np.linspace(-max_coeff, max_coeff, n_coeffs)

    # 3. Setup 5D Output Store
    store = zarr.DirectoryStore(modes_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    
    out_shape = (n_coeffs, limit_k) + shape
    chunks = (1, 1) + shape 
    
    z_modes = root.create_dataset(
        'modes',
        shape=out_shape,
        chunks=chunks,
        dtype=bool,
        compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    )
    
    root.attrs['coeffs'] = coeffs_range.tolist()
    root.attrs['n_components'] = limit_k

    # 4. Generate Modes
    logging.info(f"Deep Modes: Traversing {limit_k} latent dimensions...")

    with torch.no_grad():
        for i in tqdm(range(limit_k), desc="Processing Modes"):
            for j in range(n_coeffs):
                # Start with a "Zero" latent vector (the mean shape)
                z = torch.zeros((1, checkpoint['latent_dim']))
                
                # Vary only the i-th component
                # Formula: Score = step_coefficient * standard_deviation_of_component
                z[0, i] = coeffs_range[j] * latent_sdev[i]

                # Decode to normalized polynomial features
                recon_normalized = model.decoder(z)

                # Inverse scale to original polynomial units
                feat_vec = (recon_normalized * train_std) + train_mu
                feat_vec_np = feat_vec.squeeze().numpy()

                # Reconstruct the 3D volume
                mask_3d = mask_from_features(feat_vec_np, shape, **kwargs)
                
                z_modes[j, i, ...] = mask_3d.astype(bool)

    logging.info(f"Deep Modes: Successfully saved to {modes_zarr_path}")
    return True