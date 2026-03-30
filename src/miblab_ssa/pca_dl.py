import torch
import torch.nn as nn
import zarr
import numpy as np
import dask.array as da
from torch.utils.data import Dataset, DataLoader, Subset
import logging
from tqdm import tqdm
import pandas as pd
from typing import Tuple
from sklearn.model_selection import KFold
import os


class OrderedAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        
        # Encoder: Gradual compression from 4094 down to latent_dim
        # Path: 4094 -> 1024 -> 512 -> 256 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, latent_dim)
        )
        
        # Decoder: Mirroring the Encoder
        # Path: latent_dim -> 256 -> 512 -> 1024 -> 4094
        self.decoder = nn.Sequential(
            # Note: No BatchNorm here to avoid issues with zero-masking
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Linear(1024, input_dim)
        )

    def forward(self, x, mask_dim=None):
        z = self.encoder(x)
        
        # Nested Dropout / Ordered Masking
        if self.training and mask_dim is not None:
            mask = torch.zeros_like(z)
            mask[:, :mask_dim] = 1.0
            z = z * mask
        
        # If not training, we assume we use the full latent space 
        # unless you explicitly pass a mask_dim during inference.
        recon = self.decoder(z)
        return recon, z
    


class ZarrStreamingDataset(Dataset):
    def __init__(self, zarr_path):
        self.root = zarr.open(zarr_path, mode='r')
        self.data = self.root['features']
        # For simplicity, we assume pre-calculated stats or calculate once lazily
        # Loading just the mean/std vectors (5000 elements) is very cheap
        self.mean = np.mean(self.data, axis=0).astype(np.float32)
        self.std = np.std(self.data, axis=0).astype(np.float32) + 1e-6

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx].astype(np.float32)
        normalized = (sample - self.mean) / self.std
        return torch.from_numpy(normalized)
    
def deep_pca_from_features(
    features_zarr_path: str = None, 
    model_pth_path: str = None,
    n_components: int = 25, 
    epochs: int = 100, 
    batch_size: int = 64,
):
    # 1. Setup Lazy Loading
    logging.info(f"Connecting to {features_zarr_path}...")
    dataset = ZarrStreamingDataset(features_zarr_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Initialize Model
    n_features = dataset.data.shape[1]
    model = OrderedAutoencoder(n_features, n_components)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 3. Training Loop
    logging.info(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x in loader:
            k = np.random.randint(1, n_components + 1)
            recon, _ = model(x, mask_dim=k)
            loss = criterion(recon, x) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch+1} Loss: {epoch_loss/len(loader):.6f}")
    
    # Save Model Checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'train_mean': dataset.mean,
        'train_std': dataset.std,
        'input_dim': n_features,
        'latent_dim': n_components,
        'original_shape': dataset.root.attrs.get('original_shape'),
        'kwargs': dataset.root.attrs.get('kwargs'),
    }
    torch.save(checkpoint, model_pth_path)



def deep_cv_pca_from_features(
    features_zarr_path=None, 
    model_pth_path=None,
    n_components=25, 
    epochs=100, 
    batch_size=64,
    n_folds=5
):
    # 1. Setup Data
    logging.info(f"Connecting to {features_zarr_path}...")
    full_dataset = ZarrStreamingDataset(features_zarr_path)
    n_features = full_dataset.data.shape[1]
    n_samples = len(full_dataset)
    
    # Initialize KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # We will track the best model across all folds or aggregate metrics
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(n_samples))):
        logging.info(f"--- Starting Fold {fold + 1}/{n_folds} ---")
        
        # Create Subsets
        train_sub = Subset(full_dataset, train_idx)
        val_sub = Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)

        # 2. Initialize Model for this fold
        model = OrderedAutoencoder(n_features, n_components)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # 3. Training Loop
        for epoch in range(epochs):
            # --- Training Phase ---
            model.train()
            train_loss = 0
            for x in train_loader:
                k = np.random.randint(1, n_components + 1)
                recon, _ = model(x, mask_dim=k)
                loss = criterion(recon, x)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # --- Validation Phase ---
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_val in val_loader:
                    # Validate on the full bottleneck (no mask) 
                    # or a random k to match training behavior
                    recon_val, _ = model(x_val) 
                    v_loss = criterion(recon_val, x_val)
                    val_loss += v_loss.item()

            if (epoch + 1) % 10 == 0:
                avg_train = train_loss / len(train_loader)
                avg_val = val_loss / len(val_loader)
                logging.info(f"Fold {fold+1} | Epoch {epoch+1} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

        fold_results.append(val_loss / len(val_loader))

    # HERE: add a setup to find the correct number of epochs from the cv results
    n_epochs = 100

    # Then run again on the whole dataset
    deep_pca_from_features(
        features_zarr_path, 
        model_pth_path,
        n_components, 
        epochs = n_epochs, 
        batch_size = batch_size,
    )

    # # 4. Final Training (on all data) for the production model
    # logging.info(f"Cross-validation complete. Average Val Loss: {np.mean(fold_results):.6f}")
    # logging.info("Training final production model on full dataset...")
    
    # final_model = OrderedAutoencoder(n_features, n_components)
    # final_optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-3)
    # full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    
    # for epoch in range(epochs):
    #     final_model.train()
    #     for x in full_loader:
    #         k = np.random.randint(1, n_components + 1)
    #         recon, _ = final_model(x, mask_dim=k)
    #         loss = criterion(recon, x)
    #         final_optimizer.zero_grad()
    #         loss.backward()
    #         final_optimizer.step()
    
    # checkpoint = {
    #     'model_state_dict': final_model.state_dict(),
    #     'train_mean': full_dataset.mean,
    #     'train_std': full_dataset.std,
    #     'input_dim': n_features,
    #     'latent_dim': n_components,
    #     'cv_val_losses': fold_results  # Helpful for reporting
    # }
    # torch.save(checkpoint, model_pth_path)



def add_deep_pca_metrics(
    features_zarr_path=None, 
    model_pth_path=None,
    batch_size=64
):
    # 1. Setup Lazy Loading
    logging.info(f"Connecting to {features_zarr_path}...")
    dataset = ZarrStreamingDataset(features_zarr_path)

    checkpoint = torch.load(model_pth_path, weights_only=False)
    input_dim = checkpoint['input_dim']
    latent_dim = checkpoint['latent_dim']
    
    model = OrderedAutoencoder(input_dim, latent_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 4. Post-Training: Metrics Calculation
    model.eval()
    all_scores = []
    
    with torch.no_grad():
        eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for x in eval_loader:
            _, scores = model(x)
            all_scores.append(scores)

    # --- Variance Calculations ---
    scores_final = torch.cat(all_scores, dim=0).numpy()
    latent_std = np.std(scores_final, axis=0) + 1e-6
    latent_mean = np.mean(scores_final, axis=0)
    # explained_variance = latent_std ** 2
    # explained_variance_ratio = explained_variance / np.sum(explained_variance)

    checkpoint['latent_std'] = latent_std.astype(np.float32)
    checkpoint['latent_mean'] = latent_mean.astype(np.float32)
    torch.save(checkpoint, model_pth_path)
    


def deep_scores_from_features(
    features_zarr_path: str = None, 
    model_pth_path: str = None, 
    scores_csv_path: str = None,
    normalized_scores_csv_path: str = None, 
    chunk_size = 100
):
    """
    Passes unseen features through the trained Encoder and saves 
    'scores' and 'normalized_scores' as individual CSV files.
    """
    logging.info("Loading model and connecting to Zarr...")

    try:
        # 1. Load the trained model and parameters
        checkpoint = torch.load(model_pth_path, weights_only=False)
        input_dim = checkpoint['input_dim']
        latent_dim = checkpoint['latent_dim']

        model = OrderedAutoencoder(input_dim, latent_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # 2. Open Unseen Data
        feat_gt = zarr.open(features_zarr_path, mode='r')
        features_da = da.from_zarr(feat_gt['features']).rechunk({0: chunk_size, 1: -1})
        labels = feat_gt['labels'][:]
        
        train_mean = torch.from_numpy(checkpoint['train_mean']).float()
        train_std = torch.from_numpy(checkpoint['train_std']).float()
        latent_mean = torch.from_numpy(checkpoint['latent_mean']).float()
        latent_std = torch.from_numpy(checkpoint['latent_std']).float()

    except Exception as e:
        logging.error(f"Failed to initialize: {e}")
        raise

    n_samples = features_da.shape[0]
    all_scores = []
    all_normalized_scores = []

    # 3. Stream Inference
    logging.info("Streaming features through Encoder...")
    
    with torch.no_grad():
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            
            chunk_np = features_da[start:end].compute()
            x = torch.from_numpy(chunk_np).float()
            
            # Input Normalization
            x_normalized = (x - train_mean) / train_std
            
            # Forward pass
            z = model.encoder(x_normalized)
            z_normalized = (z - latent_mean) / latent_std
            
            all_scores.append(z.numpy())
            all_normalized_scores.append(z_normalized.numpy())

    # 4. Concatenate and Save to CSV
    # Create column names for latent dimensions (e.g., PC1, PC2...)
    col_names = [f"PC_{i+1}" for i in range(latent_dim)]
    
    # Process Raw Scores
    df_scores = pd.DataFrame(np.vstack(all_scores), columns=col_names)
    df_scores.insert(0, 'label', labels) # Add labels as first column
    
    # Process Normalized Scores
    df_norm_scores = pd.DataFrame(np.vstack(all_normalized_scores), columns=col_names)
    df_norm_scores.insert(0, 'label', labels)

    # Save to csv 
    df_scores.to_csv(scores_csv_path, index=False)
    df_norm_scores.to_csv(normalized_scores_csv_path, index=False)

    logging.info(f"Finished computing scores. Files saved to {os.path.dirname(scores_csv_path)} and {os.path.dirname(normalized_scores_csv_path)}")
    return True


def deep_pca_performance(
    model_pth_path: str = None, 
    scores_csv_path: str = None, 
    gt_features_zarr_path: str = None, 
    marginal_mse_csv_path: str = None,
    cumulative_mse_csv_path: str = None,
    n_components: int = 25,
) -> Tuple[np.ndarray, np.ndarray]:
    
    # 1. Load Model and Weights
    try:
        checkpoint = torch.load(model_pth_path, map_location='cpu', weights_only=False)
        input_dim = checkpoint['input_dim']
        latent_dim = checkpoint['latent_dim']
        
        train_mean = checkpoint['train_mean'].astype(np.float32)
        train_std = checkpoint['train_std'].astype(np.float32)
        
        model = OrderedAutoencoder(input_dim, latent_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Ground truth remains in Zarr
        feat_gt = zarr.open(gt_features_zarr_path, mode='r')
    except Exception as e:
        logging.error(f"Failed to initialize: {e}")
        raise

    # 2. Load Scores CSV and Prepare Metadata
    df_scores = pd.read_csv(scores_csv_path)

    pc_cols = [c for c in df_scores.columns if c.startswith('PC')][:n_components]
    labels_scores = df_scores['label'].astype(str).tolist()
    labels_orig = [str(l) for l in feat_gt['labels'][:]]
    orig_label_to_idx = {l: i for i, l in enumerate(labels_orig)}
    scores_matrix = df_scores.iloc[:, 1:].values.astype('float32')
    
    n_samples = len(labels_scores)
    cumulative_mse = np.zeros((n_samples, n_components), dtype=np.float32)
    marginal_mse = np.zeros((n_samples, n_components), dtype=np.float32)

    train_mean_norm = np.linalg.norm(train_mean)

    logging.info(f"Starting Deep PCA performance evaluation for {n_samples} samples.")
    
    # 3. Evaluation Loop
    with torch.no_grad():
        for i, label in enumerate(tqdm(labels_scores, desc="Subjects")):
            if label not in orig_label_to_idx:
                logging.warning(f"Label {label} missing in ground truth Zarr. Skipping.")
                continue
                
            orig_idx = orig_label_to_idx[label]
            orig_features = feat_gt['features'][orig_idx]
            
            # Extract scores for this subject
            subject_scores = scores_matrix[i, :]
            scores_tensor = torch.from_numpy(subject_scores).float().unsqueeze(0)
             
            for k_idx in range(n_components):
                # --- Marginal Reconstruction (Effect of exactly the k-th component) ---
                mask_m = torch.zeros_like(scores_tensor)
                mask_m[0, k_idx] = 1.0
                recon_norm_m = model.decoder(scores_tensor * mask_m).numpy().squeeze()
                recon_raw_m = (recon_norm_m * train_std) + train_mean
                marginal_mse[i, k_idx] = np.linalg.norm(recon_raw_m - orig_features) / train_mean_norm

                # --- Cumulative Reconstruction (Effect of components 0 through k) ---
                mask_c = torch.zeros_like(scores_tensor)
                mask_c[0, :k_idx + 1] = 1.0  # +1 to include the k-th component
                recon_norm_c = model.decoder(scores_tensor * mask_c).numpy().squeeze()
                recon_raw_c = (recon_norm_c * train_std) + train_mean
                cumulative_mse[i, k_idx] = np.linalg.norm(recon_raw_c - orig_features) / train_mean_norm
        
        # 4. Save Results
        pd.DataFrame(cumulative_mse, index=labels_scores, columns=pc_cols).to_csv(cumulative_mse_csv_path)
        pd.DataFrame(marginal_mse, index=labels_scores, columns=pc_cols).to_csv(marginal_mse_csv_path)

    logging.info("Performance evaluation complete.")
    return cumulative_mse, marginal_mse



def deep_cumulative_features_from_scores(
    model_pth_path: str = None, 
    scores_csv_path: str = None,
    gt_features_zarr_path: str = None,
    output_zarr_path: str = None,
    target_labels: list = None,
    step_size: int = 1,
    max_components: int = 10, 
):
    """
    Cumulative reconstruction from CSV scores with target label filtering.
    Output Zarr shape: (n_samples, n_steps, n_features)
    Steps: [Decoded Mean, Step_1, ..., Step_N, Ground Truth]
    """
    # 1. Load Model and Normalization Constants
    checkpoint = torch.load(model_pth_path, map_location='cpu', weights_only=False)
    n_features = checkpoint['input_dim']
    latent_dim = checkpoint['latent_dim']
    
    # Ensure these are tensors for easy math later
    train_mean = torch.from_numpy(checkpoint['train_mean']).float()
    train_std = torch.from_numpy(checkpoint['train_std']).float()
    latent_mean = torch.from_numpy(checkpoint['latent_mean']).float().view(1, -1)
    
    model = OrderedAutoencoder(n_features, latent_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    feat_gt = zarr.open(gt_features_zarr_path, mode='r')

    # 2. Load and Filter CSV Data
    df = pd.read_csv(scores_csv_path)
    label_col = df.columns[0]
    # Identify score columns explicitly
    pc_cols = [c for c in df.columns if 'PC' in c or 'score' in c.lower()][:latent_dim]

    # Sync with GT Zarr labels
    labels_orig_map = {str(l): i for i, l in enumerate(feat_gt['labels'][:])}

    # Filter for target labels AND existence in GT Zarr
    if target_labels is not None:
        # Filter dataframe to only include target_labels
        target_str = [str(l) for l in target_labels]
        df = df[df[label_col].astype(str).isin(target_str)]

    df = df[df[label_col].astype(str).isin(labels_orig_map.keys())]

    if df.empty:
        logging.error("No matching labels found in CSV. check target_labels.")
        return False

    found_labels = df[label_col].astype(str).tolist()
    scores_matrix = df[pc_cols].values[:, :latent_dim].astype('float32')

    n_samples = len(found_labels)
    save_indices = np.arange(step_size - 1, max_components, step_size)
    n_steps = len(save_indices) + 2 
    step_names = ["Mean"] + (save_indices + 1).tolist() + ["GT"]

    # 3. Setup Output Store
    root = zarr.open(output_zarr_path, mode='w')
    root.create_dataset('labels', data=np.array(found_labels, dtype='U'), overwrite=True)
    root.attrs['saved_steps'] = step_names
    root.attrs['original_shape'] = checkpoint['original_shape']
    root.attrs['kwargs'] = checkpoint['kwargs']
    
    z_recons = root.create_dataset(
        'features',
        shape=(n_samples, n_steps, n_features),
        chunks=(1, 1, n_features),
        dtype='float32',
        compressor=zarr.Blosc(cname='zstd', clevel=3)
    )

    # 4. Pre-compute decoded mean
    with torch.no_grad():
        mean_recon = model.decoder(latent_mean)
        decoded_mean = (mean_recon * train_std + train_mean).numpy().squeeze()

    # 5. Loop
    logging.info(f"Reconstructing {n_samples} filtered samples.")

    with torch.no_grad():
        for i, label in enumerate(tqdm(found_labels, desc="Cumulative Steps")):
            # Step 0: Decoded Mean
            z_recons[i, 0, :] = decoded_mean
            
            # Step -1: Ground Truth
            gt_idx = labels_orig_map[label]
            z_recons[i, -1, :] = feat_gt['features'][gt_idx]
            
            # Intermediate Steps
            scores_tensor = torch.from_numpy(scores_matrix[i, :]).float().unsqueeze(0)
            
            step_counter = 1 
            for k in save_indices:
                mask = torch.zeros_like(scores_tensor)
                mask[0, :k + 1] = 1.0
                
                recon_norm = model.decoder(scores_tensor * mask)
                recon_raw = (recon_norm * train_std + train_mean).numpy().squeeze()
                
                z_recons[i, step_counter, :] = recon_raw
                step_counter += 1

    logging.info("Deep cumulative reconstruction finished.")
    return True



def deep_features_from_scores(
    model_pth_path: str = None,
    scores_csv_path: str = None,
    features_zarr_path: str = None,
    n_components: int = None,
    chunk_size: int = 10
):
    """
    Reconstruct features using all absolute scores from a CSV file.
    Assumes first column = labels, subsequent columns = latent scores.
    """
    # 1. Load Model and Metadata
    checkpoint = torch.load(model_pth_path, weights_only=False)
    model = OrderedAutoencoder(checkpoint['input_dim'], checkpoint['latent_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load training stats for inverse scaling
    train_mu = torch.from_numpy(checkpoint['train_mean']).float()
    train_std = torch.from_numpy(checkpoint['train_std']).float()
    n_features = checkpoint['input_dim']

    # 2. Load CSV Data
    df = pd.read_csv(scores_csv_path)
    label_col = df.columns[0]
    
    found_labels = df[label_col].astype(str).tolist()
    # Scores are everything except the first column
    scores_matrix = df.iloc[:, 1:].values.astype('float32')

    # 3. Setup Output Store (Zarr)
    store = zarr.DirectoryStore(features_zarr_path)
    root = zarr.group(store=store, overwrite=True)

    z_recons = root.create_dataset(
        'features',
        shape=(len(found_labels), n_features),
        chunks=(1, n_features),
        dtype='float32',
        compressor=zarr.Blosc(cname='zstd', clevel=3)
    )
    root.create_dataset('labels', data=found_labels)

    # Copy metadata
    root.attrs['original_shape'] = checkpoint['original_shape']
    root.attrs['kwargs'] = checkpoint['kwargs']

    # 4. Reconstruction Loop
    logging.info(f"Reconstructing {len(found_labels)} samples from CSV...")

    with torch.no_grad():
        for i in range(len(found_labels)):
            # A. Get latent score
            score_tensor = torch.from_numpy(scores_matrix[i]).float().unsqueeze(0)

            # B. Optional Low-Rank Approximation
            if n_components is not None:
                mask = torch.zeros_like(score_tensor)
                mask[:, :n_components] = 1.0
                score_tensor = score_tensor * mask

            # C. Decode and Inverse Scale
            recon_features = model.decoder(score_tensor)
            feat_vec = (recon_features * train_std) + train_mu
            
            # D. Save result
            z_recons[i] = feat_vec.squeeze().numpy()

            if (i + 1) % chunk_size == 0:
                logging.info(f"Progress: {i+1}/{len(found_labels)} completed.")

    logging.info("Deep reconstruction finished.")
    return True



def deep_modes_from_pca( 
    model_pth_path: str = None, 
    modes_zarr_path: str = None, 
    n_components: int = 8, 
    n_coeffs: int = 11, 
    max_coeff: float = 5  # Typically 2 or 3 standard deviations
):
    """
    Generates 3D shape modes by traversing the latent space of the Autoencoder.
    Output: 5D Zarr (Coeff_Steps, Mode_Index, Depth, Height, Width)
    """
    # 1. Load Model and Stats
    checkpoint = torch.load(model_pth_path, map_location='cpu', weights_only=False)
    n_features= checkpoint['input_dim']
    latent_dim = checkpoint['latent_dim']

    model = OrderedAutoencoder(n_features, latent_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Feature scaling parameters
    train_mu = torch.from_numpy(checkpoint['train_mean']).float()
    train_std = torch.from_numpy(checkpoint['train_std']).float()
    latent_sdev = torch.from_numpy(checkpoint['latent_std']).float()
    latent_mean = torch.from_numpy(checkpoint['latent_mean']).float()

    limit_k = min(n_components, checkpoint['latent_dim'])
    coeffs_range = np.linspace(-max_coeff, max_coeff, n_coeffs)

    # 3. Setup Output Store
    store = zarr.DirectoryStore(modes_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    
    z_modes = root.create_dataset(
        'features',
        shape=(n_coeffs, limit_k, n_features),
        chunks=(1, 1, n_features),
        dtype='float32',
        compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    )
    
    # 4. Generate Modes
    logging.info(f"Deep Modes: Traversing {limit_k} latent dimensions...")

    with torch.no_grad():
        for i in tqdm(range(limit_k), desc="Processing Modes"):
            for j in range(n_coeffs):
                # Start with a "Zero" latent vector (the mean shape)
                # z = torch.zeros((1, checkpoint['latent_dim']))
                # Start with the global latent mean
                z = latent_mean.clone().unsqueeze(0)
                
                # Vary only the i-th component
                # Formula: Score = step_coefficient * standard_deviation_of_component + mean_of_component
                z[0, i] = (coeffs_range[j] * latent_sdev[i]) + latent_mean[i]

                # Decode and unnormalize
                recon_normalized = model.decoder(z)
                feat_vec = (recon_normalized * train_std) + train_mu

                z_modes[j, i, :] = feat_vec.cpu().numpy().flatten()

    # Copy essential attributes for reconstruction
    root.attrs['coeffs'] = coeffs_range.tolist()
    root.attrs['original_shape'] = checkpoint['original_shape']
    root.attrs['kwargs'] = checkpoint['kwargs']

    logging.info(f"Deep Modes: Successfully saved to {modes_zarr_path}")