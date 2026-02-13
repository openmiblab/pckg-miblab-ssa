import os
from typing import Callable, Tuple, List, Dict, Any
import numpy as np
from sklearn.decomposition import PCA
import logging
from tqdm import tqdm
import zarr
import dask.array as da
import pandas as pd
from dask_ml.decomposition import PCA as daskPCA




def pca_from_features(
    features_zarr_path: str, 
    output_zarr_path: str, 
    n_components=None
):
    """
    Standard Scikit-Learn PCA: Loads the entire feature matrix into RAM.
    Best for cases where N_samples * N_features * 4 bytes < Available RAM.
    """
    logging.info(f"PCA: Loading data from {os.path.basename(features_zarr_path)} into RAM...")

    # 1. Load Data from Zarr directly into NumPy
    feat_root = zarr.open(features_zarr_path, mode='r')
    data = feat_root['features'][:]  # The [:] loads the whole array into memory
    labels = feat_root['labels'][:]
    
    n_samples, n_features = data.shape
    logging.info(f"PCA: Matrix shape {n_samples}x{n_features}. Fitting Sklearn PCA...")

    # 2. Fit Scikit-Learn PCA
    # 'auto' or 'full' works best here since we are in-memory
    pca = PCA(n_components=n_components, svd_solver='auto')
    pca.fit(data)

    # 3. Prepare Output Zarr
    if not output_zarr_path.endswith('.zarr'):
        output_zarr_path += '.zarr'
    
    root = zarr.open_group(output_zarr_path, mode='w')
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)

    # 4. Save Results
    logging.info("PCA: Saving model attributes to Zarr...")

    # Save the basis (components)
    root.create_dataset('components', 
                        data=pca.components_.astype(np.float32), 
                        chunks=(1, n_features),
                        compressor=compressor)

    # Save metadata and model parameters
    root.create_dataset('mean', 
                        data=pca.mean_.astype(np.float32), 
                        compressor=compressor)
    
    root.create_dataset('variance', 
                        data=pca.explained_variance_.astype(np.float32))
    
    root.create_dataset('variance_ratio', 
                        data=pca.explained_variance_ratio_.astype(np.float32))
    
    root.create_dataset('labels', data=labels)

    # 5. Transfer Attributes (Essential for reconstruction)
    root.attrs['original_shape'] = feat_root.attrs.get('original_shape', None)
    root.attrs['kwargs'] = feat_root.attrs.get('kwargs')

    logging.info("PCA: Finished.")


def direct_pca_from_features(
    features_zarr_path: str, 
    output_zarr_path: str, 
    n_components=50,
    chunk_size=10000  # Number of features per chunk
):
    """
    Direct SVD implementation to bypass dask_ml shuffle overhead.
    Optimized for Wide Data (N_samples << N_features).
    """
    logging.info(f"PCA: Connecting to {os.path.basename(features_zarr_path)}...")

    # 1. Open Zarr safely
    feat_root = zarr.open(features_zarr_path, mode='r')
    features_ds = feat_root['features']
    labels = feat_root['labels'][:]
    
    # Rechunking: All samples (-1) for a slice of features.
    # This is the "Tall and Skinny" slice that dask.linalg.svd prefers.
    data = da.from_zarr(features_ds).rechunk({0: -1, 1: chunk_size})
    n_samples, n_features = data.shape
    k = n_components or 50

    # 2. Manual Centering & Variance calculation
    # We compute the mean and total variance first to calculate ratios later
    logging.info("PCA: Computing column means and total variance...")
    mean_dask = data.mean(axis=0)
    
    # We need the sum of individual feature variances for the ratio denominator
    # We use a map_blocks approach or standard var to keep it lazy
    total_var_dask = data.var(axis=0).sum()
    
    # Trigger compute for these small scalars/vectors
    mean_np, total_var = da.compute(mean_dask, total_var_dask)
    
    # 3. Running Compressed SVD
    logging.info(f"PCA: Running svd_compressed (k={k}) on {n_samples}x{n_features}...")
    # data_centered is a lazy operation
    data_centered = data - mean_np
    
    # u: Scores, s: Singular values, v: Components (Principal Axes)
    u, s, v = da.linalg.svd_compressed(data_centered, k=k, n_power_iter=3)

    # 4. Prepare Output Zarr
    if not output_zarr_path.endswith('.zarr'):
        output_zarr_path += '.zarr'
    
    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)

    # 5. Compute and Save Results
    # We compute s (singular values) first to derive variance metrics
    s_np = s.compute()
    
    # Eigenvalues (Explained Variance) = s^2 / (n-1)
    explained_variance = (s_np**2) / (n_samples - 1)
    explained_variance_ratio = explained_variance / total_var

    logging.info("PCA: Saving components (streaming dask -> zarr)...")
    # Stream the massive V matrix (components) directly to disk
    v.astype(np.float32).to_zarr(
        url=root.store, 
        component='components', 
        compressor=compressor,
        overwrite=True
    )

    logging.info("PCA: Saving metadata and variance metrics...")
    root.create_dataset('mean', data=mean_np.astype(np.float32), compressor=compressor)
    root.create_dataset('variance', data=explained_variance.astype(np.float32))
    root.create_dataset('variance_ratio', data=explained_variance_ratio.astype(np.float32))
    root.create_dataset('labels', data=labels)

    # 6. Transfer Attributes
    root.attrs['original_shape'] = feat_root.attrs.get('original_shape', None)
    root.attrs['kwargs'] = feat_root.attrs.get('kwargs')
    root.attrs['n_samples'] = n_samples
    root.attrs['n_features'] = n_features

    logging.info(f"PCA: Finished successfully.")


def dask_pca_from_features(
    features_zarr_path: str, 
    output_zarr_path: str, 
    n_components=None,
    chunk_size=100  # Number of samples per chunk
):
    """
    Dask PCA: Streams (N, F) features from Zarr. 
    Optimized for N_samples << N_features (Wide Data).

    Memory usage (bytes): max of 
    - n_samples x n_features x 4
    - chunksize x n_features x 8
    """
    logging.info(f"PCA: Connecting to {os.path.basename(features_zarr_path)}...")

    # 1. Open Zarr as a Dask Array (Lazy)
    feat_root = zarr.open(features_zarr_path, mode='r')
    features_ds = feat_root['features']
    labels = feat_root['labels'][:]
    
    # Rechunking: (chunk_size, all_features)
    # This ensures each worker gets a manageable batch of full wavelet vectors
    data = da.from_zarr(features_ds).rechunk({0: chunk_size, 1: -1})
    
    n_samples, n_features = data.shape

    # 2. Fit Dask PCA
    # svd_solver='full' is efficient when one dimension is significantly smaller
    logging.info(f"PCA: Fitting Dask PCA on {n_samples}x{n_features} matrix...")
    pca = daskPCA(n_components=n_components, svd_solver='full') 
    pca.fit(data)

    # 3. Prepare Output Zarr
    if not output_zarr_path.endswith('.zarr'):
        output_zarr_path += '.zarr'
    
    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)

    # 4. Save Results
    logging.info("PCA: Saving model attributes...")

    # A. Save the massive components matrix
    # If it's already a numpy array, we use create_dataset. 
    # If it's dask, we use to_zarr to save RAM.
    if hasattr(pca.components_, 'compute'):
        logging.info("Components are Dask-backed. Streaming to Zarr...")
        pca.components_.astype(np.float32).to_zarr(
            url=root.store, 
            component='components', 
            compressor=compressor,
            overwrite=True
        )
    else:
        logging.info("Components are NumPy-backed. Writing to Zarr...")
        root.create_dataset('components', 
                            data=pca.components_.astype(np.float32), 
                            chunks=(1, n_features),
                            compressor=compressor)
        
    # helper to handle both dask and numpy arrays safely
    # daskPCA can return either numpy arrays or dask arrays
    def safe_compute(arr):
        return arr.compute() if hasattr(arr, 'compute') else arr

    # B. Save smaller datasets safely
    root.create_dataset('mean', 
                        data=safe_compute(pca.mean_).astype(np.float32), 
                        compressor=compressor)
    
    root.create_dataset('variance', 
                        data=safe_compute(pca.explained_variance_).astype(np.float32))
    
    root.create_dataset('variance_ratio', 
                        data=safe_compute(pca.explained_variance_ratio_).astype(np.float32))
    
    root.create_dataset('labels', data=labels)

    # 5. Transfer Attributes
    root.attrs['original_shape'] = feat_root.attrs.get('original_shape', None)
    root.attrs['kwargs'] = feat_root.attrs.get('kwargs')

    logging.info(f"PCA: Finished.")



def scores_from_features(
    features_zarr_path, 
    pca_zarr_path, 
    output_zarr_path
):
    """
    Computes PCA scores entirely in memory using NumPy.
    Optimized for speed when RAM is sufficient.
    """
    logging.info(f"Loading data from {os.path.basename(features_zarr_path)} into RAM...")
    
    # 1. Load Input Data into RAM
    feat_root = zarr.open(features_zarr_path, mode='r')
    features = feat_root['features'][:] # (N, F)
    labels = feat_root['labels'][:]
    
    # 2. Load PCA Basis into RAM
    pca_root = zarr.open(pca_zarr_path, mode='r')
    mu = pca_root['mean'][:]               # (F,)
    eig_vecs = pca_root['components'][:]   # (K, F)
    var = pca_root['variance'][:]          # (K,)
    var_ratio = pca_root['variance_ratio'][:] 

    # 3. Compute Projections (NumPy Math)
    logging.info("Computing projections (Matrix Multiplication)...")
    
    # Centering
    centered = features - mu
    
    # Projection: (N, F) @ (F, K) -> (N, K)
    # Using .T on eig_vecs to get the correct shape for dot product
    scores = centered @ eig_vecs.T
    
    # Normalization: (N, K) / (K,)
    logging.info("Normalizing scores...")
    normalized_scores = scores / np.sqrt(var)

    # 4. Save to Zarr
    logging.info(f"Saving results to {output_zarr_path}...")
    root = zarr.open_group(output_zarr_path, mode='w')
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)

    # Save scores
    root.create_dataset('scores', 
                        data=scores.astype(np.float32), 
                        compressor=compressor)
    
    # Save normalized scores
    root.create_dataset('normalized_scores', 
                        data=normalized_scores.astype(np.float32), 
                        compressor=compressor)

    # Save metadata
    root.create_dataset('labels', data=labels)
    root.create_dataset('variance', data=var.astype(np.float32))
    root.create_dataset('variance_ratio', data=var_ratio.astype(np.float32))
    
    # Copy essential attributes for reconstruction
    root.attrs['original_shape'] = feat_root.attrs.get('original_shape')
    root.attrs['kwargs'] = feat_root.attrs.get('kwargs')

    logging.info("Finished PCA projection.")
    return scores, normalized_scores


def dask_scores_from_features(
    features_zarr_path, 
    pca_zarr_path, 
    output_zarr_path, 
):
    """
    Memory usage:

    n_features x (chunk_size + n_components) x 8
    """
    logging.info("Connecting to Zarr stores (Lazy)...")
    
    # 1. Open Zarr as Dask Arrays
    features = da.from_zarr(features_zarr_path, component='features')
    labels = da.from_zarr(features_zarr_path, component='labels')
    pca_root = zarr.open(pca_zarr_path, mode='r')
    
    # PCA attributes are small enough to compute/load once, 
    # EXCEPT for 'components' if they are 10M wide.
    # We keep components as a dask array to stream the dot product.
    mu = da.from_zarr(pca_zarr_path, component='mean')
    eig_vecs = da.from_zarr(pca_zarr_path, component='components') # (n_comp, n_features)
    var = da.from_zarr(pca_zarr_path, component='variance')
    var_ratio = pca_root['variance_ratio'][:] # Small enough for numpy

    # 2. Dask Math (Symbolic/Lazy)
    # This creates a task graph, it does NOT execute yet.
    logging.info("Building projection graph...")
    
    # Centering (Broadcasting mu over features)
    centered = features - mu
    
    # Projection: (N, F) @ (F, K) -> (N, K)
    # We use eig_vecs.T to align dimensions for the dot product
    scores = da.matmul(centered, eig_vecs.T)
    
    # Normalization by sqrt of variance
    normalized_scores = scores / da.sqrt(var)

    # 3. Prepare Output Zarr
    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)

    # 4. Execute and Save
    # We use .to_zarr() to trigger the computation chunk-by-chunk
    logging.info(f"Computing and streaming scores to {output_zarr_path}...")
    
    # Compute and save scores
    scores.astype(np.float32).to_zarr(
        url=root.store, 
        component='scores', 
        compressor=compressor,
        overwrite=True
    )
    
    # Compute and save normalized scores
    normalized_scores.astype(np.float32).to_zarr(
        url=root.store, 
        component='normalized_scores', 
        compressor=compressor,
        overwrite=True
    )

    # Save metadata (Small arrays)
    root.create_dataset('labels', data=labels.compute())
    root.create_dataset('variance', data=var.compute())
    root.create_dataset('variance_ratio', data=var_ratio)

    # Copy essential attributes for reconstruction
    feat_root = zarr.open(features_zarr_path, mode='r')
    root.attrs['original_shape'] = feat_root.attrs.get('original_shape')
    root.attrs['kwargs'] = feat_root.attrs.get('kwargs')
    
    logging.info("Finished PCA projection.")
    return True


def modes_from_pca(
    pca_zarr_path: str, 
    modes_zarr_path: str, 
    n_components=8, 
    n_coeffs=11, 
    max_coeff=2
):
    """
    Generates 3D shape modes by varying PCA coefficients and reconstructing masks.
    Output: 5D Zarr (Coeff_Steps, Mode_Index, Depth, Height, Width)

    Memory usage (1 mask + 1 feature vector) x overhead of mask_from_features
    """
    logging.info(f"Modes: Opening PCA store at {os.path.basename(pca_zarr_path)}...")
    
    # 1. Connect to PCA (Lazy)
    z_pca = zarr.open(pca_zarr_path, mode='r')
    
    # Check bounds of stored components
    n_features = z_pca['components'].shape[1]
    stored_k = z_pca['components'].shape[0]
    limit_k = min(n_components, stored_k)
    
    # Load Mean and Metadata and create coeffs
    avr = z_pca['mean'][:]
    sdev = np.sqrt(z_pca['variance'][:limit_k])
    coeffs_range = np.linspace(-max_coeff, max_coeff, n_coeffs)

    # 3. Setup 5D Output Store
    if not modes_zarr_path.endswith('.zarr'):
        modes_zarr_path += '.zarr'
        
    store = zarr.DirectoryStore(modes_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)

    z_modes = root.create_dataset(
        'features',
        shape=(n_coeffs, limit_k, n_features) ,
        chunks=(1, 1, n_features),
        dtype=np.float32,
        compressor=compressor,
    )

    # 4. Generate Modes (The "Safe" Loop)
    msg = f"Modes: Building {n_coeffs * limit_k} feature vectors..."
    logging.info(msg)
    for i in tqdm(range(limit_k), desc=msg):
    
        current_comp = z_pca['components'][i, :]
        for j in range(n_coeffs):
            feat_vec = avr + (coeffs_range[j] * sdev[i] * current_comp)
            z_modes[j, i, ...] = feat_vec

    # 5. Save attributes for the visualization tool
    root.create_dataset('labels', data=z_pca['labels'][:])
    root.attrs['coeffs'] = coeffs_range.tolist()
    root.attrs['kwargs'] = z_pca.attrs.get('kwargs')
    root.attrs['original_shape'] = z_pca.attrs.get('original_shape')
    
    logging.info(f"Modes: Successfully saved to {modes_zarr_path}")



def features_from_scores(
    pca_zarr_path, 
    scores_zarr_path,
    output_zarr_path,
    target_labels=None,
    n_components=None,
):
    z_pca = zarr.open(pca_zarr_path, mode='r')
    z_scores = zarr.open(scores_zarr_path, mode='r')
    
    # Use .get() or check keys to prevent KeyErrors
    avr = z_pca['mean'][:] 
    n_features = z_pca['components'].shape[1]
    
    # Optimization: Pre-load components once
    if n_components is None:
        eig_vecs = z_pca['components'][:]
    else:
        eig_vecs = z_pca['components'][:n_components]

    # Label indexing
    all_labels = z_scores['labels'][:].astype(str)
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    
    if target_labels is not None:
        valid_indices = [label_to_idx[str(l)] for l in target_labels if str(l) in label_to_idx]
        found_labels = [str(l) for l in target_labels if str(l) in label_to_idx]
    else:
        valid_indices = list(range(len(all_labels)))
        found_labels = all_labels

    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    
    z_recons = root.create_dataset(
        'features',
        shape=(len(valid_indices), n_features),
        chunks=(1, n_features),
        dtype='float32',
        compressor=zarr.Blosc(cname='zstd', clevel=3)
    )
    # Map to fixed-length string or object to avoid Zarr string errors
    root.array('labels', data=found_labels)

    for i, idx in enumerate(valid_indices):
        
        if n_components is None:
            score = z_scores['scores'][idx]
        else:
            score = z_scores['scores'][idx,:n_components]
            
        # Standard reconstruction: mean + (scores @ components)
        z_recons[i] = (avr + np.dot(score, eig_vecs)).astype(np.float32)

    # Copy essential attributes for reconstruction
    root.attrs['original_shape'] = z_scores.attrs.get('original_shape')
    root.attrs['kwargs'] = z_scores.attrs.get('kwargs')

    logging.info("Reconstruction finished.")


def cumulative_features_from_scores(
    pca_zarr_path: str, 
    scores_zarr_path: str,
    output_zarr_path: str,
    target_labels: list = None,
    max_components: int = 10,
    step_size: int = 1
):
    z_pca = zarr.open(pca_zarr_path, mode='r')
    z_scores = zarr.open(scores_zarr_path, mode='r')
    
    avr = z_pca['mean'][:].astype(np.float32) 
    n_features = z_pca['components'].shape[1]
    
    # Calculate checkpoint indices
    save_indices = np.arange(step_size - 1, max_components, step_size)
    
    # +1 for the Average Kidney at the start
    n_steps = len(save_indices) + 1 
    eig_vecs = z_pca['components'][:max_components].astype(np.float32)

    # 1. Handle Labels and Indices (The Filter)
    all_labels = z_scores['labels'][:].astype(str)
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    
    if target_labels is not None:
        valid_indices = [label_to_idx[str(l)] for l in target_labels if str(l) in label_to_idx]
        found_labels = [str(all_labels[idx]) for idx in valid_indices]
    else:
        valid_indices = list(range(len(all_labels)))
        found_labels = all_labels.tolist()

    n_samples = len(valid_indices)

    # 2. Setup Output Store
    root = zarr.open(output_zarr_path, mode='w')
    root.attrs.update(z_pca.attrs) 
    # Metadata: Step 0 is 'Average', subsequent are PCA checkpoints
    root.attrs['saved_steps'] = [0] + (save_indices + 1).tolist() 

    # --- Explicit Dataset Creation ---
    root.create_dataset('labels', data=np.array(found_labels, dtype='U'), overwrite=True)
    
    z_recons = root.create_dataset(
        'features',
        shape=(n_samples, n_steps, n_features),
        chunks=(1, 1, n_features),
        dtype='float32',
        compressor=zarr.Blosc(cname='zstd', clevel=3),
        overwrite=True
    )

    z_mse_vs_avg = root.create_dataset(
        'mse_vs_avg',
        shape=(n_samples, n_steps),
        dtype='float32',
        overwrite=True
    )

    # 4. Cumulative Loop
    logging.info(f"Reconstructing {n_samples} samples in {n_steps} steps (Step 0 = Average)...")

    for i, idx in tqdm(enumerate(valid_indices), total=n_samples):
        # --- Step 0: Save the Average ---
        z_recons[i, 0, :] = avr
        z_mse_vs_avg[i, 0] = 0.0 # MSE vs itself is 0
        
        scores = z_scores['scores'][idx, :max_components]
        current_reconstruction = avr.copy()
        
        # Start counter at 1 because index 0 is the Average
        step_counter = 1 
        for k in range(max_components):
            current_reconstruction += (scores[k] * eig_vecs[k])
            
            if k in save_indices:
                z_recons[i, step_counter, :] = current_reconstruction
                mse = np.mean((avr - current_reconstruction)**2)
                z_mse_vs_avg[i, step_counter] = mse
                step_counter += 1

    return True


def pca_performance(
    pca_zarr_path: str, 
    scores_zarr_path: str, 
    original_features_zarr_path: str, 
    marginal_mse_path: str,
    cumulative_mse_path: str,
    n_components: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    
    # 1. Open Zarr Stores
    try:
        z_pca = zarr.open(pca_zarr_path, mode='r')
        z_scores = zarr.open(scores_zarr_path, mode='r')
        z_orig = zarr.open(original_features_zarr_path, mode='r')
    except Exception as e:
        logging.error(f"Failed to open Zarr stores: {e}")
        raise

    # 2. Load Shared Basis and Metadata
    avr = z_pca['mean'][:]
    eig_vecs = z_pca['components'][:n_components, :]
    
    labels_scores = [str(l) for l in z_scores['labels'][:]]
    labels_orig = [str(l) for l in z_orig['labels'][:]]
    orig_label_to_idx = {l: i for i, l in enumerate(labels_orig)}
    n_samples = len(labels_scores)
    
    # 3. Initialize or Load Progress
    marginal_mse = np.zeros((n_samples, n_components), dtype=np.float32)
    cumulative_mse = np.zeros((n_samples, n_components), dtype=np.float32)

    # 4. Sequential Processing Loop
    logging.info(f"Starting PCA performance evaluation for {n_samples} subjects.")
    
    for i, label in enumerate(tqdm(labels_scores, desc="Total Progress")):
        # --- Skip Logic ---
        if label not in orig_label_to_idx:
            logging.warning(f"Label {label} not found in original masks. Skipping.")
            continue
            
        # --- Computation ---
        orig_idx = orig_label_to_idx[label]
        orig_features = z_orig['features'][orig_idx]
        subject_scores = z_scores['scores'][i, :n_components]
        orig_features_norm = np.linalg.norm(orig_features)
        
        current_cumulative_vec = avr.copy()
        
        for k in range(n_components):
            score_k = subject_scores[k]
            vec_k = eig_vecs[k, :]
            
            # Marginal Reconstruction
            marginal_feat_vec = avr + (score_k * vec_k)
            marginal_mse[i, k] = np.linalg.norm(marginal_feat_vec-orig_features)/orig_features_norm
            
            # Cumulative Reconstruction
            current_cumulative_vec += (score_k * vec_k)
            cumulative_mse[i, k] = np.linalg.norm(current_cumulative_vec-orig_features)/orig_features_norm
        
        # 5. Save Progress per Subject
        pd.DataFrame(marginal_mse, index=labels_scores).to_csv(marginal_mse_path)
        pd.DataFrame(cumulative_mse, index=labels_scores).to_csv(cumulative_mse_path)

    logging.info(f"Performance evaluation complete.")
    return marginal_mse, cumulative_mse