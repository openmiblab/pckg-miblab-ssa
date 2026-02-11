import os
from typing import Callable, Tuple, List, Dict, Any
import numpy as np
from sklearn.decomposition import PCA
import logging
import dask
from dask.diagnostics import ProgressBar
from itertools import product
from tqdm import tqdm
import zarr
import dask.array as da
import dask.delayed
import pandas as pd
from dask_ml.decomposition import PCA as daskPCA


from miblab_ssa.metrics import dice_coefficient



def features_from_dataset_in_npz(
    features_from_mask:Callable,
    masks:list, 
    filepath:str, 
    labels:list,
    **kwargs, # kwargs for features_from_mask
):

    logging.info("Features: scheduling tasks..")
    tasks = [
        dask.delayed(features_from_mask)(mask, **kwargs) 
        for mask in masks
    ]
    logging.info('Features: computing..')
    with ProgressBar():
        features = dask.compute(*tasks)
    feature_matrix = np.stack(features, axis=0, dtype=np.float32)

    logging.info('Features: saving..')
    if not filepath.endswith('.npz'):
        filepath += '.npz'
    np.savez_compressed(
        filepath, 
        features=feature_matrix, 
        original_shape=masks[0].shape, 
        labels=labels,
        **kwargs,
    )
    logging.info('Spectral features: finished..')

def pca_from_features_npz(feature_file, pca_file):
    """
    Fits PCA and saves results while preserving all original metadata.
    """
    with np.load(feature_file) as data:
        features = data['features']
        original_shape = data['original_shape']
        labels = data['labels']
        kwargs = data['kwargs']

    # Fit the PCA
    pca = PCA()
    pca.fit(features)

    # This saves the original metadata + the new PCA keys
    np.savez(pca_file, 
        mean = pca.mean_,
        components = pca.components_,
        variance = pca.explained_variance_,
        variance_ratio = pca.explained_variance_ratio_, 
        original_shape = original_shape,
        labels = labels,
        kwargs = kwargs,    
    )

    return pca.explained_variance_ratio_


def coefficients_from_features_npz(feature_file, pca_file, coeffs_file):

    # Load the features
    with np.load(feature_file) as data:
        features = data['features'] # (n_samples, n_features)
        labels = data['labels']

    # Load the PCA matrices
    # 1. Load the matrices
    with np.load(pca_file) as data:
        mean_vec = data['mean']        # Shape: (n_features,)
        components = data['components'] # Shape: (n_components, n_features)
        variance = data['variance']    # Shape: (n_components,)

    # 1. Center the data
    # Broadcasting handles (N, F) - (F,) automatically
    centered_features = features - mean_vec

    # 2. Projection (The "Transform" step)
    # Matrix Multiplication: (N, F) @ (F, K) -> (N, K)
    scores = centered_features @ components.T

    # 3. Calculate Sigma (Z-Score)
    # Broadcasting handles (N, K) / (K,) automatically
    coeffs = scores / np.sqrt(variance)

    np.savez(coeffs_file, coeffs=coeffs, labels=labels)


def modes_from_pca_npz(
    mask_from_features: Callable,
    pca_file, 
    modes_file, 
    n_components=8, 
    n_coeffs=11, 
    max_coeff=2,
):
    # coeffs is list of coefficient vectors
    # Each coefficient vector has dimensionless coefficients in the components
    # x_i = mean + α_i * sqrt(variance_i) * component_i
    coeffs = np.linspace(-max_coeff, max_coeff, n_coeffs)

    with np.load(pca_file) as data:
        var = data['variance']
        avr = data['mean']
        comps = data['components']
        original_shape = data['original_shape']
        kwargs = data['kwargs']

    sdev = np.sqrt(var)    # Shape: (n_components,)
    mask_shape = (n_coeffs, n_components) + tuple(original_shape)
    masks = np.empty(mask_shape, dtype=bool)

    n_iter = n_coeffs * n_components
    iterator = product(range(n_coeffs), range(n_components))
    for j, i in tqdm(iterator, total=n_iter, desc='Computing modes from PCA'):
        feat = avr + coeffs[j] * sdev[i] * comps[i,:]
        masks[j,i,...] = mask_from_features(feat, original_shape, **kwargs)

    np.savez(modes_file, masks=masks, coeffs=coeffs)


# ZARRAYS


def features_from_dataset_zarr(
    features_from_mask: Callable,
    masks_zarr_path: str,
    output_zarr_path: str,
    **kwargs,
    
):
    """Max memory usage: (1 mask + 1 feature vector) x overhea dof features_from_mask"""
    logging.info(f"Feature calc: connecting to {os.path.basename(masks_zarr_path)}..")
    
    # 1. Input: Lazy connection to masks
    d_masks = da.from_zarr(masks_zarr_path, component='masks')
    n_samples = d_masks.shape[0]

    # 2. Metadata Shape Probe (Run once to get feature dimensions)
    # We compute just the first mask to see what the feature vector looks like
    sample_mask = d_masks[0].compute() 
    sample_feature = features_from_mask(sample_mask, **kwargs)
    n_features = sample_feature.shape[0]
    dtype = sample_feature.dtype
    
    logging.info(f"Feature vector shape: ({n_features},). Chunks: (1, {n_features})")

    # 3. Construction: Build a simple Task Graph
    # We treat every single mask as an independent task
    delayed_func = dask.delayed(features_from_mask)
    lazy_rows = []
    
    for i in range(n_samples):
        # delayed task handles one volume at a time
        task = delayed_func(d_masks[i], **kwargs)
        # from_delayed turns that task into a dask 'block'
        d_row = da.from_delayed(task, shape=(n_features,), dtype=dtype)
        lazy_rows.append(d_row) 

    # Stack them into a matrix and ensure the chunking is (1, n_features)
    d_feature_matrix = da.stack(lazy_rows).rechunk({0: 1, 1: -1})

    # 4. Storage Preparation
    if not output_zarr_path.endswith('.zarr'):
        output_zarr_path += '.zarr'
        
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)

    # 5. Execution: Stream to Disk
    logging.info(f"Streaming {n_samples} feature vectors to {output_zarr_path}...")
    with ProgressBar():
        # Because chunks=(1, n_features), Dask writes one small file per sample.
        # This is extremely resilient on HPC.
        d_feature_matrix.to_zarr(
            output_zarr_path, 
            component='features', 
            overwrite=True,
            compressor=compressor
        )    

    # 6. Metadata Transfer
    input_root = zarr.open(masks_zarr_path, mode='r')
    output_root = zarr.open(output_zarr_path, mode='a')
    output_root.array('labels', input_root['labels'][:])
    output_root.attrs['original_shape'] = d_masks.shape[1:]
    output_root.attrs['kwargs'] = kwargs
    
    logging.info('Feature calc: finished.')



def pca_from_features_zarr(
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
    return pca.explained_variance_ratio_



def scores_from_features_zarr(
    features_zarr_path, 
    pca_zarr_path, 
    output_zarr_path, 
    chunk_size=100
):
    """
    Memory usage:

    n_features x (chunk_size + n_components) x 8
    """
    logging.info("Connecting to Zarr stores (Lazy)...")
    
    # 1. Open Zarr as Dask Arrays
    # We chunk by 'chunk_size' samples and ALL features (-1)
    features = da.from_zarr(features_zarr_path, component='features').rechunk({0: chunk_size, 1: -1})
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
    
    logging.info("Finished PCA projection.")
    return True


def modes_from_pca_zarr(
    mask_from_features: Callable,
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
    stored_k = z_pca['components'].shape[0]
    limit_k = min(n_components, stored_k)
    
    # Load Mean and Metadata
    # mean and sdev are small (F,), safe for RAM
    avr = z_pca['mean'][:]
    sdev = np.sqrt(z_pca['variance'][:limit_k])
    
    shape = tuple(z_pca.attrs['original_shape'])
    kwargs = z_pca.attrs.get('kwargs') 

    # 2. Setup Step Coefficients
    coeffs_range = np.linspace(-max_coeff, max_coeff, n_coeffs)

    # 3. Setup 5D Output Store
    if not modes_zarr_path.endswith('.zarr'):
        modes_zarr_path += '.zarr'
        
    store = zarr.DirectoryStore(modes_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    
    # Shape: (Steps, Modes, D, H, W)
    out_shape = (n_coeffs, limit_k) + shape
    # Chunking by 1 mask per chunk is perfect for 3D slice viewers
    chunks = (1, 1) + shape 
    
    z_modes = root.create_dataset(
        'modes',
        shape=out_shape,
        chunks=chunks,
        dtype=bool,
        compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    )
    
    # Save attributes for the visualization tool
    root.attrs['coeffs'] = coeffs_range.tolist()
    root.attrs['n_components'] = limit_k

    # 4. Generate Modes (The "Safe" Loop)
    logging.info(f"Modes: Reconstructing {n_coeffs * limit_k} volumes...")
    
    # OPTIMIZATION: We only load ONE principal component into RAM at a time.
    # If F=2,000,000, loading all 100 components at once is 800MB.
    # Loading one at a time is only 8MB.
    for i in tqdm(range(limit_k), desc=f"Modes: Reconstructing {n_coeffs * limit_k} volumes..."):
        # Load ONE component vector (1, F)
        current_comp = z_pca['components'][i, :]
        
        for j in range(n_coeffs):
            # Formula: x = mean + (deviation_scalar * eigen_vector)
            # coeffs_range[j] is the sigma (e.g. -2.0, 0.0, 2.0)
            feat_vec = avr + (coeffs_range[j] * sdev[i] * current_comp)
            
            # Reconstruct the 3D volume from the 1D feature vector
            mask_3d = mask_from_features(feat_vec, shape, **kwargs)
            
            # Explicitly cast to bool before writing to save Zarr buffer space
            z_modes[j, i, ...] = mask_3d.astype(bool)

    logging.info(f"Modes: Successfully saved to {modes_zarr_path}")



def masks_from_scores_zarr(
    mask_from_features: Callable,
    pca_zarr_path: str, 
    scores_zarr_path: str,
    output_zarr_path: str,
    target_labels: list = None,
    components: list = None,
    chunk_size: int = 10  # Process N reconstructions before clearing cache
):
    """
    Memory-safe reconstruction of 3D masks from PCA scores.

    memory usage: n_components x n_features x 4 + mask size x (mask_from_features buffer)
    """
    # 1. Open Zarr Stores lazily
    z_pca = zarr.open(pca_zarr_path, mode='r')
    z_scores = zarr.open(scores_zarr_path, mode='r')
    
    # Load small metadata into RAM
    avr = z_pca['mean'][:]  # 40MB - Safe
    shape = tuple(z_pca.attrs['original_shape'])
    kwargs = z_pca.attrs.get('kwargs', {})
    
    # Handle labels
    all_labels = [str(l) for l in z_scores['labels'][:]]
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    
    if target_labels is not None:
        valid_indices = [label_to_idx[l] for l in target_labels if l in label_to_idx]
        found_labels = [l for l in target_labels if l in label_to_idx]
    else:
        valid_indices = list(range(len(all_labels)))
        found_labels = all_labels

    if not valid_indices:
        logging.error("No valid labels found.")
        return

    # 2. Optimized Component Loading
    # Instead of loading 41GB, we only load the subset of rows we actually use.
    logging.info("Loading required PCA components...")
    if components is not None:
        # Load only specific rows (e.g., top 100 components = ~4GB)
        eig_vecs = z_pca['components'].get_basic_selection(components)
    else:
        # If None, we point to the whole dataset but DO NOT use [:]
        # This remains a Zarr Array object (on-disk)
        eig_vecs = z_pca['components']

    # 3. Setup Output Store
    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    z_recons = root.create_dataset(
        'reconstructed_masks',
        shape=(len(valid_indices),) + shape,
        chunks=(1,) + shape,
        dtype=bool,
        compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    )
    root.create_dataset('labels', data=found_labels, dtype=object)

    # 4. Reconstruction Loop
    logging.info(f"Reconstructing {len(valid_indices)} volumes...")

    for i, idx in enumerate(valid_indices):
        # Retrieve score for this subject
        full_score = z_scores['scores'][idx, :]
        
        if components is not None:
            limited_score = full_score[components]
            # limited_vecs is already in RAM as a subset
            projection = np.dot(limited_score, eig_vecs)
        else:
            # If all components are needed, we compute the dot product 
            # without loading the whole 41GB array into a single variable.
            # Zarr's __getitem__ handles the buffering.
            projection = np.dot(full_score, eig_vecs)
        
        # Features = Mean + Projection
        feat_vec = avr + projection
        
        # Generate 3D volume
        mask_3d = mask_from_features(feat_vec, shape, **kwargs)
        z_recons[i, ...] = mask_3d.astype(bool)
        
        if (i + 1) % chunk_size == 0:
            logging.info(f"Progress: {i+1}/{len(valid_indices)} masks completed.")

    logging.info("Reconstruction finished successfully.")





def pca_performance(
    mask_from_features: Callable, 
    pca_zarr_path: str, 
    scores_zarr_path: str, 
    original_masks_zarr_path: str, 
    marginal_dice_path: str,
    cumulative_dice_path: str,
    n_components: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates PCA reconstruction accuracy by computing marginal and cumulative Dice scores.
    Runs sequentially to ensure stability and low memory footprint.

    Memory footprint 2 times the mask size + mask_from_features overhead
    """

    # 2. Open Zarr Stores
    try:
        z_pca = zarr.open(pca_zarr_path, mode='r')
        z_scores = zarr.open(scores_zarr_path, mode='r')
        z_orig = zarr.open(original_masks_zarr_path, mode='r')
    except Exception as e:
        logging.error(f"Failed to open Zarr stores: {e}")
        raise

    # 3. Load Shared Basis and Metadata
    # We load these once into memory for speed
    avr = z_pca['mean'][:]
    eig_vecs = z_pca['components'][:n_components, :]
    shape = tuple(z_pca.attrs['original_shape'])
    kwargs = z_pca.attrs.get('kwargs', {})
    
    # Align subject labels
    labels_scores = [str(l) for l in z_scores['labels'][:]]
    labels_orig = [str(l) for l in z_orig['labels'][:]]
    orig_label_to_idx = {l: i for i, l in enumerate(labels_orig)}
    
    n_samples = len(labels_scores)
    
    # 4. Initialize Output Matrices
    marginal_dice = np.zeros((n_samples, n_components), dtype=np.float32)
    cumulative_dice = np.zeros((n_samples, n_components), dtype=np.float32)

    # 5. Sequential Processing Loop
    logging.info(f"Starting PCA performance evaluation for {n_samples} subjects.")
    
    for i, label in enumerate(tqdm(labels_scores, desc="Evaluating Subjects")):
        # Check if label exists in ground truth
        if label not in orig_label_to_idx:
            logging.warning(f"Label {label} not found in original masks. Skipping.")
            continue
            
        # Get ground truth mask and subject-specific scores
        orig_idx = orig_label_to_idx[label]
        orig_mask = z_orig['masks'][orig_idx]
        subject_scores = z_scores['scores'][i, :n_components]
        
        # Start cumulative reconstruction from the average
        current_cumulative_vec = avr.copy()
        
        tasks = []
        for k in range(n_components):
            task_k = dask.delayed(_pca_performance_component)(current_cumulative_vec, k, i, subject_scores, eig_vecs, avr, mask_from_features, shape, orig_mask, marginal_dice, cumulative_dice, **kwargs)
            tasks.append(task_k)
        dask.compute(tasks)

    # 6. Save Results to CSV
    # Using labels as index to ensure the CSV is self-documenting
    pd.DataFrame(marginal_dice, index=labels_scores).to_csv(marginal_dice_path)
    pd.DataFrame(cumulative_dice, index=labels_scores).to_csv(cumulative_dice_path)
    
    logging.info(f"Performance evaluation complete.")
    logging.info(f"Matrices saved to: \n1. {marginal_dice_path}\n2. {cumulative_dice_path}")

    return marginal_dice, cumulative_dice

def _pca_performance_component(current_cumulative_vec, k, i, subject_scores, eig_vecs, avr, mask_from_features, shape, orig_mask, marginal_dice, cumulative_dice, **kwargs):
    score_k = subject_scores[k]
    vec_k = eig_vecs[k, :]
    
    # --- Marginal Reconstruction (PC k alone) ---
    # Reconstruct mask using Mean + single PC contribution
    marginal_feat_vec = avr + (score_k * vec_k)
    m_recon = mask_from_features(marginal_feat_vec, shape, **kwargs)
    marginal_dice[i, k] = dice_coefficient(m_recon, orig_mask)
    
    # --- Cumulative Reconstruction (Sum 0 to k) ---
    # Efficiently update the running sum vector
    current_cumulative_vec += (score_k * vec_k)
    c_recon = mask_from_features(current_cumulative_vec, shape, **kwargs)
    cumulative_dice[i, k] = dice_coefficient(c_recon, orig_mask)


# Simple in-memory version
# def scores_from_features_zarr(features_zarr_path, pca_zarr_path, output_zarr_path):
#     logging.info("Loading feature matrix into RAM...")
    
#     # 1. Load EVERYTHING into RAM
#     feat_root = zarr.open(features_zarr_path, mode='r')
#     # The [:] triggers a full load into a standard NumPy array
#     features = feat_root['features'][:] 
#     labels = feat_root['labels'][:]

#     # 2. Load PCA Model
#     pca_root = zarr.open(pca_zarr_path, mode='r')
#     mu = pca_root['mean'][:]
#     eig_vecs = pca_root['components'][:]
#     var = pca_root['variance'][:]
#     var_ratio = pca_root['variance_ratio'][:]

#     # 3. Pure NumPy Math (Instantaneous)
#     logging.info("Computing projection...")
#     centered = features - mu
#     scores = np.dot(centered, eig_vecs.T)
#     normalized_scores = scores / np.sqrt(var)

#     # 4. Save back to Zarr (for consistency in your pipeline)
#     store = zarr.DirectoryStore(output_zarr_path)
#     root = zarr.group(store=store, overwrite=True)
#     root.create_dataset('scores', data=scores, chunks=(1, scores.shape[1]))
#     root.create_dataset('normalized_scores', data=normalized_scores, chunks=(1, scores.shape[1]))
#     root.create_dataset('labels', data=labels)
#     root.create_dataset('variance', data=var)
#     root.create_dataset('variance_ratio', data=var_ratio)
    
#     logging.info("Finished PCA projection.")


# def masks_from_scores_zarr(
#     mask_from_features: Callable,
#     pca_zarr_path: str, 
#     scores_zarr_path: str,
#     output_zarr_path: str,
#     target_labels: list=None,
#     components: list=None,
# ):
#     """
#     Reconstructs 3D masks only for the provided list of kidney labels.
    
#     Args:
#         mask_from_features: The reconstruction function.
#         pca_zarr_path: Path to the PCA model Zarr.
#         scores_zarr_path: Path to the scores/labels Zarr.
#         output_zarr_path: Path to save the reconstructed masks.
#         target_labels (list): List of strings (e.g. ['101-L', '105-R']).
#         components (list): List of principal components (indices) to use in the reconstruction.
#     """
#     # 1. Open Zarr Stores
#     z_pca = zarr.open(pca_zarr_path, mode='r')
#     z_scores = zarr.open(scores_zarr_path, mode='r')
    
#     # Load metadata and labels
#     avr = z_pca['mean'][:]
#     eig_vecs = z_pca['components'][:]
#     shape = tuple(z_pca.attrs['original_shape'])
#     kwargs = z_pca.attrs.get('kwargs', {})
    
#     # Load all labels and create a lookup dictionary {label: index}
#     all_labels = z_scores['labels'][:]
#     label_to_idx = {str(label): i for i, label in enumerate(all_labels)}
    
#     # 2. Filter for valid indices based on target_labels

#     if target_labels is None:
#         found_labels = []
#         valid_indices = []
#         for label in target_labels:
#             if label in label_to_idx:
#                 found_labels.append(label)
#                 valid_indices.append(label_to_idx[label])
#             else:
#                 logging.warning(f"Label {label} not found in scores Zarr. Skipping.")

#         if not valid_indices:
#             logging.error("No valid labels found. Aborting reconstruction.")
#             return
#     else:
#         found_labels = target_labels
#         valid_indices = list(range(len(found_labels)))

#     # 3. Setup Output Store
#     store = zarr.DirectoryStore(output_zarr_path)
#     root = zarr.group(store=store, overwrite=True)
    
#     z_recons = root.create_dataset(
#         'reconstructed_masks',
#         shape=(len(valid_indices),) + shape,
#         chunks=(1,) + shape,
#         dtype=bool,
#         compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
#     )
    
#     root.create_dataset('labels', data=found_labels, dtype=object)

#     # 4. Reconstruction Loop
#     logging.info(f"Reconstructing {len(valid_indices)} specific kidney volumes...")

    
#     for i, idx in enumerate(valid_indices):
#         # 1. Retrieve the full score vector for this subject
#         full_score = z_scores['scores'][idx, :]
        
#         # 2. Slice both scores and eigenvectors to only use the first N
#         # This effectively ignores the higher-frequency details
#         if components is None:
#             limited_score = full_score
#             limited_vecs = eig_vecs
#         else:
#             limited_score = full_score[components]
#             limited_vecs = eig_vecs[components, :]
        
#         # 3. Project back: Features = Mean + (Limited_Scores @ Limited_Eigenvectors)
#         # The mean (avr) remains the same size
#         feat_vec = avr + np.dot(limited_score, limited_vecs)
        
#         # 4. Reconstruct 3D mask
#         # If using wavelets, ensure kwargs contains the correct coeff_slices
#         mask_3d = mask_from_features(feat_vec, shape, **kwargs)
        
#         z_recons[i, ...] = mask_3d.astype(bool)
        
#         if i % 5 == 0:
#             logging.info(f"Progress: {i+1}/{len(valid_indices)} reconstructed.")

#     logging.info(f"Successfully saved {len(found_labels)} reconstructions to {output_zarr_path}")


# Simpler version using standard PCA - fails for large feature vectors
# def pca_from_features_zarr(
#     features_zarr_path: str, 
#     output_zarr_path: str, 
#     n_components=None,
# ):
#     """
#     Hybrid PCA: Loads the (N, F) feature matrix into RAM (safe for 32k features),
#     fits standard PCA, and saves results to Zarr.
#     """
#     logging.info(f"PCA: Loading features from {os.path.basename(features_zarr_path)}...")

#     # 1. Load Feature Matrix into RAM
#     # 1108 x 32000 float32 is ~140MB. Safe for any 8GB worker.
#     feat_root = zarr.open(features_zarr_path, mode='r')
#     features = feat_root['features'][:] 
#     labels = feat_root['labels'][:]

#     # 2. Fit Standard PCA (In-Memory)
#     # solver='full' or 'auto' is usually best for these dimensions
#     logging.info(f"PCA: Fitting Sklearn PCA on {features.shape} matrix...")
#     pca = PCA(n_components=n_components, svd_solver='auto')
#     pca.fit(features)

#     # 3. Prepare Output Zarr
#     if not output_zarr_path.endswith('.zarr'):
#         output_zarr_path += '.zarr'
    
#     store = zarr.DirectoryStore(output_zarr_path)
#     root = zarr.group(store=store, overwrite=True)
#     compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)

#     # 4. Save Results
#     # We save components with (1, n_features) chunks for easy row-access later
#     logging.info("PCA: Saving model attributes to Zarr...")
    
#     root.create_dataset('components', 
#                        data=pca.components_.astype(np.float32), 
#                        chunks=(1, features.shape[1]),
#                        compressor=compressor)
    
#     root.create_dataset('mean', 
#                        data=pca.mean_.astype(np.float32), 
#                        compressor=compressor)
    
#     root.create_dataset('variance', data=pca.explained_variance_.astype(np.float32))
#     root.create_dataset('variance_ratio', data=pca.explained_variance_ratio_.astype(np.float32))
#     root.create_dataset('labels', data=labels)

#     # 5. Transfer Attributes
#     root.attrs['original_shape'] = feat_root.attrs.get('original_shape', None)
#     root.attrs['kwargs'] = feat_root.attrs.get('kwargs')

#     logging.info(f"PCA: Finished. Top variance ratio: {pca.explained_variance_ratio_[0]:.4f}")
#     return pca.explained_variance_ratio_
