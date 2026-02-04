import os
import numpy as np
from sklearn.decomposition import PCA
import logging
import dask
from dask.diagnostics import ProgressBar
from itertools import product
from tqdm import tqdm
import zarr
import dask.array as da
from dask_ml.decomposition import PCA as DaskPCA
import numpy as np
import psutil
from collections.abc import Callable


def features_from_dataset_in_memory(
    features_from_mask:Callable,
    masks_file:str, 
    filepath:str, 
    **kwargs, # kwargs for features_from_mask
):
    logging.info("Features: loading masks..")
    with np.load(masks_file) as data:
        masks = data['masks']
        labels = data['labels']

    logging.info("Features: scheduling tasks..")
    tasks = [
        dask.delayed(features_from_mask)(masks[i,...], **kwargs) 
        for i in range(masks.shape[0])
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
        original_shape=masks.shape[1:], 
        labels=labels,
        **kwargs,
    )
    logging.info('Spectral features: finished..')


def features_from_dataset_zarr(
    features_from_mask:Callable,
    masks_zarr_path: str, 
    output_zarr_path: str, 
    chunk_size='auto', 
    **kwargs, # keyword arguments for features_from_mask
):
    logging.info(f"Feature calc: connecting to {os.path.basename(masks_zarr_path)}..")
    
    # 1. Input: Connect to the Masks Zarr (Lazy)
    d_masks = da.from_zarr(masks_zarr_path, component='masks')
    n_samples = d_masks.shape[0]

    # 2. Metadata: Determine output shape dynamically
    # We compute ONE sample immediately to find out how big the feature vector is.
    # This prevents us from hardcoding the feature size.
    logging.info("Feature calc: computing shape probe on first mask..")
    
    # We use .compute() on the first slice to run it eagerly
    sample_mask = d_masks[0].compute() 
    sample_feature = features_from_mask(sample_mask, **kwargs)
    
    n_features = sample_feature.shape[0]
    dtype = sample_feature.dtype
    logging.info(f"Feature vector shape detected: ({n_features},). Type: {dtype}")

    # 3. Construction: Build the Dask Graph (The "Lazy" Array)
    # We create a list of Dask Arrays, one per mask.
    lazy_rows = []
    
    # Create the delayed function wrapper once
    delayed_func = dask.delayed(features_from_mask)

    for i in range(n_samples):
        # Create a delayed task for this mask
        # Note: d_masks[i] is lazy, so we aren't reading the mask yet
        task = delayed_func(d_masks[i], **kwargs)
        
        # Convert the delayed task into a Dask Array (Row)
        # We MUST specify shape and dtype so Dask knows how to stitch them together
        d_row = da.from_delayed(task, shape=(n_features,), dtype=dtype)
        
        # Reshape to (1, F) so we can stack them vertically later
        d_row = d_row[None, :] 
        lazy_rows.append(d_row)

    # Stack them into one big matrix (N, F)
    # This matrix exists only as a graph of future tasks, not in RAM.
    d_feature_matrix = da.vstack(lazy_rows)

    # 4. Storage: Prepare Output Zarr
    if not output_zarr_path.endswith('.zarr'):
        output_zarr_path += '.zarr'
        
    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)

    # 5. Output Chunking logic
    # If the output matrix is huge, we should chunk it sensibly on disk.
    # Defaulting to None lets Dask decide, or we can enforce typical sizes.
    # If 'auto', we let Dask decide based on the input chunks.
    if chunk_size != 'auto':
        d_feature_matrix = d_feature_matrix.rechunk({0: chunk_size})

    logging.info(f"Feature calc: Streaming results to {output_zarr_path}...")
    
    # 6. Execution: Stream to Disk
    # .to_zarr computes the chunks in parallel and writes them directly to disk.
    # It never holds the full matrix in memory.
    with ProgressBar():
        d_feature_matrix.to_zarr(store, component='features', compute=True)

    # 7. Metadata: Save labels and attributes
    # Copy labels from input to output
    input_root = zarr.open(masks_zarr_path, mode='r')
    root.array('labels', input_root['labels'][:])
    
    # Save attributes (original shape, order, etc.)
    root.attrs['original_shape'] = d_masks.shape[1:] # (D, H, W)
    root.attrs['kwargs'] = kwargs
    
    logging.info('Feature calc: finished.')




def pca_from_features_in_memory(feature_file, pca_file):
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



def pca_from_features_zarr(
    features_zarr_path: str, 
    output_zarr_path: str, 
    n_components=None,
    chunk_size='auto'
):
    """
    Fits PCA on a large-than-memory features Zarr array and saves results to Zarr.
    """
    logging.info(f"PCA: Connecting to feature store at {os.path.basename(features_zarr_path)}..")

    # 1. Connect to Features
    # Note: Component must match what was saved in the previous step ('features')
    d_features = da.from_zarr(features_zarr_path, component='features')
    
    # 2. Optimize Chunking
    if chunk_size == 'auto':
        chunk_size = get_optimal_chunk_size(
            d_features.shape[1:], 
            dtype=d_features.dtype
        )
        logging.info(f"PCA: Auto-chunking set to {chunk_size} samples per batch.")

    d_features = d_features.rechunk({0: chunk_size})

    # 3. Fit PCA
    if n_components is None:
        n_components = min(d_features.shape)
        
    logging.info(f"PCA: Fitting model with n_components={n_components}...")
    
    # svd_solver='randomized' is efficient for large Dask arrays
    pca = DaskPCA(n_components=n_components, svd_solver='auto')
    
    # This triggers the computation. 
    # Note: pca.components_ becomes a NumPy array in RAM after this.
    pca.fit(d_features)

    # --- FIX STARTS HERE ---
    # dask_ml keeps attributes as lazy arrays. We must compute them to get NumPy arrays.
    logging.info("PCA: Computing attributes (mean, components) into memory...")
    
    # We compute these efficiently in parallel
    # components_ is (n_components, n_features)
    # mean_ is (n_features,)
    pca_mean, pca_components, pca_var, pca_ratio = dask.compute(
        pca.mean_, 
        pca.components_, 
        pca.explained_variance_, 
        pca.explained_variance_ratio_
    )
    # --- FIX ENDS HERE ---
    
    # 4. Prepare Output Zarr
    logging.info(f"PCA: Saving results to {output_zarr_path}...")
    
    if not output_zarr_path.endswith('.zarr'):
        output_zarr_path += '.zarr'

    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)

    # 5. Save PCA Attributes to Zarr
    # We use a compressor to save disk space for these dense matrices
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)

    # A. Components (The largest array: n_components x n_features)
    # We chunk it by component (1 component = 1 chunk) to make retrieving single modes fast
    root.create_dataset(
        'components', 
        data=pca.components_, 
        chunks=(1, None), # Chunk per component
        compressor=compressor
    )

    # B. Mean (n_features,)
    root.create_dataset('mean', data=pca_mean, compressor=compressor)

    # C. Variance stats (Small 1D arrays)
    root.create_dataset('variance', data=pca_var)
    root.create_dataset('variance_ratio', data=pca_ratio)

    # 6. Transfer Metadata & Labels
    logging.info("PCA: Copying all original metadata...")
    input_root = zarr.open(features_zarr_path, mode='r')
    
    # Preserve kwargs
    root.attrs['kwargs'] = input_root.attrs['kwargs']
    root.attrs['original_shape'] = input_root.attrs['original_shape']
        
    # Preserve labels
    root.create_dataset('labels', data=input_root['labels'][:])

    logging.info("PCA: Finished.")
    return pca_ratio


def coefficients_from_features_in_memory(feature_file, pca_file, coeffs_file):

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


def coefficients_from_features_zarr(
    features_zarr_path: str, 
    pca_zarr_path: str, 
    output_zarr_path: str,
    chunk_size='auto'
):
    """
    Computes PCA coefficients (scores normalized by variance) from Zarr inputs
    and streams the results to a new Zarr store.
    """
    logging.info(f"Coeffs: Connecting to features at {os.path.basename(features_zarr_path)}..")
    
    # 1. Connect to Inputs (Lazy)
    # Features (N, F)
    d_features = da.from_zarr(features_zarr_path, component='features')
    
    # PCA Model (Loaded into RAM)
    # Since the PCA matrices (components, mean) are usually fit for RAM 
    # (unless F is massive >100k), we typically load them as NumPy arrays 
    # to broadcast them efficiently across the Dask chunks.
    logging.info(f"Coeffs: Loading PCA model from {os.path.basename(pca_zarr_path)}..")
    z_pca = zarr.open(pca_zarr_path, mode='r')
    
    mean_vec = z_pca['mean'][:]            # (F,)
    components = z_pca['components'][:]    # (K, F)
    variance = z_pca['variance'][:]        # (K,)
    
    # 2. Rechunk Features
    # We apply the same optimization logic as before to handle large N
    if chunk_size == 'auto':
        chunk_size = get_optimal_chunk_size(
            d_features.shape[1:], 
            dtype=d_features.dtype
        )
    
    d_features = d_features.rechunk({0: chunk_size})

    # 3. Define the Computation (Lazy Graph)
    
    # A. Center the data
    # Dask handles the broadcasting: (Chunk_i, F) - (F,)
    centered_features = d_features - mean_vec

    # B. Projection
    # Matrix Multiplication: (N, F) @ (F, K) -> (N, K)
    # Since 'components' is a numpy array, Dask sends it to every worker automatically.
    scores = centered_features @ components.T

    # C. Normalize (Z-Score)
    # (N, K) / (K,)
    coeffs = scores / np.sqrt(variance)

    # 4. Prepare Output Storage
    if not output_zarr_path.endswith('.zarr'):
        output_zarr_path += '.zarr'
        
    logging.info(f"Coeffs: Streaming results to {output_zarr_path}...")
    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    
    # 5. Execute and Save
    # We save the coefficients to component 'coeffs'
    with ProgressBar():
        coeffs.to_zarr(store, component='coeffs', compute=True)
        
    # 6. Transfer Metadata (Labels, etc.)
    # Often helpful to keep the labels associated with these coefficients
    input_root = zarr.open(features_zarr_path, mode='r')
    root.create_dataset('labels', data=input_root['labels'][:])
        
    logging.info("Coeffs: Finished.")



def modes_from_pca_in_memory(
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



def modes_from_pca_zarr(
    mask_from_features: Callable,
    pca_zarr_path: str, 
    modes_zarr_path: str, 
    n_components=8, 
    n_coeffs=11, 
    max_coeff=2
):
    """
    Generates 3D shape modes from a Zarr PCA model and saves them to a Zarr array.
    
    Output Shape: (n_coeffs, n_components, Depth, Height, Width)
    """
    logging.info(f"Modes: Loading PCA model from {os.path.basename(pca_zarr_path)}..")
    
    # 1. Load PCA Model (Small enough for RAM)
    # We open in read mode
    z_pca = zarr.open(pca_zarr_path, mode='r')
    
    # Read the attributes we need
    avr = z_pca['mean'][:]               # (F,)
    
    # Handle case where stored components > requested n_components
    stored_components = z_pca['components'] # Lazy load first
    limit_k = min(n_components, stored_components.shape[0])
    comps = stored_components[:limit_k]  # Load only what we need (K, F)
    
    # Calculate Standard Deviation from Variance
    variance = z_pca['variance'][:limit_k]
    sdev = np.sqrt(variance)
    
    # Retrieve Metadata
    # We need the original 3D shape to reconstruct the masks
    shape = tuple(z_pca.attrs['original_shape'])
    kwargs = z_pca.attrs['kwargs'] 

    # 2. Setup Coefficients
    # e.g., linspace(-2, 2, 11) -> [-2., -1.6, ... 0 ... 1.6, 2.]
    coeffs = np.linspace(-max_coeff, max_coeff, n_coeffs)

    # 3. Setup Output Zarr
    if not modes_zarr_path.endswith('.zarr'):
        modes_zarr_path += '.zarr'
        
    logging.info(f"Modes: Creating 5D output store at {modes_zarr_path}..")
    store = zarr.DirectoryStore(modes_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    
    # Define 5D Shape: (Steps, Modes, D, H, W)
    out_shape = (n_coeffs, limit_k) + shape
    
    # Chunking Strategy:
    # We write 1 mask at a time. So a chunk size of (1, 1, D, H, W) is safest.
    # It ensures that updating one mask doesn't require reading/writing neighbors.
    chunks = (1, 1) + shape
    
    z_masks = root.create_dataset(
        'modes',
        shape=out_shape,
        chunks=chunks,
        dtype=bool, # Masks are boolean
        compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    )
    
    # Save metadata for the viewer
    root.attrs['coeffs'] = coeffs.tolist()
    root.attrs['n_components'] = limit_k

    # 4. Generate Modes
    # Total iterations
    n_iter = n_coeffs * limit_k
    iterator = product(range(n_coeffs), range(limit_k))
    
    logging.info(f"Modes: Generating {n_iter} 3D masks...")
    
    for j, i in tqdm(iterator, total=n_iter, desc='Reconstructing Modes'):
        # Formula: x = mean + (sigma * scalar * vector)
        # j = coefficient index (e.g., -2 sigma)
        # i = component index (e.g., Mode 1)
        
        # Calculate feature vector
        feat = avr + (coeffs[j] * sdev[i] * comps[i, :])
        
        # Reconstruct 3D mask (CPU intensive step)
        mask_3d = mask_from_features(feat, shape, **kwargs)
        
        # Write directly to disk
        # This writes to the specific chunk for (j, i), keeping RAM clean
        z_masks[j, i, ...] = mask_3d

    logging.info("Modes: Finished.")





# Helper function
def get_optimal_chunk_size(shape, dtype, target_mb=250):
    """
    Calculates the optimal number of masks per chunk based on the specific dtype size.
    """
    # 1. Dynamically get bytes per voxel based on the dtype argument
    # np.int32 -> 4 bytes
    # np.float64 -> 8 bytes
    # np.bool_ -> 1 byte
    bytes_per_voxel = np.dtype(dtype).itemsize
    
    # 2. Calculate size of ONE mask in Megabytes (MB)
    one_mask_bytes = np.prod(shape) * bytes_per_voxel
    one_mask_mb = one_mask_bytes / (1024**2)
    
    # 3. Constraint A: Dask Target Size (~250MB)
    if one_mask_mb > target_mb:
        dask_optimal_count = 1
    else:
        dask_optimal_count = int(target_mb / one_mask_mb)

    # 4. Constraint B: System RAM Safety Net (10% of Available RAM)
    available_ram_mb = psutil.virtual_memory().available / (1024**2)
    safe_ram_limit_mb = available_ram_mb * 0.10
    ram_limited_count = int(safe_ram_limit_mb / one_mask_mb)
    
    # 5. Pick the safer number
    final_count = min(dask_optimal_count, ram_limited_count)
    
    return max(1, final_count)