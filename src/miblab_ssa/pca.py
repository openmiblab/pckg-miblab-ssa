import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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
import dask.array as da
import dask.delayed
import math

from miblab_ssa import utils


def features_from_dataset_in_memory(
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





def features_from_dataset_zarr(
    features_from_mask: Callable,
    masks_zarr_path: str,
    output_zarr_path: str,
    **kwargs,
):
    logging.info(f"Feature calc: connecting to {os.path.basename(masks_zarr_path)}..")
    
    # 1. Connect and Detect Memory
    d_masks = da.from_zarr(masks_zarr_path, component='masks')
    mem_gb = utils.get_memory_limit()
    usable_mem_gb = (mem_gb / 2) * 0.8

    # 2. Metadata Shape Probe
    sample_mask = d_masks[0].compute() 
    sample_feature = features_from_mask(sample_mask, **kwargs)
    n_features = sample_feature.shape[0]
    dtype = sample_feature.dtype
    logging.info(f"Feature vector shape: ({n_features},). Type: {dtype}")
    
    # 3. Calculate Optimal Chunk Size for the Output Matrix
    # We want the chunks of the feature matrix to be manageable for downstream PCA/Stats
    bytes_per_row = n_features * sample_feature.itemsize
    # Aim for ~100MB chunks (sweet spot for Dask dataframes/matrices) or based on RAM
    samples_per_chunk = max(1, math.floor(usable_mem_gb * 1024**3 / bytes_per_row))
    samples_per_chunk = min(samples_per_chunk, d_masks.shape[0])

    logging.info(f"Feature matrix chunks: ({samples_per_chunk}, {n_features})")

    # 4. Construction: Build the Dask Graph
    delayed_func = dask.delayed(features_from_mask)
    lazy_rows = []
    for i in range(d_masks.shape[0]):
        # We pass the delayed task into the array
        task = delayed_func(d_masks[i], **kwargs)
        d_row = da.from_delayed(task, shape=(n_features,), dtype=dtype)
        lazy_rows.append(d_row) 

    # Stack and RECHUNK immediately
    # This ensures that when to_zarr is called, Dask writes in the calculated block sizes
    d_feature_matrix = da.stack(lazy_rows).rechunk({0: samples_per_chunk, 1: -1})

    # 5. Storage Preparation
    if not output_zarr_path.endswith('.zarr'):
        output_zarr_path += '.zarr'

    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)        
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)

    # 6. Execution: Stream to Disk
    logging.info(f"Streaming features to disk at {output_zarr_path}...")
    with ProgressBar():
        # Dask will use the rechunked structure to determine the Zarr chunks automatically
        d_feature_matrix.to_zarr(
            store, 
            component='features', 
            overwrite=True,
            compressor=compressor
        )    

    # 7. Metadata Transfer
    input_root = zarr.open(masks_zarr_path, mode='r')
    root.array('labels', input_root['labels'][:])
    root.attrs['original_shape'] = d_masks.shape[1:]
    root.attrs['kwargs'] = kwargs
    
    logging.info('Feature calc: finished.')


def _OLD_features_from_dataset_zarr(
    features_from_mask: Callable,
    masks_zarr_path: str,
    output_zarr_path: str,
    **kwargs,
):
    logging.info(f"Feature calc: connecting to {os.path.basename(masks_zarr_path)}..")
    
    # 1. Input: Connect to the Masks Zarr (Lazy)
    d_masks = da.from_zarr(masks_zarr_path, component='masks')

    # 2. Metadata Shape Probe
    logging.info("Feature calc: computing shape probe on first mask..")
    sample_mask = d_masks[0].compute() 
    sample_feature = features_from_mask(sample_mask, **kwargs)
    n_features = sample_feature.shape[0]
    dtype = sample_feature.dtype
    logging.info(f"Feature vector shape: ({n_features},). Type: {dtype}")

    # 3. Construction: Build the Dask Graph
    lazy_rows = []
    delayed_func = dask.delayed(features_from_mask)

    for i in range(d_masks.shape[0]):
        task = delayed_func(d_masks[i], **kwargs)
        d_row = da.from_delayed(task, shape=(n_features,), dtype=dtype)
        lazy_rows.append(d_row[None, :]) 

    # 4. Final Array Creation
    d_feature_matrix = da.vstack(lazy_rows)

    # 4. Storage Preparation
    if not output_zarr_path.endswith('.zarr'):
        output_zarr_path += '.zarr'
        
    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)

    # 6. Execution: Stream to Disk
    logging.info(f"Streaming to disk...")
    with ProgressBar():
        d_feature_matrix.to_zarr(
            store, 
            component='features', 
            compute=True, 
            compressor=compressor
        )    

    # 7. Metadata Transfer
    input_root = zarr.open(masks_zarr_path, mode='r')
    root.array('labels', input_root['labels'][:])
    root.attrs['original_shape'] = d_masks.shape[1:]
    root.attrs['kwargs'] = kwargs
    
    logging.info('Feature calc: finished.')


def pca_from_features_zarr(
    features_zarr_path: str, 
    output_zarr_path: str, 
    n_components=None,
):
    """
    Fits PCA on out-of-memory features and streams results to disk.
    """
    logging.info(f"PCA: Connecting to feature store at {os.path.basename(features_zarr_path)}..")

    # 1. Connect to Features
    # Dask inherits the chunking from your previous 'features_from_dataset_zarr' step
    d_features = da.from_zarr(features_zarr_path, component='features')
    
    # 2. Fit PCA (Lazy/Streaming)
    logging.info(f"PCA: Fitting model on {d_features.shape} array...")
    pca = DaskPCA(n_components=n_components, svd_solver='auto')
    pca.fit(d_features)

    # 3. Compute Small Attributes
    # We compute everything EXCEPT the massive components_ matrix first
    logging.info("PCA: Computing variance and mean statistics...")
    pca_mean, pca_var, pca_ratio = dask.compute(
        pca.mean_, 
        pca.explained_variance_, 
        pca.explained_variance_ratio_
    )
    
    # 4. Prepare Output Zarr
    if not output_zarr_path.endswith('.zarr'):
        output_zarr_path += '.zarr'
    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)

    # 5. Save Massive Components (Streaming)
    # INSTEAD of data=pca.components_, we use .to_zarr() to stream it 
    # block-by-block from the Dask graph to the Zarr file.
    logging.info("PCA: Streaming components to disk...")
    
    # Force cast to Dask Array in case the solver returned a NumPy array
    # We use (1, -1) immediately to define the graph's chunking
    if isinstance(pca.components_, np.ndarray):
        d_components = da.from_array(pca.components_, chunks=(1, -1))
    else:
        # If it's already Dask, just apply the rechunk
        d_components = pca.components_.rechunk({0: 1, 1: -1})
    
    d_components.to_zarr(
        store, 
        component='components', 
        compressor=compressor, 
        overwrite=True
    )

    # 6. Save Small Attributes (In-Memory)
    root.create_dataset('mean', data=pca_mean, compressor=compressor)
    root.create_dataset('variance', data=pca_var)
    root.create_dataset('variance_ratio', data=pca_ratio)

    # 7. Transfer Metadata & Labels
    logging.info("PCA: Copying original metadata...")
    input_root = zarr.open(features_zarr_path, mode='r')
    
    root.attrs['kwargs'] = input_root.attrs.get('kwargs', {})
    root.attrs['original_shape'] = input_root.attrs.get('original_shape', None)
    
    # Labels are small (N_samples), safe to load into RAM
    root.create_dataset('labels', data=input_root['labels'][:])

    logging.info("PCA: Finished.")
    return pca_ratio



def coefficients_from_features_zarr(
    features_zarr_path: str, 
    pca_zarr_path: str, 
    output_zarr_path: str,
):
    """
    Computes PCA coefficients using a pure streaming approach.
    No large matrices are loaded into RAM; everything is pulled from Zarr as needed.
    """
    logging.info(f"Coeffs: Connecting to stores...")
    
    # 1. Connect to Inputs (Both Lazy)
    d_features = da.from_zarr(features_zarr_path, component='features') # (N, F)
    
    # Connect to PCA attributes as Dask arrays instead of NumPy
    # This prevents the initial RAM spike
    mean_vec = da.from_zarr(pca_zarr_path, component='mean')           # (F,)
    components = da.from_zarr(pca_zarr_path, component='components')   # (K, F)
    variance = da.from_zarr(pca_zarr_path, component='variance')       # (K,)

    # 2. Define Projection (Mathematical Graph)
    # Centering (Streaming)
    centered = d_features - mean_vec
    
    # Projection (Streaming dot product)
    # Dask will align the chunks of features and components automatically
    scores = da.dot(centered, components.T)
    
    # Normalize (Streaming)
    # We compute sqrt(variance) lazily as well
    coeffs = scores / da.sqrt(variance)

    # 3. Prepare Output
    if not output_zarr_path.endswith('.zarr'):
        output_zarr_path += '.zarr'
    
    store = zarr.DirectoryStore(output_zarr_path)
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)

    # 4. Execute with Thread Governor
    logging.info(f"Coeffs: Streaming projection to {output_zarr_path}...")
    
    # This is the moment where Dask manages the RAM budget
    with ProgressBar():
        coeffs.to_zarr(
            store, 
            component='coeffs', 
            overwrite=True, 
            compressor=compressor
        )
            
    # 5. Metadata Transfer (Small items only)
    # We still need labels for the final plot
    input_root = zarr.open(features_zarr_path, mode='r')
    output_root = zarr.open(store, mode='a')
    
    # labels are small (N_samples), safe to load [:]
    output_root.create_dataset('labels', data=input_root['labels'][:], overwrite=True)
    
    # Transfer variance ratio if available (useful for choosing plot axes)
    pca_root = zarr.open(pca_zarr_path, mode='r')
    output_root.create_dataset('variance_ratio', data=pca_root['variance_ratio'][:], overwrite=True)
        
    logging.info("Coeffs: Finished successfully.")




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
    for i in range(limit_k):
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



def classify_shapes(features_reduced, n_clusters=2, random_state=0):
    """
    Cluster shapes.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features_reduced)
    return labels, kmeans



# UNUSED
def reconstruct_from_scores(scores, model_path="pca_kidney_model.npz"):
    # 1. Load Model
    data = np.load(model_path)
    mean_vec = data['mean']
    components = data['components']
    
    # 2. Reverse Projection
    # Scores (1, n_modes) dot Components (n_modes, n_features) -> (1, n_features)
    # This rebuilds the shape variation from the origin
    shape_variation = np.dot(scores, components)
    
    # 3. Add Mean
    # Move the shape from the origin back to the "Average Kidney" location
    reconstructed_vector = shape_variation + mean_vec
    
    return reconstructed_vector



