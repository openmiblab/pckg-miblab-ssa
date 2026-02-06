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
from collections.abc import Callable
import dask.delayed




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





def features_from_dataset_zarr(
    features_from_mask: Callable,
    masks_zarr_path: str,
    output_zarr_path: str,
    **kwargs,
):
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
):
    """
    Hybrid PCA: Loads the (N, F) feature matrix into RAM (safe for 32k features),
    fits standard PCA, and saves results to Zarr.
    """
    logging.info(f"PCA: Loading features from {os.path.basename(features_zarr_path)}...")

    # 1. Load Feature Matrix into RAM
    # 1108 x 32000 float32 is ~140MB. Safe for any 8GB worker.
    feat_root = zarr.open(features_zarr_path, mode='r')
    features = feat_root['features'][:] 
    labels = feat_root['labels'][:]

    # 2. Fit Standard PCA (In-Memory)
    # solver='full' or 'auto' is usually best for these dimensions
    logging.info(f"PCA: Fitting Sklearn PCA on {features.shape} matrix...")
    pca = PCA(n_components=n_components, svd_solver='auto')
    pca.fit(features)

    # 3. Prepare Output Zarr
    if not output_zarr_path.endswith('.zarr'):
        output_zarr_path += '.zarr'
    
    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)

    # 4. Save Results
    # We save components with (1, n_features) chunks for easy row-access later
    logging.info("PCA: Saving model attributes to Zarr...")
    
    root.create_dataset('components', 
                       data=pca.components_.astype(np.float32), 
                       chunks=(1, features.shape[1]),
                       compressor=compressor)
    
    root.create_dataset('mean', 
                       data=pca.mean_.astype(np.float32), 
                       compressor=compressor)
    
    root.create_dataset('variance', data=pca.explained_variance_.astype(np.float32))
    root.create_dataset('variance_ratio', data=pca.explained_variance_ratio_.astype(np.float32))
    root.create_dataset('labels', data=labels)

    # 5. Transfer Attributes
    root.attrs['original_shape'] = feat_root.attrs.get('original_shape', None)
    root.attrs['kwargs'] = feat_root.attrs.get('kwargs')

    logging.info(f"PCA: Finished. Top variance ratio: {pca.explained_variance_ratio_[0]:.4f}")
    return pca.explained_variance_ratio_



def coefficients_from_features_zarr(features_zarr_path, pca_zarr_path, output_zarr_path):
    logging.info("Loading feature matrix into RAM...")
    
    # 1. Load EVERYTHING into RAM (only 140MB, so this is safe)
    feat_root = zarr.open(features_zarr_path, mode='r')
    # The [:] triggers a full load into a standard NumPy array
    features = feat_root['features'][:] 
    labels = feat_root['labels'][:]

    # 2. Load PCA Model
    pca_root = zarr.open(pca_zarr_path, mode='r')
    mu = pca_root['mean'][:]
    eig_vecs = pca_root['components'][:]
    var = pca_root['variance'][:]

    # 3. Pure NumPy Math (Instantaneous)
    logging.info("Computing projection...")
    centered = features - mu
    scores = np.dot(centered, eig_vecs.T)
    coeffs = scores / np.sqrt(var)

    # 4. Save back to Zarr (for consistency in your pipeline)
    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    root.create_dataset('coeffs', data=coeffs, chunks=(1, coeffs.shape[1]))
    root.create_dataset('labels', data=labels)
    
    logging.info("Finished PCA projection using NumPy.")



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



