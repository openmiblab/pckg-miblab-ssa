from typing import Callable
import logging
import dask
from tqdm import tqdm
import zarr
import dask.delayed
import numpy as np
import gc
from joblib import Parallel, delayed


def features_from_dataset(
    features_from_mask: Callable,
    masks_zarr_path: str,
    output_zarr_path: str,
    n_jobs: int = -1, # -1 uses all available cores
    **kwargs
):
    # 1. Open Input and Probe Shape
    input_root = zarr.open(masks_zarr_path, mode='r')
    n_samples = input_root['masks'].shape[0]
    
    # Run one probe to get n_features and ensure params are valid
    sample_mask = input_root['masks'][0]
    sample_feat = features_from_mask(sample_mask, **kwargs)
    n_features = sample_feat.shape[0]
    dtype = sample_feat.dtype
    
    # 2. Pre-allocate Output Zarr on Disk
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    output_root = zarr.open(output_zarr_path, mode='w')
    output_root.create_dataset(
        'features', 
        shape=(n_samples, n_features), 
        chunks=(1, n_features), 
        dtype=dtype, 
        compressor=compressor,
        overwrite=True
    )
    
    # 3. Define the Atomic Worker Task
    def process_and_write(idx, in_path, out_path, func, params):
        try:
            # Each worker opens its own read-only handle
            in_store = zarr.open(in_path, mode='r')
            mask = in_store['masks'][idx]
            
            # Compute features
            feat = func(mask, **params)
            
            # Write to specific chunk (Thread/Process safe for different chunks)
            out_store = zarr.open(out_path, mode='a')
            out_store['features'][idx] = feat.astype(dtype)
            
            # Explicit cleanup to prevent memory accumulation
            del mask, feat, in_store, out_store
            gc.collect()
            return True
        except Exception as e:
            # Since logging.info often fails in sub-processes, 
            # we return the error to the main process
            return f"Error at index {idx}: {str(e)}"

    # 4. Execute Parallel Loop
    logging.info(f"Launching Joblib with {n_jobs} workers for {n_samples} samples...")
    
    # 'loky' is the safest backend for Joblib on clusters
    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
        delayed(process_and_write)(i, masks_zarr_path, output_zarr_path, features_from_mask, kwargs) 
        for i in range(n_samples)
    )

    # 5. Handle Results and Errors
    errors = [r for r in results if isinstance(r, str)]
    if errors:
        for err in errors:
            logging.error(err)
    
    # 6. Copy Metadata
    output_root.create_dataset('labels', data=input_root['labels'][:])
    output_root.attrs['original_shape'] = input_root['masks'].shape[1:]
    output_root.attrs['kwargs'] = kwargs
    
    logging.info("Feature calculation complete.")



def features_from_dataset_sequential(
    features_from_mask: Callable,
    masks_zarr_path: str,
    output_zarr_path: str,
    **kwargs
):
    # 1. Open Input and Probe Shape
    input_root = zarr.open(masks_zarr_path, mode='r')
    n_samples = input_root['masks'].shape[0]
    
    # Run one probe to get n_features (Order 32 -> 6545)
    sample_mask = input_root['masks'][0]
    sample_feat = features_from_mask(sample_mask, **kwargs)
    n_features = sample_feat.shape[0]
    dtype = sample_feat.dtype
    
    # 2. Pre-allocate Output Zarr on Disk
    # chunks=(1, n_features) means each row is an independent file on disk
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    output_root = zarr.open(output_zarr_path, mode='w')
    output_root.create_dataset(
        'features', 
        shape=(n_samples, n_features), 
        chunks=(1, n_features), 
        dtype=dtype, 
        compressor=compressor,
        overwrite=True
    )
    
    # 3. Define the Atomic "Read-Compute-Write" Task
    #@dask.delayed
    def process_and_write(idx):
        # WORKER-SIDE: Open input
        in_store = zarr.open(masks_zarr_path, mode='r')
        mask = in_store['masks'][idx]
        
        # Compute
        feat = features_from_mask(mask, **kwargs)
        
        # WORKER-SIDE: Open output and write to specific slice
        # Zarr handles concurrent writes to DIFFERENT chunks safely
        out_store = zarr.open(output_zarr_path, mode='a')
        out_store['features'][idx] = feat.astype(dtype)

        # Why is this not shown?
        logging.info(f"Finished computing features for sample {idx+1}")
    
    # 4. Create Task List
    logging.info(f"Computing {n_samples} samples with direct-to-disk writing...")
    tasks = [process_and_write(i) for i in range(n_samples)]
    #dask.compute(*tasks)

    # 6. Copy Metadata
    output_root.create_dataset('labels', data=input_root['labels'][:])
    output_root.attrs['original_shape'] = input_root['masks'].shape[1:]
    output_root.attrs['kwargs'] = kwargs
    
    logging.info("Feature calculation complete.")




def dataset_from_features(
    mask_from_features: Callable,
    features_zarr_path: str,
    output_zarr_path: str,
):
    input_root = zarr.open(features_zarr_path, mode='r')
    
    # --- 1. Inherit Metadata and Setup ---
    kwargs = input_root.attrs.get('kwargs', {})
    target_shape = tuple(input_root.attrs['original_shape'])
    feat_ds = input_root['features']
    leading_shape = feat_ds.shape[:-1] 
    n_total_tasks = int(np.prod(leading_shape))
    
    # --- 2. Pre-allocate Output Zarr ---
    output_root = zarr.open(output_zarr_path, mode='w')
    output_root.attrs.update(input_root.attrs)
    
    # Copy other datasets (labels, etc.)
    for name, item in input_root.arrays():
        if name == 'features': continue
        output_root.array(name, data=item[:], compressor=item.compressor)

    full_output_shape = leading_shape + target_shape
    chunks = (1,) * len(leading_shape) + target_shape
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    
    output_root.create_dataset(
        'masks', 
        shape=full_output_shape, 
        chunks=chunks, 
        dtype=bool, 
        compressor=compressor,
        overwrite=True
    )

    # --- 3. Atomic Worker Task with Checkpoint ---
    def reconstruct_and_write(flat_idx, in_path, out_path, func, shape, params):
        try:
            multi_idx = np.unravel_index(flat_idx, leading_shape)
            
            # Open output to check if already done (Checkpoint)
            out_store = zarr.open(out_path, mode='a')
            # Check if any part of the slice is True (already reconstructed)
            if np.any(out_store['masks'][multi_idx]):
                return True
            
            # Read features
            in_store = zarr.open(in_path, mode='r')
            feat_vec = in_store['features'][multi_idx]
            
            # Compute inverse (Pyramidal Spline reconstruction)
            mask = func(feat_vec, shape, **params)
            
            # Write and Cleanup
            out_store['masks'][multi_idx] = mask.astype(bool)
            
            del mask, feat_vec
            gc.collect()
            return True
        except Exception as e:
            return f"Error at index {flat_idx}: {str(e)}"

    # --- 4. Parallel Execution via Dask Backend ---
    logging.info(f"Reconstructing {n_total_tasks} masks using Dask-backed Joblib...")
    
    # This automatically uses the active Dask Client workers
    results = Parallel(verbose=10)(
        delayed(reconstruct_and_write)(
            i, features_zarr_path, output_zarr_path, mask_from_features, target_shape, kwargs
        ) for i in range(n_total_tasks)
    )

    # Log errors if any
    errors = [r for r in results if isinstance(r, str)]
    if errors:
        for err in errors:
            logging.error(err)
            
    return True


def dataset_from_features_sequential(
    mask_from_features: Callable,
    features_zarr_path: str,
    output_zarr_path: str,
):
    input_root = zarr.open(features_zarr_path, mode='r')
    output_root = zarr.open(output_zarr_path, mode='w')

    # --- 1. Inherit Attributes ---
    # Copies all top-level metadata (e.g., 'original_shape', 'kwargs', etc.)
    output_root.attrs.update(input_root.attrs)
    
    # --- 2. Inherit All Datasets except 'features' ---
    for name, item in input_root.arrays():
        if name == 'features':
            continue
        data = item[:]
        output_root.array(name, data=data, compressor=item.compressor)

    # --- 3. Setup New 'masks' Dataset ---
    feat_ds = input_root['features']
    leading_shape = feat_ds.shape[:-1] 
    n_total_tasks = int(np.prod(leading_shape))
    
    kwargs = input_root.attrs.get('kwargs', {})
    target_shape = tuple(input_root.attrs['original_shape'])
    full_output_shape = leading_shape + target_shape
    chunks = (1,) * len(leading_shape) + target_shape
    
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    output_root.create_dataset(
        'masks', 
        shape=full_output_shape, 
        chunks=chunks, 
        dtype=bool, 
        compressor=compressor,
        overwrite=True
    )

    # --- 4. Parallel Reconstruction ---
    #@dask.delayed
    def reconstruct_and_write(flat_idx):
        multi_idx = np.unravel_index(flat_idx, leading_shape)
        
        # Open inside worker for thread-safety/process-safety
        in_store = zarr.open(features_zarr_path, mode='r')
        feat_vec = in_store['features'][multi_idx]
        
        mask = mask_from_features(feat_vec, target_shape, **kwargs)
        
        out_store = zarr.open(output_zarr_path, mode='a')
        out_store['masks'][multi_idx] = mask.astype(bool)
        return True

    logging.info(f"Reconstructing {n_total_tasks} masks...")
    tasks = [reconstruct_and_write(i) for i in range(n_total_tasks)]
    #dask.compute(*tasks)
    
    return True
