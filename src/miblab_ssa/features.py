from typing import Callable, Optional, List
from dask.distributed import get_worker
from datetime import datetime
import logging
from tqdm import tqdm
import zarr
import numpy as np
import gc
from joblib import Parallel, delayed
import pandas as pd
import os
import numpyradiomics as npr
from copy import deepcopy

from miblab_ssa.metrics import dice_coefficient, surface_distances



def _reconstruct_and_save_worker(
    smooth_mask_func, 
    input_zarr_path, 
    output_zarr_path, 
    sample_idx, 
    target_idx, 
    min_order, 
    max_order, 
    n_samples,
    kwargs
):
    """Worker function to reconstruct masks and save directly to a 5D Zarr."""
    worker = get_worker()
    worker_id = f"Worker {worker.name}"
    try:
        root_in = zarr.open(input_zarr_path, mode='r')
        root_out = zarr.open(output_zarr_path, mode='a')
        
        orig_mask = root_in['masks'][sample_idx]
        
        # 1. Save reconstructed versions
        for n in range(min_order, max_order + 1):
            kwargs['order'] = n
            reconstructed = smooth_mask_func(orig_mask, **kwargs)
            order_idx = n - min_order
            root_out['reconstructed_masks'][target_idx, order_idx] = reconstructed
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {worker_id} finished order {n} / {max_order} of sample {target_idx + 1} / {n_samples}", flush=True)
        
        # 2. Save Ground Truth as the final index in the order dimension
        # The index is (max - min + 1), which is the slot immediately after the last order
        gt_idx = max_order - min_order + 1
        root_out['reconstructed_masks'][target_idx, gt_idx] = orig_mask
        
        return target_idx, True
    except Exception as e:
        print(f"Error processing sample index {sample_idx}: {e}")
        return target_idx, False


def save_reconstructed_masks(
    smooth_mask_func: Callable = None,
    dataset_zarr_path: str = None,
    output_zarr_path: str = None,
    target_labels: Optional[List] = None,
    min_order: int = 2,
    max_order: int = 36,
    n_jobs: int = -1,
    **kwargs
):
    """
    Reconstructs masks and saves them as a 5D dataset: (N_targets, N_orders + 1, Z, Y, X).
    The final index in the N_orders dimension is the original Ground Truth.
    """
    root_in = zarr.open(dataset_zarr_path, mode='r')
    all_labels = [str(l) for l in root_in['labels'][:]]
    
    target_indices = []
    valid_target_labels = []

    if target_labels is None:
        target_indices = list(range(len(all_labels)))
        valid_target_labels = all_labels
    else:
        for label in target_labels:
            if str(label) in all_labels:
                target_indices.append(all_labels.index(str(label)))
                valid_target_labels.append(str(label))
            else:
                logging.warning(f"Label {label} not found in input dataset. Skipping.")

    n_targets = len(target_indices)
    if n_targets == 0:
        return

    # --- Setup the 5D Output Zarr Array ---
    # We add +1 to the order dimension to accommodate the Ground Truth mask
    num_orders_plus_gt = (max_order - min_order + 1) + 1
    spatial_shape = root_in['masks'].shape[1:] 
    output_shape = (n_targets, num_orders_plus_gt, *spatial_shape)
    output_chunks = (1, 1, *spatial_shape)
    
    root_out = zarr.open(output_zarr_path, mode='w') 
    
    root_out.create_dataset(
        'masks', 
        shape=output_shape, 
        chunks=output_chunks, 
        dtype=root_in['masks'].dtype
    )
    
    # Metadata: Fixed-length strings or VLenUTF8 for labels
    root_out.create_dataset('labels', data=np.array(valid_target_labels))
    
    # Orders metadata: append a special value (e.g., -1 or 0) to represent Ground Truth
    order_values = np.arange(min_order, max_order + 1)
    order_values_with_gt = np.append(order_values, -1) # -1 signals "Original/GT"
    root_out.attrs['saved_steps'] = order_values_with_gt

    # --- Parallel Processing ---
    Parallel(n_jobs=n_jobs)(
        delayed(_reconstruct_and_save_worker)(
            smooth_mask_func, 
            dataset_zarr_path, 
            output_zarr_path, 
            orig_idx, 
            target_idx, 
            min_order, 
            max_order, 
            n_targets,
            deepcopy(kwargs)
        ) for target_idx, orig_idx in enumerate(target_indices)
    )
        
    return root_out



def _process_sample_orders(
    smooth_mask_func, 
    zarr_path, 
    sample_idx, 
    min_order, 
    max_order, 
    n_samples, 
    kwargs,
):
    """Worker function to compute both Dice and Hausdorff for all orders."""
    try:
        logging.info(f"Processing sample {sample_idx + 1} / {n_samples}")
        root = zarr.open(zarr_path, mode='r')
        orig_mask = root['masks'][sample_idx]
        
        dice_results = {}
        haus_results = {}
        
        for n in range(min_order, max_order + 1):
            kwargs['order'] = n
            reconstructed = smooth_mask_func(orig_mask, **kwargs)
            
            # Calculate both metrics
            dice_results[n] = dice_coefficient(orig_mask, reconstructed)
            haus_results[n], _ = surface_distances(orig_mask, reconstructed)
        
        logging.info(f"Finished processing sample {sample_idx + 1} / {n_samples}")
        return sample_idx, dice_results, haus_results
    except Exception as e:
        print(f"Error processing sample {sample_idx}: {e}")
        return sample_idx, None, None
    

def reconstruction_fidelity(
    smooth_mask_func: Callable = None,
    dataset_zarr_path: str = None,
    dice_csv_path: str = None,
    hausdorff_csv_path: str = None,
    min_order: int = 2,
    max_order: int = 36,
    n_jobs: int = -1,
    **kwargs
):
    """
    Builds two CSVs (Dice and Hausdorff) across spectral reconstruction orders.
    """
    # 1. Setup dimensions and labels
    root = zarr.open(dataset_zarr_path, mode='r')
    n_samples = root['masks'].shape[0]
    labels = [str(l) for l in root['labels'][:]]

    # 2. Parallel Processing
    print(f"Evaluating reconstruction fidelity for {n_samples} samples up to order {max_order}...")
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_sample_orders)(
            smooth_mask_func, 
            dataset_zarr_path, 
            i, 
            min_order, 
            max_order, 
            n_samples,
            deepcopy(kwargs),
        ) for i in tqdm(range(n_samples))
    )

    # 3. Aggregate results into matrices
    # We only need (max_order - min_order + 1) columns
    num_orders = max_order - min_order + 1
    dice_matrix = np.full((n_samples, num_orders), np.nan)
    haus_matrix = np.full((n_samples, num_orders), np.nan)
    
    for sample_idx, d_results, h_results in results:
        if d_results is not None:
            # Loop only through the orders that were actually computed
            for i, order in enumerate(range(min_order, max_order + 1)):
                dice_matrix[sample_idx, i] = d_results[order]
                haus_matrix[sample_idx, i] = h_results[order]

    # 4. Save to CSVs
    order_cols = [f"Order_{n}" for n in range(min_order, max_order + 1)]
    
    # Save Dice
    df_dice = pd.DataFrame(dice_matrix, index=labels, columns=order_cols)
    df_dice.index.name = "Sample_ID"
    df_dice.to_csv(dice_csv_path)
    
    # Save Hausdorff
    df_haus = pd.DataFrame(haus_matrix, index=labels, columns=order_cols)
    df_haus.index.name = "Sample_ID"
    df_haus.to_csv(hausdorff_csv_path)

    print(f"Fidelity CSVs saved:\n - {dice_csv_path}\n - {hausdorff_csv_path}")
    
    return df_dice, df_haus


def features_from_dataset(
    features_from_mask: Callable = None,
    masks_zarr_path: str = None,
    output_zarr_path: str = None,
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



def _process_single_subject(i, mask_zarr_path, n_metrics_steps):
    recon = zarr.open(mask_zarr_path, mode='r')
    masks = recon['masks']
    
    # GT is always at the last index (-1)
    original = masks[i, -1, ...].astype(np.float32)
    
    dice_row = np.zeros(n_metrics_steps, dtype=np.float32)
    haus_row = np.zeros(n_metrics_steps, dtype=np.float32)
    
    for j in range(n_metrics_steps):
        # This will now correctly process index 0 (Mean) 
        # through index n-2 (Last reconstruction step)
        recon_mask = masks[i, j, ...].astype(np.float32)
        
        dice_row[j] = 1.0 - dice_coefficient(original, recon_mask)
        hd, _ = surface_distances(original, recon_mask)
        haus_row[j] = hd
            
    return dice_row, haus_row

def recon_error(
    dataset_zarr_path: str = None, 
    dice_csv_path: str = None, 
    hausdorff_csv_path: str = None, 
    n_jobs: int = -1,
):
    # 1. Load metadata
    recon = zarr.open(dataset_zarr_path, mode='r')
    labels = recon['labels'][:].astype(str)
    cols = recon.attrs['saved_steps']
    n_samples = recon['masks'].shape[0]
    
    # We want to calculate error for every column EXCEPT the last one (GT)
    # If cols is ["Mean", 1, 5, 10, "GT"], n_metrics_steps is 4
    n_metrics_steps = len(cols) - 1

    # 2. Parallel Dispatch
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_subject)(i, dataset_zarr_path, n_metrics_steps) 
        for i in tqdm(range(n_samples), desc="Parallel Metrics Calculation")
    )

    # 3. Reconstruct matrices
    dice_matrix = np.array([r[0] for r in results])
    haus_matrix = np.array([r[1] for r in results])

    # 4. Save with correct column headers (everything except 'GT')
    step_names = cols[:-1]
    
    pd.DataFrame(dice_matrix, index=labels, columns=step_names).to_csv(dice_csv_path)
    pd.DataFrame(haus_matrix, index=labels, columns=step_names).to_csv(hausdorff_csv_path)
    
    print(f"Metrics saved to {dice_csv_path} and {hausdorff_csv_path}")



def dataset_from_features(
    mask_from_features_func: Callable = None,
    features_zarr_path: str = None,
    dataset_zarr_path: str = None,
):
    input_root = zarr.open(features_zarr_path, mode='r')
    
    # --- 1. Inherit Metadata and Setup ---
    kwargs = input_root.attrs.get('kwargs')
    target_shape = tuple(input_root.attrs['original_shape'])
    feat_ds = input_root['features']
    leading_shape = feat_ds.shape[:-1] 
    n_total_tasks = int(np.prod(leading_shape))
    
    # --- 2. Pre-allocate Output Zarr ---
    output_root = zarr.open(dataset_zarr_path, mode='w')
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
            i, features_zarr_path, dataset_zarr_path, mask_from_features_func, target_shape, kwargs
        ) for i in range(n_total_tasks)
    )

    # Log errors if any
    errors = [r for r in results if isinstance(r, str)]
    if errors:
        for err in errors:
            logging.error(err)
            
    return True



def _extract_single_mask(zarr_path, c, r):
    """Helper function to process a single mask in parallel."""
    try:
        # Re-open zarr in each worker to ensure thread/process safety
        root = zarr.open(zarr_path, mode='r')
        mask_3d = root['masks'][c, r, :, :, :]
        features = npr.shape(mask_3d)
        return c, r, features
    except Exception as e:
        return c, r, None

def dataset_shapes(
    dataset_zarr_path: str = None, 
    dir_csv: str = None,
    col_names: str = 'coeffs',
    row_names: str = 'components',
    n_jobs: int = -1
):
    """
    Extracts radiomics shape features in parallel using joblib.
    """
    # 1. Setup Data and Dimensions
    root = zarr.open(dataset_zarr_path, mode='r')
    n_cols, n_rows = root['masks'].shape[0], root['masks'].shape[1]
    cols_list = root.attrs[col_names]
    os.makedirs(dir_csv, exist_ok=True)

    # 2. Parallel Extraction
    print(f"Parallel extraction for {n_cols * n_rows} masks using {n_jobs} cores...")
    
    # Generate task list (all combinations of c and r)
    tasks = [(c, r) for c in range(n_cols) for r in range(n_rows)]
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_extract_single_mask)(dataset_zarr_path, c, r) 
        for c, r in tqdm(tasks)
    )

    # 3. Aggregate Results into Grids
    feature_grids = {}

    for c, r, features in results:
        if features is None:
            continue
            
        # Initialize grids on the first successful result
        if not feature_grids:
            for feat_name in features.keys():
                feature_grids[feat_name] = np.full((n_rows, n_cols), np.nan)

        for feat_name, value in features.items():
            feature_grids[feat_name][r, c] = value

    # 4. Save Grids to CSV
    row_indices = np.arange(1, n_rows + 1)
    for feat_name, grid in feature_grids.items():
        df = pd.DataFrame(grid, index=row_indices, columns=cols_list)
        df.index.name = row_names
        
        file_path = os.path.join(dir_csv, f"{feat_name}.csv")
        df.to_csv(file_path)

    print(f"Done. Files saved to {dir_csv}")


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



