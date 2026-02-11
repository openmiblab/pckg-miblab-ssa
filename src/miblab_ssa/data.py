import logging

import numpy as np
from tqdm import tqdm
import zarr
import dask.array as da

def save_masks_as_zarr(zarr_path, masks: list, labels: list, key='values'):

    # 1. Probe Metadata
    n_samples = len(masks)
    with np.load(masks[0]) as data:
        spatial_shape = data[key].shape

    # 2. Initialize Zarr Group
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store, overwrite=True)
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)

    # 4. Create Compressed Dataset
    z_masks = root.create_dataset(
        'masks', 
        shape=(n_samples,) + spatial_shape, 
        chunks=(1,) + spatial_shape, 
        dtype=bool,
        compressor=compressor 
    )

    # 5. Save Metadata
    root.array('labels', labels)

    # 6. Stream Data to Disk
    for i, mask_path in tqdm(enumerate(masks), total=n_samples, desc="Writing Zarr"):
        with np.load(mask_path) as data:
            z_masks[i] = data[key].astype(bool)

    logging.info(f"Zarr saved successfully to {zarr_path}")



def masks_from_zarr(zarr_path, target_labels):
    """
    Returns a list of dask arrays for the specified labels.
    
    Args:
        zarr_path (str): Path to the Zarr store created by save_masks_as_zarr.
        target_labels (list): List of strings (e.g., ['PatientA_L', 'PatientB_R']).
        
    Returns:
        list: A list of dask.array objects, one for each found label.
    """
    # 1. Open the Zarr group (read-only)
    root = zarr.open(zarr_path, mode='r')
    
    # 2. Load the full dask array for the 'masks' dataset
    # This doesn't load data into RAM; it just maps the chunks
    all_masks_dask = da.from_zarr(root['masks'])
    
    # 3. Retrieve and map the labels
    # We cast to string because Zarr might store them as object/bytes
    stored_labels = [str(l) for l in root['labels'][:]]
    label_to_idx = {label: i for i, label in enumerate(stored_labels)}
    
    # 4. Filter and collect dask arrays
    requested_dask_arrays = []
    
    for label in target_labels:
        if label in label_to_idx:
            idx = label_to_idx[label]
            # Slice the dask array (this is still a lazy operation)
            requested_dask_arrays.append(all_masks_dask[idx])
        else:
            logging.warning(f"Label '{label}' not found in Zarr store.")
            
    return requested_dask_arrays
