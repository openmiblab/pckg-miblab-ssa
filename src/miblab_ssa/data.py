import logging

import numpy as np
from tqdm import tqdm
import zarr




def save_masks_as_zarr(zarr_path, masks: list, labels: list, max_ram_mb=1024, key='values'):
    """
    Saves a list of volumetric masks into a chunked Zarr format, optimized for 
    subsequent Dask parallel computations (like Dice matrices or PCA).

    Args:
        zarr_path (str): Path to the output .zarr directory.
        masks (list): List of paths to .npz files containing mask data, 
                      each saved as 3D arrays of equal shape.
        labels (list): Corresponding identifiers for each mask.
        max_ram_mb (int): RAM limit in MB used to determine the chunk size. 
                          Should match your HPC worker memory allocation.
        key (str): The key in the .npz file containing the volume data.
    """
    # 1. Initialize Zarr Group
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store, overwrite=True)

    # 2. Probe Metadata
    # We load the first file just to get dimensions
    n_samples = len(masks)
    with np.load(masks[0]) as data:
        spatial_shape = data[key].shape
    
    n_voxels_per_sample = np.prod(spatial_shape)

    # 3. Calculate Ideal Chunking
    # We calculate based on the RAM limit. Even though storage is bool (1 byte),
    # we chunk for the future int32 math (4 bytes) to prevent downstream OOMs.
    limit_bytes = max_ram_mb * 1024 * 1024
    bytes_per_sample_math = n_voxels_per_sample * 4  # Future-proofing for float32 operations
    n_samples_per_chunk = max(1, int(limit_bytes // bytes_per_sample_math))

    logging.info(f"Storage: {n_samples} samples, Shape {spatial_shape}")
    logging.info(f"Chunking: {n_samples_per_chunk} samples per block based on {max_ram_mb}MB limit.")

    # 4. Create Compressed Dataset
    # Using Blosc + Zstd is the 'Gold Standard' for medical volumes
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    
    z_masks = root.create_dataset(
        'masks', 
        shape=(n_samples,) + spatial_shape, 
        chunks=(n_samples_per_chunk,) + spatial_shape, 
        dtype=bool,
        compressor=compressor 
    )

    # 5. Save Metadata
    root.array('labels', labels)

    # 6. Stream Data to Disk
    # This loop is RAM-efficient as it only holds one volume at a time
    for i, mask_path in tqdm(enumerate(masks), total=n_samples, desc="Writing Zarr"):
        with np.load(mask_path) as data:
            # We explicitly cast to bool here to save space on disk
            z_masks[i] = data[key].astype(bool)

    logging.info(f"Zarr saved successfully to {zarr_path}")