import logging

import numpy as np
from tqdm import tqdm
import zarr




def save_masks_as_zarr(zarr_path, masks: list, labels: list, key='values'):
    """
    Saves a list of volumetric masks into a chunked Zarr format, optimized for 
    subsequent Dask parallel computations (like Dice matrices or PCA).

    Args:
        zarr_path (str): Path to the output .zarr directory.
        masks (list): List of paths to .npz files containing mask data, 
                      each saved as 3D arrays of equal shape.
        labels (list): Corresponding identifiers for each mask.
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

    # 4. Create Compressed Dataset
    # Using Blosc + Zstd is the 'Gold Standard' for medical volumes
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
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
            # We explicitly cast to bool here to save space on disk
            z_masks[i] = data[key].astype(bool)

    logging.info(f"Zarr saved successfully to {zarr_path}")