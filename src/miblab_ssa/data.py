import logging

import numpy as np
from tqdm import tqdm
import zarr


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

