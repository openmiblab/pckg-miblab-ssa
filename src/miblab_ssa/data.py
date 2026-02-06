import logging

import numpy as np
from tqdm import tqdm
import zarr

import math


from miblab_ssa import utils


def save_masks_as_zarr(zarr_path, masks: list, labels: list, key='values'):
    # Detect memory automatically
    mem_gb = utils.get_memory_limit()
    
    # We use a safety buffer (e.g., 80% of detected limit) 
    # to avoid OOM (Out Of Memory) errors during the dot product
    # We use half of available memory as limit because some 
    # calculations like DICE load two chunks
    usable_mem_gb = (mem_gb / 4) * 0.8
    
    logging.info(f"Detected {mem_gb:.2f} GB RAM. Using {usable_mem_gb:.2f} GB for chunk calculation.")

    # 1. Initialize Zarr Group
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store, overwrite=True)

    # 2. Probe Metadata
    n_samples = len(masks)
    with np.load(masks[0]) as data:
        spatial_shape = data[key].shape

    # 3. Calculate chunks using the detected usable_mem_gb
    voxels_per_volume = np.prod(spatial_shape)
    bytes_per_volume = voxels_per_volume * 4 # Assuming float32 inflation
    
    samples_per_chunk = max(1, math.floor((usable_mem_gb * 1024**3) / bytes_per_volume))
    samples_per_chunk = min(samples_per_chunk, len(masks))
    
    chunks = (samples_per_chunk,) + spatial_shape
    
    logging.info(f"Calculated chunks: {chunks} ({samples_per_chunk} samples per chunk)")
    logging.info(f"Estimated RAM per chunk (as float32): { (samples_per_chunk * bytes_per_volume) / 1024**2 :.2f} MB")

    # 4. Create Compressed Dataset
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    z_masks = root.create_dataset(
        'masks', 
        shape=(n_samples,) + spatial_shape, 
        chunks=chunks, 
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

