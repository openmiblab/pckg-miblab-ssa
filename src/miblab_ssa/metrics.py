import logging
import numpy as np
from skimage import measure
from scipy.spatial import cKDTree
import dask
from dask.diagnostics import ProgressBar
import dask.array as da
import psutil
import zarr
import numpy as np
import logging
from multiprocessing import Pool
from tqdm import tqdm


def dice_coefficient(vol_a, vol_b):
    """
    Compute the Dice similarity coefficient between two binary masks.

    Parameters
    ----------
    mask1 : np.ndarray
        First binary mask (values should be 0 or 1).
    mask2 : np.ndarray
        Second binary mask (values should be 0 or 1).

    Returns
    -------
    float
        Dice coefficient, ranging from 0 (no overlap) to 1 (perfect overlap).

    Notes
    -----
    The Dice coefficient is defined as:
        Dice = 2 * |A ∩ B| / (|A| + |B|)
    """
    vol_a = vol_a.astype(bool)
    vol_b = vol_b.astype(bool)
    intersection = np.logical_and(vol_a, vol_b).sum()
    size_a = vol_a.sum()
    size_b = vol_b.sum()
    if size_a + size_b == 0:
        return 1.0
    return 2.0 * intersection / (size_a + size_b)

def surface_distances(vol_a, vol_b, spacing=(1.0,1.0,1.0)):
    """
    Compute surface distances (Hausdorff and mean) between two binary volumes.
    Args:
      vol_a, vol_b: binary 3D arrays
      spacing: voxel spacing (dz,dy,dx)
    Returns:
      hausdorff, mean_dist
    """
    # extract meshes
    verts_a, faces_a, _, _ = measure.marching_cubes(vol_a.astype(np.uint8), level=0.5, spacing=spacing)
    verts_b, faces_b, _, _ = measure.marching_cubes(vol_b.astype(np.uint8), level=0.5, spacing=spacing)

    # build kd-trees
    tree_a = cKDTree(verts_a)
    tree_b = cKDTree(verts_b)

    # distances from A→B and B→A
    d_ab, _ = tree_b.query(verts_a, k=1)
    d_ba, _ = tree_a.query(verts_b, k=1)

    hausdorff = max(d_ab.max(), d_ba.max())
    mean_dist = 0.5 * (d_ab.mean() + d_ba.mean())
    return hausdorff, mean_dist



import numpy as np
import dask
import logging
from dask.diagnostics import ProgressBar

def load_mask_npz(path, key):
    """Simple loader for a single NPZ file."""
    with np.load(path) as data:
        # Load as bool to keep the initial transfer lean
        return data[key].astype(bool)

def dice_matrix_npz(mask_paths, key='values'):
    """
    NPZ + Dask Delayed Baseline:
    1. Uses dask.delayed to parallelize the loading of NPZ files.
    2. Computes the list into a single NumPy array.
    3. Performs the Dice calculation in-memory.
    """
    n_samples = len(mask_paths)
    logging.info(f"Delayed Baseline: Building graph for {n_samples} masks...")

    # 1. Create a list of delayed tasks
    lazy_masks = [dask.delayed(load_mask_npz)(p, key) for p in mask_paths]

    # 2. Parallel Compute into RAM
    # This triggers the actual loading across your available workers/threads
    logging.info("Executing parallel load into RAM...")
    with ProgressBar():
        # dask.compute returns a list of NumPy arrays
        mask_list = dask.compute(*lazy_masks)

    # 3. Stack and Flatten (N, D, H, W) -> (N, Voxels)
    logging.info("Stacking and flattening masks...")
    masks_flat = np.stack(mask_list).reshape(n_samples, -1)
    del mask_list # Clear the list of individual arrays

    # 4. Math: In-memory Volumes and Intersections
    logging.info("Calculating volumes...")
    volumes = masks_flat.sum(axis=1).astype(np.float32)

    logging.info("Computing intersections (np.dot)...")
    # We cast to float32 right at the multiplication step
    intersections = np.dot(masks_flat.astype(np.float32), 
                           masks_flat.T.astype(np.float32))

    # 5. Final Dice Calculation
    logging.info("Finalizing Dice matrix...")
    v_sum = volumes[:, None] + volumes[None, :]
    dice = (2 * intersections) / v_sum
    
    return np.nan_to_num(dice, nan=1.0)




def hausdorff_matrix_npz(mask_paths, key='values', chunk_size = 1000): # (n_subjects, n_voxels)
    # Chunk output to produce less and larger tasks, and less files
    # Otherwise dask takes too long to schedule

    n_samples = len(mask_paths)
    logging.info(f"Delayed Baseline: Building graph for {n_samples} masks...")

    # 1. Create a list of delayed tasks
    lazy_masks = [dask.delayed(load_mask_npz)(p, key) for p in mask_paths]

    # 2. Parallel Compute into RAM
    # This triggers the actual loading across your available workers/threads
    logging.info("Executing parallel load into RAM...")
    with ProgressBar():
        # dask.compute returns a list of NumPy arrays
        mask_list = dask.compute(*lazy_masks)

    # 3. Stack and Flatten (N, D, H, W) -> (N, Voxels)
    logging.info("Stacking and flattening masks...")
    masks_flat = np.stack(mask_list).reshape(n_samples, -1)
    del mask_list # Clear the list of individual arrays

    # Convert from Boolean (True/False) to Integer (1/0)
    # This ensures the dot product counts overlapping voxels.
    masks_flat = masks_flat.astype(np.float32)
    
    n = masks_flat.shape[0]
    # Build a list of all index pairs in the sorted list that need computing
    # Since the matrix is symmetric only half needs to be computed
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    # Split the list of index pairs up into chunks
    chunks = [pairs[i:i+chunk_size] for i in range(0, len(pairs), chunk_size)]

    # Compute dice scores for each chunk in parallel
    logging.info("Hausdorff matrix - scheduling tasks..")
    tasks = [
        dask.delayed(_hausdorff_matrix_chunk)(masks_flat, chunk) 
        for chunk in chunks
    ]
    logging.info("Hausdorff matrix - computing tasks..")
    with ProgressBar():
        chunks = dask.compute(*tasks)

    # Gather up all the chunks to build one matrix
    logging.info(f"Hausdorff matrix - building matrix..")
    haus_matrix = np.zeros((n, n), dtype=np.float32)
    for chunk in chunks:
        for (i, j), haus_ij in chunk.items():
            haus_matrix[i, j] = haus_ij
            haus_matrix[j, i] = haus_ij

    return haus_matrix


def _hausdorff_matrix_chunk(M, pairs):
    chunk = {}
    for (i,j) in pairs:
        # Load masks
        mask_i = M[i, ...].astype(bool)
        mask_j = M[j, ...].astype(bool)
        # Compute metrics
        haus_ij, _ = surface_distances(mask_i, mask_j)
        # Add to results
        chunk[(i, j)] = haus_ij
    return chunk



def dice_matrix_zarr(zarr_path, block_size=100):
    d_masks = da.from_zarr(zarr_path, component='masks')
    n_samples = d_masks.shape[0]
    
    # Pre-calculate volumes (1D is always safe)
    volumes = d_masks.sum(axis=(1, 2, 3)).compute().astype(np.float32)
    
    # Flatten masks (N, Voxels)
    d_masks_flat = d_masks.reshape(n_samples, -1).astype(np.float32)
    
    # Initialize the result matrix in NumPy (only ~5MB for 1108x1108)
    intersections = np.zeros((n_samples, n_samples), dtype=np.float32)
    
    logging.info(f"Computing Dice in blocks of {block_size} rows...")
    
    # We process 100 rows at a time. 
    # This keeps the Dask Task Graph small and manageable for the Scheduler.
    for i in range(0, n_samples, block_size):
        end_i = min(i + block_size, n_samples)
        
        # This dot product is (Block, Voxels) @ (Voxels, N)
        # It creates ~110k tasks instead of 1.2M tasks.
        block_intersections = da.matmul(d_masks_flat[i:end_i], d_masks_flat.T).compute()
        intersections[i:end_i, :] = block_intersections
        
    v_sum = volumes[:, None] + volumes[None, :]
    dice = (2 * intersections) / v_sum
    return np.nan_to_num(dice, nan=1.0)



def hausdorff_matrix_zarr(zarr_path: str):
    # 1. Open metadata
    z_root = zarr.open(zarr_path, mode='r')
    n = z_root['masks'].shape[0]

    logging.info(f"Hausdorff matrix: Scheduling {n} row tasks...")

    # 2. Schedule one task per row
    # Each task computes the distances for row i from [i to n]
    tasks = [
        dask.delayed(_compute_hausdorff_row)(zarr_path, i, n) 
        for i in range(n)
    ]

    # 3. Compute
    logging.info(f"Hausdorff matrix: Computing {n} row tasks...")
    with ProgressBar():
        rows = dask.compute(*tasks)

    # 4. Assemble
    # 'rows' is now a list of arrays of varying lengths
    haus_matrix = np.zeros((n, n), dtype=np.float32)
    for i, row_values in enumerate(rows):
        # row_values contains distances for [i, i+1, ... n-1]
        haus_matrix[i, i:] = row_values
        haus_matrix[i:, i] = row_values # Mirror to lower triangle

    return haus_matrix

def _compute_hausdorff_row(zarr_path, i, n):
    """Computes all distances for a single row starting from the diagonal."""
    z_masks = zarr.open(zarr_path, mode='r')['masks']
    
    # Load mask_i once for the entire row
    mask_i = z_masks[i].astype(bool)
    
    # Pre-allocate result for the partial row
    row_len = n - i
    row_results = np.zeros(row_len, dtype=np.float32)
    
    for idx, j in enumerate(range(i, n)):
        if i == j:
            row_results[idx] = 0.0
            continue
            
        mask_j = z_masks[j].astype(bool)
        h_val, _ = surface_distances(mask_i, mask_j)
        row_results[idx] = h_val
        
    return row_results