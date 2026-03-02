import logging
import numpy as np
from skimage import measure
from scipy.spatial import cKDTree
import dask
from dask.diagnostics import ProgressBar
import dask.array as da
import zarr
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
    if (np.sum(vol_a)==0) or (np.sum(vol_b)==0):
        return 0, 0
    
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




def dice_matrix(zarr_path, block_size=100):
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
    for i in tqdm(range(0, n_samples, block_size)):
        end_i = min(i + block_size, n_samples)
        
        # This dot product is (Block, Voxels) @ (Voxels, N)
        # It creates ~110k tasks instead of 1.2M tasks.
        block_intersections = da.matmul(d_masks_flat[i:end_i], d_masks_flat.T).compute()
        intersections[i:end_i, :] = block_intersections
        
    v_sum = volumes[:, None] + volumes[None, :]
    dice = (2 * intersections) / v_sum
    return np.nan_to_num(dice, nan=1.0)



def hausdorff_matrix(zarr_path: str):
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

    logging.info(f"Hausdorff matrix: finished computing row {i} of {n}.")
        
    return row_results