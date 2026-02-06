import logging
import numpy as np
from skimage import measure
from scipy.spatial import cKDTree
import dask
from dask.diagnostics import ProgressBar
import dask.array as da
import psutil
import zarr


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


def dice_matrix_in_memory(M:np.ndarray):
    """
    Computes a Dice similarity matrix for all numpy masks in a folder using 
    vectorized sparse matrix multiplication.
    """
    # Esure the matrix is 2D
    M = M.reshape((M.shape[0], -1))

    # Convert from Boolean (True/False) to Integer (1/0)
    # This ensures the dot product counts overlapping voxels.
    M = M.astype(np.int32)
    
    # 3. Vectorized Intersection Calculation (Matrix Multiplication)
    # Intersections[i, j] = dot_product(mask_i, mask_j)
    # This replaces the nested loop. M.T means M transpose.
    intersection_matrix = M @ M.T
    
    # 4. Compute Dice Score
    # Formula: 2 * (A n B) / (|A| + |B|)
    
    # The diagonal of the intersection matrix represents |A n A|, which is just |A| (the volume)
    volumes = intersection_matrix.diagonal()
    
    # Broadcasting sum: creates a matrix where cell [i,j] = volume[i] + volume[j]
    volumes_sum_matrix = volumes[:, None] + volumes[None, :]
    
    # Avoid division by zero (though volumes shouldn't be 0 for valid masks)
    # If both volumes are 0, Dice is technically 1.0 (empty matches empty), 
    # but usually we handle this based on context. Here we use np.errstate to handle specific cases.
    with np.errstate(divide='ignore', invalid='ignore'):
        dice_matrix = (2 * intersection_matrix) / volumes_sum_matrix
        
    # Handle NaN cases where volumes_sum_matrix might be 0
    dice_matrix = np.nan_to_num(dice_matrix, nan=1.0)

    return dice_matrix

def hausdorff_matrix_in_memory(M, chunk_size = 1000): # (n_subjects, n_voxels)
    # Chunk output to produce less and larger tasks, and less files
    # Otherwise dask takes too long to schedule

    # Convert from Boolean (True/False) to Integer (1/0)
    # This ensures the dot product counts overlapping voxels.
    M = M.astype(np.int32)
    
    n = M.shape[0]
    # Build a list of all index pairs in the sorted list that need computing
    # Since the matrix is symmetric only half needs to be computed
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    # Split the list of index pairs up into chunks
    chunks = [pairs[i:i+chunk_size] for i in range(0, len(pairs), chunk_size)]

    # Compute dice scores for each chunk in parallel
    logging.info("Hausdorff matrix - scheduling tasks..")
    tasks = [
        dask.delayed(_hausdorff_matrix_chunk)(M, chunk) 
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


def dice_matrix_zarr(zarr_path):
    # 1. Connect to Zarr
    d_masks = da.from_zarr(zarr_path, component='masks')
    
    # Flatten
    n_samples = d_masks.shape[0]
    d_masks = d_masks.reshape(n_samples, -1).astype(np.int32)

    # Matrix Multiplication
    intersection_graph = da.dot(d_masks, d_masks.T)

    print(f"Computing Dice matrix...")
    with ProgressBar():
        intersection_matrix = intersection_graph.compute()

    # 7. Rest of the Dice logic...
    volumes = intersection_matrix.diagonal()
    volumes_sum_matrix = volumes[:, None] + volumes[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        dice = (2 * intersection_matrix) / volumes_sum_matrix
        
    return np.nan_to_num(dice, nan=1.0)

def dice_matrix_zarr(zarr_path):
    # 1. Connect
    d_masks = da.from_zarr(zarr_path, component='masks')
    n_samples = d_masks.shape[0]
    
    # 2. Flatten and cast to int32 (4 bytes per voxel)
    # We use int32 because boolean True + True = 2 for intersection
    d_masks_flat = d_masks.reshape(n_samples, -1).astype(np.int32)
    
    # 3. Calculate intersections manually via block-wise map_blocks
    # This prevents the 'factor of 124' rechunking warning
    logging.info("Computing Intersection Matrix...")
    
    # intersection = d_masks_flat @ d_masks_flat.T
    # But we use the governor to ensure it's calculated in pieces
    with ProgressBar():
        # Force Dask to compute the dot product block-by-block
        # to avoid the intermediate 'shuffle' that causes the OOM
        intersections = da.matmul(d_masks_flat, d_masks_flat.T).compute()

    # 4. Sums for Dice Denominator
    logging.info("Computing sums...")
    sums = d_masks_flat.sum(axis=1).compute()
        
    # 5. Final Dice Calculation (NumPy level)
    # Dice = (2 * intersect) / (sum_a + sum_b)
    logging.info("Finalizing Dice values...")
    dice = (2.0 * intersections) / (sums[:, None] + sums[None, :])
    
    # Replace NaNs (if any empty masks exist)
    dice = np.nan_to_num(dice)
    
    return dice





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