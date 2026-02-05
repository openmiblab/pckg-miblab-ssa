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




def get_chunk_size(shape, dtype, max_chunk_size_mb=128):
    # 1. Dynamically get bytes per voxel
    bytes_per_voxel = np.dtype(dtype).itemsize

    # shape[-1] is the number of features (voxels)
    n_features = shape[-1]
    
    # 2. Convert MB to Bytes
    max_bytes = max_chunk_size_mb * 1024 * 1024
    bytes_per_sample = n_features * bytes_per_voxel

    # 3. Calculate samples and ensure it's at least 1
    # We use int() because rechunk requires integers
    n_samples_per_chunk = int(max_bytes // bytes_per_sample)
    
    return max(1, n_samples_per_chunk)


def dice_matrix_zarr(zarr_path, max_ram=500):
    # max_ram in MB is the RAM memory occupied by the computation
    # per worker. This is not counting system overhead so make sure 
    # to leave a margin for that. 

    # 1. Connect to Zarr
    d_masks = da.from_zarr(zarr_path, component='masks')

    # 2. Flatten Spatial Dimensions
    # It is usually safer to reshape BEFORE rechunking 
    # so we know the exact size of the feature dimension.
    n_samples = d_masks.shape[0]
    d_masks = d_masks.reshape(n_samples, -1)
    
    # 3. Determine Chunk Size
    # We use np.int32 because that's what we cast to in step 4
    max_chunk_size = max_ram / 2
    n_samples_per_chunk = get_chunk_size(d_masks.shape, np.int32, max_chunk_size)

    # 4. Apply Chunking AND Casting
    # CRITICAL: {1: -1} ensures the feature dimension is not split.
    # This makes the dot product significantly faster.
    d_masks = d_masks.rechunk({0: n_samples_per_chunk, 1: -1}).astype(np.int32)

    # 5. Matrix Multiplication (Lazy)
    # This computes the dot product: Matrix (N, F) @ Matrix (F, N) = (N, N)
    intersection_graph = d_masks @ d_masks.T

    print(f"Computing {n_samples}x{n_samples} Dice matrix...")
    with ProgressBar():
        # This triggers the parallel computation
        intersection_matrix = intersection_graph.compute()

    # 6. Compute Dice Score
    # Dice = (2 * Intersection) / (Vol_A + Vol_B)
    volumes = intersection_matrix.diagonal()
    volumes_sum_matrix = volumes[:, None] + volumes[None, :]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        dice = (2 * intersection_matrix) / volumes_sum_matrix
        
    return np.nan_to_num(dice, nan=1.0)



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