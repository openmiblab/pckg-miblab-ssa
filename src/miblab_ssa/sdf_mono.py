import numpy as np
from scipy.ndimage import distance_transform_edt
from itertools import product
from sklearn.linear_model import Ridge

def compute_implicit_coeffs(mask, order=6, n_samples=50000):
    """
    Fits an implicit surface using narrow-band sampling for high accuracy.
    Lower order (4-8) is often more stable than high order (10+).
    """
    # 1. Compute SDF
    dist_outside = distance_transform_edt(1 - mask)
    dist_inside = distance_transform_edt(mask)
    sdf = dist_outside - dist_inside

    # 2. Define Coordinate System (Relative to Object)
    coords = np.argwhere(mask)
    min_c = coords.min(axis=0)
    max_c = coords.max(axis=0)
    center = (min_c + max_c) / 2.0
    # Use max extent for uniform scaling (preserves aspect ratio)
    scale = (max_c - min_c).max() / 2.0 * 1.1  # 10% padding

    # 3. Intelligent Sampling (The Key Fix)
    # Instead of zooming the grid, we pick specific points to learn from.
    
    # Grid of all voxel indices
    nz, ny, nx = mask.shape
    
    # Strategy: Pick mostly points near the boundary (narrow band)
    # plus some points inside and outside to define the bulk.
    boundary_mask = np.abs(sdf) < 2.0  # Points within 2 pixels of surface
    inside_mask = (sdf < -2.0)         # Deep inside
    outside_mask = (sdf > 2.0)         # Far outside

    # Get indices
    idx_boundary = np.argwhere(boundary_mask)
    idx_inside = np.argwhere(inside_mask)
    idx_outside = np.argwhere(outside_mask)

    # Sampling counts
    n_bound = int(n_samples * 0.5)  # 50% samples on boundary
    n_in = int(n_samples * 0.25)    # 25% inside
    n_out = int(n_samples * 0.25)   # 25% outside

    # Helper to safely sample
    def sample_indices(indices, n):
        if len(indices) == 0: return indices
        # If we have fewer points than requested, take them all
        if len(indices) < n: return indices 
        choice = np.random.choice(len(indices), n, replace=False)
        return indices[choice]

    s_bound = sample_indices(idx_boundary, n_bound)
    s_in = sample_indices(idx_inside, n_in)
    s_out = sample_indices(idx_outside, n_out)

    # Combine samples
    all_indices = np.vstack([s_bound, s_in, s_out])
    
    # 4. Extract Coordinates and Values
    # Z, Y, X (indices)
    z_idx, y_idx, x_idx = all_indices[:, 0], all_indices[:, 1], all_indices[:, 2]
    
    # Normalize to [-1, 1]
    Z_norm = (z_idx - center[0]) / scale
    Y_norm = (y_idx - center[1]) / scale
    X_norm = (x_idx - center[2]) / scale
    
    Values = sdf[z_idx, y_idx, x_idx]

    # 5. Build Design Matrix
    # Basis: x^i * y^j * z^k
    # Pre-calculating powers is faster than looping
    basis_cols = []
    for i, j, k in product(range(order + 1), repeat=3):
        if i + j + k <= order:
            col = (X_norm**i) * (Y_norm**j) * (Z_norm**k)
            basis_cols.append(col)
    
    A = np.column_stack(basis_cols)

    # 6. Fit
    clf = Ridge(alpha=1e-4) # Small regularization
    clf.fit(A, Values)

    return clf.coef_, center, scale, order

def reconstruct_implicit(coeffs, shape, center, scale, order):
    """
    Reconstructs the volume. 
    Optimization: Only evaluate within the bounding box of the object.
    """
    vol = np.zeros(shape, dtype=np.uint8)
    
    # Determine Bounding Box in indices (where normalized coords are roughly [-1, 1])
    # Evaluating the polynomial over the entire 301^3 volume is wasteful and slow.
    pad = int(scale * 1.1)
    z_min, z_max = max(0, int(center[0]-pad)), min(shape[0], int(center[0]+pad))
    y_min, y_max = max(0, int(center[1]-pad)), min(shape[1], int(center[1]+pad))
    x_min, x_max = max(0, int(center[2]-pad)), min(shape[2], int(center[2]+pad))

    # Generate grid only for the ROI
    z_idx, y_idx, x_idx = np.meshgrid(
        np.arange(z_min, z_max),
        np.arange(y_min, y_max),
        np.arange(x_min, x_max),
        indexing='ij'
    )
    
    # Normalize
    Z = (z_idx - center[0]) / scale
    Y = (y_idx - center[1]) / scale
    X = (x_idx - center[2]) / scale
    
    # Flatten ROI
    Xf, Yf, Zf = X.flatten(), Y.flatten(), Z.flatten()
    
    # Evaluate
    recon_sdf = np.zeros_like(Xf)
    col_idx = 0
    for i, j, k in product(range(order + 1), repeat=3):
        if i + j + k <= order:
            weight = coeffs[col_idx]
            recon_sdf += weight * (Xf**i) * (Yf**j) * (Zf**k)
            col_idx += 1
            
    # Threshold
    # SDF < 0 is inside
    mask_roi = (recon_sdf < 0)
    
    # Place back into full volume
    vol[z_min:z_max, y_min:y_max, x_min:x_max] = mask_roi.reshape(z_idx.shape)
    
    return vol

import numpy as np
from itertools import product

def reconstruct_implicit_fast(coeffs, shape, center, scale, order):
    """
    Optimized reconstruction using precomputed powers and broadcasting.
    """
    vol = np.zeros(shape, dtype=np.uint8)
    
    # 1. Determine ROI (Bounding Box)
    pad = int(scale * 1.1)
    z_min, z_max = max(0, int(center[0]-pad)), min(shape[0], int(center[0]+pad))
    y_min, y_max = max(0, int(center[1]-pad)), min(shape[1], int(center[1]+pad))
    x_min, x_max = max(0, int(center[2]-pad)), min(shape[2], int(center[2]+pad))
    
    # Dimensions of the ROI
    nz, ny, nx = z_max - z_min, y_max - y_min, x_max - x_min
    if nz <= 0 or ny <= 0 or nx <= 0: return vol

    # 2. Create Normalized Coordinate Vectors (1D)
    # Instead of making a meshgrid immediately, we keep them 1D first.
    z_coords = (np.arange(z_min, z_max) - center[0]) / scale
    y_coords = (np.arange(y_min, y_max) - center[1]) / scale
    x_coords = (np.arange(x_min, x_max) - center[2]) / scale
    
    # 3. Precompute Powers (Vandermonde-ish)
    # Shape: (order+1, N)
    # P_x[p, :] contains x^p for all x in the ROI
    P_z = np.array([z_coords**p for p in range(order + 1)])
    P_y = np.array([y_coords**p for p in range(order + 1)])
    P_x = np.array([x_coords**p for p in range(order + 1)])

    # 4. Re-assemble the Polynomial via Tensor Contraction
    # The polynomial is: Sum_{i,j,k} ( C_{ijk} * Z^i * Y^j * X^k )
    # This is a tensor dot product.
    
    # First, we need to unpack the flat 'coeffs' list back into a 3D tensor C[i,j,k].
    # Since our flat list skipped terms where i+j+k > order, we must be careful.
    C_tensor = np.zeros((order + 1, order + 1, order + 1))
    
    col_idx = 0
    # Use the exact same iteration order as the training step
    for i in range(order + 1):
        for j in range(order + 1):
            for k in range(order + 1):
                if i + j + k <= order:
                    # Note: Your training loop was (X^i, Y^j, Z^k), 
                    # so map indices carefully: C[k, j, i] if Z is axis 0, etc.
                    # Your generic loop: (X**i) * (Y**j) * (Z**k)
                    # Let's align C_tensor indices to (k, j, i) -> (Z, Y, X)
                    C_tensor[k, j, i] = coeffs[col_idx]
                    col_idx += 1

    # 5. Perform Contraction (Einstein Summation)
    # We want result[z, y, x] = Sum_{k,j,i} C[k,j,i] * P_z[k,z] * P_y[j,y] * P_x[i,x]
    # 'kji,kz,jy,ix->zyx'
    #   k,j,i are power indices
    #   z,y,x are spatial indices
    
    # einsum is efficient but can be memory hungry for large chunks.
    # If memory fails, we can do it in two steps.
    
    # Step A: Contract X axis -> Temp[k, j, y, x]
    # T1 = np.einsum('kji,ix->kjx', C_tensor, P_x) 
    # But contracting 3 axes at once is cleanest if it fits in RAM:
    
    recon_roi = np.einsum('kji,kz,jy,ix->zyx', C_tensor, P_z, P_y, P_x, optimize='optimal')
    
    # 6. Threshold and Place
    mask_roi = (recon_roi < 0)
    vol[z_min:z_max, y_min:y_max, x_min:x_max] = mask_roi
    
    return vol

def reconstruct_shape(mask, order=8): # Order 8 is a good balance for kidneys
    coeffs, center, scale, order = compute_implicit_coeffs(mask, order=order)
    mask_rec = reconstruct_implicit_fast(coeffs, mask.shape, center, scale, order)
    return mask_rec