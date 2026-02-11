import numpy as np
from numpy.polynomial.legendre import legvander
from scipy.ndimage import distance_transform_edt
from sklearn.linear_model import Ridge
from itertools import product
from scipy.ndimage import label, center_of_mass
import gc





def features_from_mask(mask, order=15, n_samples=50000, random_seed=42):
    """
    Memory-optimized Legendre feature extraction.
    Uses float32 and Normal Equations to keep RAM < 1GB.
    """
    # 1. Setup Fixed Coordinate System
    np.random.seed(random_seed)
    nz, ny, nx = mask.shape
    center = np.array([nz, ny, nx], dtype=np.float32) / 2.0
    scale = np.max([nz, ny, nx]).astype(np.float32) / 2.0

    # 2. Compute SDF (immediately cast to float32)
    # distance_transform_edt can be a memory hog; we compute and cast
    sdf = (distance_transform_edt(1 - mask) - distance_transform_edt(mask)).astype(np.float32)
    
    # 3. Robust Sampling Strategy
    boundary_mask = np.abs(sdf) < 3.0
    idx_boundary = np.argwhere(boundary_mask)
    idx_random = np.random.randint(0, [nz, ny, nx], size=(n_samples, 3))
    
    n_bound = int(n_samples * 0.5)
    if len(idx_boundary) > 0:
        # Use replace=True if boundary points are fewer than requested samples
        idx_b_select = idx_boundary[np.random.choice(len(idx_boundary), n_bound, replace=True)]
    else:
        idx_b_select = idx_random[:n_bound]

    indices = np.vstack([idx_b_select, idx_random])
    Values = sdf[indices[:, 0], indices[:, 1], indices[:, 2]]
    
    # Clear large temporary arrays early
    del sdf, boundary_mask, idx_boundary, idx_random
    gc.collect()

    # 4. Normalize Coords and Filter Valid Cube
    pts = ((indices - center) / scale).astype(np.float32)
    valid_mask = np.all(np.abs(pts) <= 1.0, axis=1)
    
    # Coordinates for vandermonde
    Z, Y, X = pts[valid_mask, 0], pts[valid_mask, 1], pts[valid_mask, 2]
    Values = Values[valid_mask]

    # 5. Build Basis Matrix A (Pre-allocated float32)
    # Number of features for total order truncation: (order+1)(order+2)(order+3)/6
    poly_indices = [(i, j, k) for i, j, k in product(range(order + 1), repeat=3) 
                    if i + j + k <= order]
    n_feat = len(poly_indices)
    
    A = np.zeros((len(Values), n_feat), dtype=np.float32)
    
    # Pre-compute 1D Vandermonde matrices
    Tz = legvander(Z, order).astype(np.float32)
    Ty = legvander(Y, order).astype(np.float32)
    Tx = legvander(X, order).astype(np.float32)

    for idx, (i, j, k) in enumerate(poly_indices):
        # Column-wise assignment is memory efficient
        A[:, idx] = Tz[:, i] * Ty[:, j] * Tx[:, k]

    # 6. Solve via Normal Equations (Ridge equivalent)
    # Solve (A.T @ A + alpha*I) @ w = A.T @ Values
    alpha = 1e-3
    AtA = A.T @ A
    Atb = A.T @ Values
    
    # Regularization (identity matrix scaled by alpha)
    AtA.flat[::n_feat + 1] += alpha 
    
    coeffs = np.linalg.solve(AtA, Atb)

    # 7. Final Cleanup
    del A, AtA, Atb, Tz, Ty, Tx, pts, indices
    gc.collect()

    return coeffs.astype(np.float32)




def mask_from_features(coeffs, shape, order):
    """
    Reconstructs mask from Legendre coefficients with corrected orientation.
    """
    vol = np.zeros(shape, dtype=np.uint8)
    nz, ny, nx = shape
    
    # 1. Coordinate System (Must match features_from_mask exactly)
    center = np.array([nz, ny, nx], dtype=np.float32) / 2.0
    scale = np.max([nz, ny, nx]).astype(np.float32) / 2.0
    
    # 2. ROI Optimization
    pad = int(scale * 1.0) 
    z_min, z_max = max(0, int(center[0]-pad)), min(nz, int(center[0]+pad))
    y_min, y_max = max(0, int(center[1]-pad)), min(ny, int(center[1]+pad))
    x_min, x_max = max(0, int(center[2]-pad)), min(nx, int(center[2]+pad))
    
    # 3. Generate 1D Normalized Coords
    z_coords = (np.arange(z_min, z_max) - center[0]) / scale
    y_coords = (np.arange(y_min, y_max) - center[1]) / scale
    x_coords = (np.arange(x_min, x_max) - center[2]) / scale
    
    # 4. Generate 1D Legendre Basis (Transposed for contraction)
    Tz = legvander(z_coords, order).T.astype(np.float32)
    Ty = legvander(y_coords, order).T.astype(np.float32)
    Tx = legvander(x_coords, order).T.astype(np.float32)

    # 5. Tensor Assembly
    # The loop order MUST match features_from_mask exactly: i, j, k
    C_tensor = np.zeros((order + 1, order + 1, order + 1), dtype=np.float32)
    idx = 0
    from itertools import product
    for i, j, k in product(range(order + 1), repeat=3):
        if i + j + k <= order:
            # i -> Z, j -> Y, k -> X
            C_tensor[i, j, k] = coeffs[idx]
            idx += 1

    # 6. Fast Contraction (Einsum)
    # i, j, k are basis indices; z, y, x are spatial volume indices
    recon_roi = np.einsum('ijk,iz,jy,kx->zyx', C_tensor, Tz, Ty, Tx, optimize='optimal')
    
    # 7. Geometric Masking
    # Meshgrid is used only for the valid_box check to save memory
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    valid_box = (np.abs(zz) <= 1.0) & (np.abs(yy) <= 1.0) & (np.abs(xx) <= 1.0)
    
    # SDF < 0 defines the interior of the mask
    mask_roi = (recon_roi < 0) & valid_box
    
    # 8. Post-process: Keep largest component near center
    if np.any(mask_roi):
        labeled, n_components = label(mask_roi)
        if n_components > 1:
            roi_center = np.array([(z_max-z_min)/2, (y_max-y_min)/2, (x_max-x_min)/2])
            centers = np.array(center_of_mass(mask_roi, labeled, range(1, n_components+1)))
            winner_idx = np.argmin(np.linalg.norm(centers - roi_center, axis=1)) + 1 
            mask_roi = (labeled == winner_idx)
    
    vol[z_min:z_max, y_min:y_max, x_min:x_max] = mask_roi
    return vol

def smooth_mask(mask:np.ndarray, order=20):
    coeffs = features_from_mask(mask, order=order)
    mask_rec = mask_from_features(coeffs, mask.shape, order)
    return mask_rec