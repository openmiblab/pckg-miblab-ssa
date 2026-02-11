import numpy as np
import gc
from itertools import product
from scipy.ndimage import distance_transform_edt
from numpy.polynomial.legendre import legvander
from scipy.ndimage import label, center_of_mass

def features_from_mask(mask, order=15, n_samples=60000, random_seed=42):
    """
    Extracts Legendre coefficients with a heavy weight on the mask surface.
    Optimized for capturing the renal hilum.
    """
    np.random.seed(random_seed)
    nz, ny, nx = mask.shape
    
    # 1. Coordinate System
    center = np.array([nz, ny, nx], dtype=np.float32) / 2.0
    scale = np.max([nz, ny, nx]).astype(np.float32) / 2.0

    # 2. Compute Signed Distance Field (SDF)
    sdf = (distance_transform_edt(1 - mask) - distance_transform_edt(mask)).astype(np.float32)
    
    # 3. Surface-Biased Sampling
    # We take 80% of samples near the boundary (|SDF| < 5)
    boundary_mask = np.abs(sdf) < 5.0
    idx_boundary = np.argwhere(boundary_mask)
    idx_random = np.random.randint(0, [nz, ny, nx], size=(int(n_samples * 0.2), 3))
    
    n_bound = int(n_samples * 0.8)
    if len(idx_boundary) > 0:
        idx_b_select = idx_boundary[np.random.choice(len(idx_boundary), n_bound, replace=True)]
    else:
        idx_b_select = idx_random[:n_bound]

    indices = np.vstack([idx_b_select, idx_random])
    values = sdf[indices[:, 0], indices[:, 1], indices[:, 2]]
    
    # 4. Weighting Function: W = exp(-|SDF| / sigma)
    # sigma=2.0 makes the weight drop off quickly as we move away from the surface
    weights = np.exp(-np.abs(values) / 2.0).astype(np.float32)
    
    # Clean up heavy SDF
    del sdf, boundary_mask, idx_boundary, idx_random
    gc.collect()

    # 5. Normalize and Map Axes (0->Z, 1->Y, 2->X)
    pts = ((indices - center) / scale).astype(np.float32)
    valid = np.all(np.abs(pts) <= 1.0, axis=1)
    
    z_norm, y_norm, x_norm = pts[valid, 0], pts[valid, 1], pts[valid, 2]
    values = values[valid]
    weights = weights[valid]

    # 6. Build Weighted Basis Matrix
    poly_indices = [(i, j, k) for i, j, k in product(range(order + 1), repeat=3) if i + j + k <= order]
    A = np.zeros((len(values), len(poly_indices)), dtype=np.float32)
    
    Tz = legvander(z_norm, order).astype(np.float32)
    Ty = legvander(y_norm, order).astype(np.float32)
    Tx = legvander(x_norm, order).astype(np.float32)

    for idx, (i, j, k) in enumerate(poly_indices):
        # We multiply the basis column by weights to focus the fit
        A[:, idx] = (Tz[:, i] * Ty[:, j] * Tx[:, k]) * weights

    # 7. Weighted Normal Equations Solve
    # Applying weights to 'values' as well to complete the weighted least squares
    weighted_values = values * weights
    AtA = A.T @ A
    Atb = A.T @ weighted_values
    
    # Regularization
    AtA.flat[::len(poly_indices) + 1] += 1e-3
    
    coeffs = np.linalg.solve(AtA, Atb)

    del A, AtA, Atb, Tz, Ty, Tx
    gc.collect()

    return coeffs.astype(np.float32)




def mask_from_features(coeffs, shape, order):
    """
    Reconstructs the mask from surface-weighted Legendre coefficients.
    """
    vol = np.zeros(shape, dtype=np.uint8)
    nz, ny, nx = shape
    center = np.array([nz, ny, nx], dtype=np.float32) / 2.0
    scale = np.max([nz, ny, nx]).astype(np.float32) / 2.0
    
    # 1. Define ROI
    pad = int(scale * 1.0) 
    z_min, z_max = max(0, int(center[0]-pad)), min(nz, int(center[0]+pad))
    y_min, y_max = max(0, int(center[1]-pad)), min(ny, int(center[1]+pad))
    x_min, x_max = max(0, int(center[2]-pad)), min(nx, int(center[2]+pad))

    # 2. 1D Coords and Basis
    z_c = (np.arange(z_min, z_max) - center[0]) / scale
    y_c = (np.arange(y_min, y_max) - center[1]) / scale
    x_c = (np.arange(x_min, x_max) - center[2]) / scale
    
    Tz = legvander(z_c, order).astype(np.float32)
    Ty = legvander(y_c, order).astype(np.float32)
    Tx = legvander(x_c, order).astype(np.float32)

    # 3. Assemble Tensor (Must match extraction loop exactly)
    C_tensor = np.zeros((order + 1, order + 1, order + 1), dtype=np.float32)
    poly_indices = [(i, j, k) for i, j, k in product(range(order + 1), repeat=3) if i + j + k <= order]
    for idx, (i, j, k) in enumerate(poly_indices):
        C_tensor[i, j, k] = coeffs[idx]

    # 4. Contract Axis 0->Z, 1->Y, 2->X
    recon_roi = np.einsum('ijk,zi,yj,xk->zyx', C_tensor, Tz, Ty, Tx, optimize='optimal')
    
    # 5. Geometric Crop
    zz, yy, xx = np.meshgrid(z_c, y_c, x_c, indexing='ij')
    valid_box = (np.abs(zz) <= 1.0) & (np.abs(yy) <= 1.0) & (np.abs(xx) <= 1.0)
    mask_roi = (recon_roi < 0) & valid_box
    
    # 6. Anatomical Connectivity (Keep largest part near center)
    if np.any(mask_roi):
        labeled, n_components = label(mask_roi)
        if n_components > 1:
            rc = np.array([(z_max-z_min)/2, (y_max-y_min)/2, (x_max-x_min)/2])
            centers = np.array(center_of_mass(mask_roi, labeled, range(1, n_components+1)))
            winner_idx = np.argmin(np.linalg.norm(centers - rc, axis=1)) + 1 
            mask_roi = (labeled == winner_idx)
    
    vol[z_min:z_max, y_min:y_max, x_min:x_max] = mask_roi
    return vol

def smooth_mask(mask:np.ndarray, order=20):
    coeffs = features_from_mask(mask, order=order)
    mask_rec = mask_from_features(coeffs, mask.shape, order)
    return mask_rec