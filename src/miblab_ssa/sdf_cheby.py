import numpy as np
from scipy.ndimage import distance_transform_edt, label, center_of_mass
from numpy.polynomial.chebyshev import chebvander
from sklearn.linear_model import Ridge
from itertools import product


def features_from_mask(mask, order=15, n_samples=50000, random_seed=42):
    """
    Extracts PCA-ready shape coefficients using a fixed coordinate system 
    based on the mask volume dimensions.
    """
    # Fix the seed so the random sampling is identical every time
    np.random.seed(random_seed)

    # 1. Define Fixed Coordinate System (Middle of the Cube)
    # Since your data is registered, we use the volume's geometrical center.
    nz, ny, nx = mask.shape
    center = np.array([nz, ny, nx]) / 2.0
    
    # Scale covers the entire volume (radius from center to edge)
    # For a cube of size N, radius is N/2. 
    scale = np.max([nz, ny, nx]) / 2.0

    # 2. Compute SDF
    dist_outside = distance_transform_edt(1 - mask)
    dist_inside = distance_transform_edt(mask)
    sdf = dist_outside - dist_inside
    
    # 3. Robust Sampling Strategy
    boundary_mask = np.abs(sdf) < 3.0
    idx_boundary = np.argwhere(boundary_mask)
    
    # Generate random indices globally
    idx_random = np.random.randint(0, [nz, ny, nx], size=(n_samples, 3))
    
    # Combine (50% boundary, 50% random air)
    n_bound = int(n_samples * 0.5)
    if len(idx_boundary) > 0:
        idx_b_select = idx_boundary[np.random.choice(len(idx_boundary), n_bound, replace=True)]
    else:
        idx_b_select = idx_boundary # Fallback

    indices = np.vstack([idx_b_select, idx_random])
    
    z_idx, y_idx, x_idx = indices[:, 0], indices[:, 1], indices[:, 2]
    Values = sdf[z_idx, y_idx, x_idx]
    
    # 4. Normalize Coords to [-1, 1] using Fixed Center/Scale
    Z = (z_idx - center[0]) / scale
    Y = (y_idx - center[1]) / scale
    X = (x_idx - center[2]) / scale
    
    # Filter valid cube (points outside the fitting domain must be ignored)
    valid_mask = (np.abs(X) <= 1) & (np.abs(Y) <= 1) & (np.abs(Z) <= 1)
    
    X, Y, Z = X[valid_mask], Y[valid_mask], Z[valid_mask]
    Values = Values[valid_mask]
    
    # 5. Generate Basis
    Tx = chebvander(X, order)
    Ty = chebvander(Y, order)
    Tz = chebvander(Z, order)
    
    basis_cols = []
    # Deterministic loop order for PCA consistency
    for i, j, k in product(range(order + 1), repeat=3):
        if i + j + k <= order:
            col = Tx[:, i] * Ty[:, j] * Tz[:, k]
            basis_cols.append(col)
            
    A = np.column_stack(basis_cols)
    
    # 6. Solve
    clf = Ridge(alpha=1e-3, fit_intercept=False)
    clf.fit(A, Values)
    
    return clf.coef_

def mask_from_features(coeffs, shape, order):
    """
    Reconstructs mask from coefficients using the fixed volume center/scale.
    """
    vol = np.zeros(shape, dtype=np.uint8)
    
    # 1. Define Fixed Coordinate System (Same as features_from_mask)
    nz, ny, nx = shape
    center = np.array([nz, ny, nx]) / 2.0
    scale = np.max([nz, ny, nx]) / 2.0
    
    # 2. ROI optimization (We only loop over the area covered by 'scale')
    pad = int(scale * 1.0) 
    z_min, z_max = max(0, int(center[0]-pad)), min(shape[0], int(center[0]+pad))
    y_min, y_max = max(0, int(center[1]-pad)), min(shape[1], int(center[1]+pad))
    x_min, x_max = max(0, int(center[2]-pad)), min(shape[2], int(center[2]+pad))
    
    roi_nz, roi_ny, roi_nx = z_max - z_min, y_max - y_min, x_max - x_min
    if roi_nz <= 0 or roi_ny <= 0 or roi_nx <= 0: return vol

    # 3. Coords
    z_coords = (np.arange(z_min, z_max) - center[0]) / scale
    y_coords = (np.arange(y_min, y_max) - center[1]) / scale
    x_coords = (np.arange(x_min, x_max) - center[2]) / scale
    
    # 4. Basis
    Tz = chebvander(z_coords, order).T
    Ty = chebvander(y_coords, order).T
    Tx = chebvander(x_coords, order).T

    # 5. Tensor Assembly
    poly_indices = []
    for i, j, k in product(range(order + 1), repeat=3):
        if i + j + k <= order:
            poly_indices.append((i, j, k))

    C_tensor = np.zeros((order + 1, order + 1, order + 1))
    for idx, (i, j, k) in enumerate(poly_indices):
        C_tensor[k, j, i] = coeffs[idx]

    # 6. Contraction
    recon_roi = np.einsum('kji,kz,jy,ix->zyx', C_tensor, Tz, Ty, Tx, optimize='optimal')
    
    # 7. Hard Geometric Crop
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    valid_box = (np.abs(zz) <= 1.0) & (np.abs(yy) <= 1.0) & (np.abs(xx) <= 1.0)
    
    mask_roi = (recon_roi < 0) & valid_box
    
    # 8. Intelligent Filtering (Center Distance)
    if np.any(mask_roi):
        labeled, n_components = label(mask_roi)
        if n_components > 1:
            # We assume the kidney is near the center of the volume
            roi_center = np.array([roi_nz/2, roi_ny/2, roi_nx/2])
            
            centers = center_of_mass(mask_roi, labeled, range(1, n_components+1))
            centers = np.array(centers)
            
            if len(centers) > 0:
                dists = np.linalg.norm(centers - roi_center, axis=1)
                winner_idx = np.argmin(dists) + 1 
                mask_roi = (labeled == winner_idx)
    
    vol[z_min:z_max, y_min:y_max, x_min:x_max] = mask_roi
    
    return vol

def smooth_mask(mask:np.ndarray, order=20):
    coeffs = features_from_mask(mask, order=order)
    mask_rec = mask_from_features(coeffs, mask.shape, order)
    return mask_rec

# N = (L+1) (L+2) (L+3) / 6
