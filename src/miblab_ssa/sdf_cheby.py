import numpy as np
import gc
from scipy.ndimage import distance_transform_edt, label, center_of_mass
from numpy.polynomial.chebyshev import chebvander
from sklearn.linear_model import Ridge
from itertools import product


def descriptors(coeffs, deg_vecs):
    """
    Extracts physical descriptors from Chebyshev coefficients.
    
    Args:
        coeffs: The clf.coef_ array from your function.
        deg_vecs: The (3, N) degree array from get_chebyshev_frequency_vectors.
    """
    # 1. Magnitudes and Powers
    magnitudes = np.abs(coeffs)
    powers = coeffs**2
    total_mag = np.sum(magnitudes)
    total_power = np.sum(powers)
    
    # 2. Total Degree (Proxy for frequency)
    # Sum of degrees (i+j+k)
    total_degrees = np.sum(deg_vecs, axis=0)
    
    # 3. Shape Complexity (Weighted Mean Degree)
    # High mean degree = more complex surface curvature
    mean_degree = np.sum(total_degrees * magnitudes) / total_mag
    
    # 4. Anisotropy (Directional Complexity)
    # Mean degree per axis
    mean_deg_x = np.sum(deg_vecs[0] * magnitudes) / total_mag
    mean_deg_y = np.sum(deg_vecs[1] * magnitudes) / total_mag
    mean_deg_z = np.sum(deg_vecs[2] * magnitudes) / total_mag
    
    # 5. Low-Degree Compaction (Coarse vs Fine detail)
    # Degree <= 2 represents the basic ellipsoid/size
    coarse_power = np.sum(powers[total_degrees <= 2])
    compaction_ratio = coarse_power / (total_power + 1e-9)

    return {
        'total_power': float(total_power),
        'mean_degree_complexity': float(mean_degree),
        'compaction_ratio_coarse': float(compaction_ratio),
        'anisotropy_index': float(np.std([mean_deg_x, mean_deg_y, mean_deg_z]) / mean_degree),
        'mean_deg_x': float(mean_deg_x),
        'mean_deg_y': float(mean_deg_y),
        'mean_deg_z': float(mean_deg_z)
    }

def spectral_vectors(order):
    """
    Returns the polynomial degrees (i, j, k) for each Chebyshev coefficient.
    
    Returns:
        np.ndarray: A (3, N) array where:
            row 0 = Degree in X
            row 1 = Degree in Y
            row 2 = Degree in Z
    """
    degrees = []
    # Must follow the EXACT same loop order as your features_from_mask function
    for i, j, k in product(range(order + 1), repeat=3):
        if i + j + k <= order:
            degrees.append([i, j, k])
            
    # Transpose to get (3, N)
    return np.array(degrees).T.astype(np.int32)

def features_from_mask(mask, order=15, n_samples=60000, random_seed=42):
    """
    Extracts Chebyshev coefficients with a heavy weight on the mask surface.
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
    
    Tz = chebvander(z_norm, order).astype(np.float32)
    Ty = chebvander(y_norm, order).astype(np.float32)
    Tx = chebvander(x_norm, order).astype(np.float32)

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

# OLD VERSION BUT SHOULD HAVE WORKED - not sure why. Uses Ridge Regression
# def features_from_mask(mask, order=15, n_samples=50000, random_seed=42):
#     """
#     Extracts PCA-ready shape coefficients using a fixed coordinate system 
#     based on the mask volume dimensions.
#     """
#     # Fix the seed so the random sampling is identical every time
#     np.random.seed(random_seed)

#     # 1. Define Fixed Coordinate System (Middle of the Cube)
#     # Since your data is registered, we use the volume's geometrical center.
#     nz, ny, nx = mask.shape
#     center = np.array([nz, ny, nx]) / 2.0
    
#     # Scale covers the entire volume (radius from center to edge)
#     # For a cube of size N, radius is N/2. 
#     scale = np.max([nz, ny, nx]) / 2.0

#     # 2. Compute SDF
#     dist_outside = distance_transform_edt(1 - mask)
#     dist_inside = distance_transform_edt(mask)
#     sdf = dist_outside - dist_inside
    
#     # 3. Robust Sampling Strategy
#     boundary_mask = np.abs(sdf) < 3.0
#     idx_boundary = np.argwhere(boundary_mask)
    
#     # Generate random indices globally
#     idx_random = np.random.randint(0, [nz, ny, nx], size=(n_samples, 3))
    
#     # Combine (50% boundary, 50% random air)
#     n_bound = int(n_samples * 0.5)
#     if len(idx_boundary) > 0:
#         idx_b_select = idx_boundary[np.random.choice(len(idx_boundary), n_bound, replace=True)]
#     else:
#         idx_b_select = idx_boundary # Fallback

#     indices = np.vstack([idx_b_select, idx_random])
    
#     z_idx, y_idx, x_idx = indices[:, 0], indices[:, 1], indices[:, 2]
#     Values = sdf[z_idx, y_idx, x_idx]
    
#     # 4. Normalize Coords to [-1, 1] using Fixed Center/Scale
#     Z = (z_idx - center[0]) / scale
#     Y = (y_idx - center[1]) / scale
#     X = (x_idx - center[2]) / scale
    
#     # Filter valid cube (points outside the fitting domain must be ignored)
#     valid_mask = (np.abs(X) <= 1) & (np.abs(Y) <= 1) & (np.abs(Z) <= 1)
    
#     X, Y, Z = X[valid_mask], Y[valid_mask], Z[valid_mask]
#     Values = Values[valid_mask]
    
#     # 5. Generate Basis
#     Tx = chebvander(X, order)
#     Ty = chebvander(Y, order)
#     Tz = chebvander(Z, order)
    
#     basis_cols = []
#     # Deterministic loop order for PCA consistency
#     for i, j, k in product(range(order + 1), repeat=3):
#         if i + j + k <= order:
#             col = Tx[:, i] * Ty[:, j] * Tz[:, k]
#             basis_cols.append(col)
            
#     A = np.column_stack(basis_cols)
    
#     # 6. Solve
#     clf = Ridge(alpha=1e-3, fit_intercept=False)
#     clf.fit(A, Values)
    
#     return clf.coef_.astype(np.float32)


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
