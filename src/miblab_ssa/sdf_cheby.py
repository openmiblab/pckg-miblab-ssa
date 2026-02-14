import numpy as np
import gc
from scipy.ndimage import distance_transform_edt, label
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



def features_from_mask(mask, order=15, n_samples=50000, random_seed=42):
    np.random.seed(random_seed)
    nz, ny, nx = mask.shape
    
    # 1. Compute SDF
    # ~mask is the same as 1-mask for boolean/binary
    sdf = (distance_transform_edt(~mask) - distance_transform_edt(mask)).astype(np.float32)
    
    # 2. Hybrid Sampling (50% Boundary, 50% Random)
    n_boundary = n_samples // 2
    n_random = n_samples - n_boundary
    
    # Target points near the surface (SDF close to 0)
    boundary_mask = np.abs(sdf) < 3.0
    idx_boundary = np.argwhere(boundary_mask)
    
    if len(idx_boundary) > 0:
        b_choices = np.random.choice(len(idx_boundary), n_boundary, replace=True)
        idx_b = idx_boundary[b_choices]
    else:
        idx_b = np.random.randint(0, [nz, ny, nx], size=(n_boundary, 3))
        
    idx_r = np.random.randint(0, [nz, ny, nx], size=(n_random, 3))
    
    # Combine and extract values
    indices = np.vstack([idx_b, idx_r])
    z_idx, y_idx, x_idx = indices[:, 0], indices[:, 1], indices[:, 2]
    values = sdf[z_idx, y_idx, x_idx]
    
    # 3. Normalize Coords to [-1, 1]
    center = np.array([nz, ny, nx]) / 2.0
    scale = np.max([nz, ny, nx]) / 2.0
    
    Z = (z_idx - center[0]) / scale
    Y = (y_idx - center[1]) / scale
    X = (x_idx - center[2]) / scale
    
    # 4. Generate Basis (Aligned: i=Z, j=Y, k=X)
    Tz = chebvander(Z, order)
    Ty = chebvander(Y, order)
    Tx = chebvander(X, order)
    
    basis_cols = []
    for i, j, k in product(range(order + 1), repeat=3):
        if i + j + k <= order:
            # Multiplied to match (z, y, x) logic
            basis_cols.append(Tz[:, i] * Ty[:, j] * Tx[:, k])
            
    A = np.column_stack(basis_cols)
    
    # 5. Solve using Ridge
    # 'cholesky' is usually the fastest solver for dense A matrices of this size
    clf = Ridge(alpha=1e-2, fit_intercept=False, solver='cholesky')
    clf.fit(A, values)
    
    return clf.coef_.astype(np.float32)


def mask_from_features(coeffs, shape, order):
    """
    Simplified reconstruction of mask from Chebyshev coefficients.
    """
    nz, ny, nx = shape
    center = np.array([nz, ny, nx]) / 2.0
    scale = np.max([nz, ny, nx]) / 2.0

    # 1. Create global coordinates normalized to [-1, 1]
    z_coords = (np.arange(nz) - center[0]) / scale
    y_coords = (np.arange(ny) - center[1]) / scale
    x_coords = (np.arange(nx) - center[2]) / scale

    # 2. Precompute Vandermonde basis matrices
    # Transpose (.T) so they are ready for einsum contraction
    Tz = chebvander(z_coords, order).T
    Ty = chebvander(y_coords, order).T
    Tx = chebvander(x_coords, order).T

    # 3. Reconstruct the Coefficient Tensor
    # Must match the 'product(range(order + 1), repeat=3)' order exactly
    C_tensor = np.zeros((order + 1, order + 1, order + 1))
    idx = 0
    for i, j, k in product(range(order + 1), repeat=3):
        if i + j + k <= order:
            C_tensor[i, j, k] = coeffs[idx]
            idx += 1

    # Now i -> Z, j -> Y, k -> X
    # 'iz' means i-index maps to z-coordinate, etc.
    recon_sdf = np.einsum('ijk,iz,jy,kx->zyx', C_tensor, Tz, Ty, Tx, optimize='optimal')

    # 6. Geometric Crop (Optional but recommended for high-order stability)
    # Ensures no "ghosting" artifacts at the very edges of the volume
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    valid_box = (np.abs(zz) <= 1.0) & (np.abs(yy) <= 1.0) & (np.abs(xx) <= 1.0)
    recon_sdf = (recon_sdf < 0) & valid_box

    # POST-PROCESS: Keep only the largest component (The Kidney)
    if np.any(recon_sdf):
        labeled, num_features = label(recon_sdf)
        if num_features > 1:
            # Sort components by size
            sizes = np.bincount(labeled.ravel())
            largest_label = sizes[1:].argmax() + 1
            recon_sdf = (labeled == largest_label)

    return recon_sdf


def smooth_mask(mask:np.ndarray, order=20):
    coeffs = features_from_mask(mask, order=order)
    mask_rec = mask_from_features(coeffs, mask.shape, order)
    return mask_rec


# N = (L+1) (L+2) (L+3) / 6
