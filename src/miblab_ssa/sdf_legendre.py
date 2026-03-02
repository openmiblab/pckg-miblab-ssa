import numpy as np
from scipy.ndimage import distance_transform_edt, label
from numpy.polynomial.legendre import legvander
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



def features_from_mask_regression(mask, order=15, n_samples=50000, random_seed=42):
    np.random.seed(random_seed)
    nz, ny, nx = mask.shape
    
    # 1. Compute SDF
    sdf = (distance_transform_edt(~mask) - distance_transform_edt(mask)).astype(np.float32)
    
    # 2. Hybrid Sampling (50% Boundary, 50% Random)
    n_boundary = n_samples // 2
    n_random = n_samples - n_boundary
    
    boundary_mask = np.abs(sdf) < 3.0
    idx_boundary = np.argwhere(boundary_mask)
    
    if len(idx_boundary) > 0:
        b_choices = np.random.choice(len(idx_boundary), n_boundary, replace=True)
        idx_b = idx_boundary[b_choices]
    else:
        idx_b = np.random.randint(0, [nz, ny, nx], size=(n_boundary, 3))
        
    idx_r = np.random.randint(0, [nz, ny, nx], size=(n_random, 3))
    
    indices = np.vstack([idx_b, idx_r])
    z_idx, y_idx, x_idx = indices[:, 0], indices[:, 1], indices[:, 2]
    values = sdf[z_idx, y_idx, x_idx]
    
    # 3. Normalize Coords to [-1, 1]
    center = np.array([nz, ny, nx]) / 2.0
    scale = np.max([nz, ny, nx]) / 2.0
    
    Z = (z_idx - center[0]) / scale
    Y = (y_idx - center[1]) / scale
    X = (x_idx - center[2]) / scale
    
    # 4. Generate Legendre Basis (Aligned: i=Z, j=Y, k=X)
    Tz = legvander(Z, order)
    Ty = legvander(Y, order)
    Tx = legvander(X, order)
    
    basis_cols = []
    for i, j, k in product(range(order + 1), repeat=3):
        if i + j + k <= order:
            basis_cols.append(Tz[:, i] * Ty[:, j] * Tx[:, k])
            
    A = np.column_stack(basis_cols)
    
    # 5. Solve using Ridge
    clf = Ridge(alpha=1e-2, fit_intercept=False, solver='cholesky')
    clf.fit(A, values)
    
    return clf.coef_.astype(np.float32)



def features_from_mask(mask, order=15, saturation_threshold=5.0):
    nz, ny, nx = mask.shape
    
    # 1. Full Volume SDF (Deterministic, no sampling)
    mask = mask.astype(bool)
    sdf = (distance_transform_edt(~mask) - distance_transform_edt(mask)).astype(np.float32)
    # Tanh saturation is vital for "explainability" in integration
    sdf_sat = saturation_threshold * np.tanh(sdf / saturation_threshold)

    # 2. Setup Normalized Coordinates
    z_coords = (np.arange(nz) - nz/2.0) / (max(nz, ny, nx)/2.0)
    y_coords = (np.arange(ny) - ny/2.0) / (max(nz, ny, nx)/2.0)
    x_coords = (np.arange(nx) - nx/2.0) / (max(nz, ny, nx)/2.0)

    # 3. Precompute 1D Legendre Basis Mats
    Lz = legvander(z_coords, order) # Shape (nz, order+1)
    Ly = legvander(y_coords, order) # Shape (ny, order+1)
    Lx = legvander(x_coords, order) # Shape (nx, order+1)

    # 4. Integrate (The "Spectral" Way)
    # We use tensordot to efficiently project the 3D volume onto the basis
    # This is the full-volume equivalent of your Ridge Regression
    coeffs = []
    order_sq = order**2
    for i, j, k in product(range(order + 1), repeat=3):
        if (i**2 + j**2 + k**2) <= order_sq:
            # This line performs the 3D integral for one specific basis function
            # It's identical to: np.sum(sdf_sat * 3D_Legendre_Basis_i_j_k)
            # But much faster using separated 1D bases
            term = np.einsum('zyx,z,y,x->', sdf_sat, Lz[:, i], Ly[:, j], Lx[:, k])
            
            # Normalization factor for Legendre: (2n+1)/2
            # Since it's 3D, we multiply the factors for i, j, and k
            norm = ((2*i + 1)/2) * ((2*j + 1)/2) * ((2*k + 1)/2)
            coeffs.append(term * norm / (nz*ny*nx))

    return np.array(coeffs, dtype=np.float32)


def mask_from_features(coeffs, shape, order):
    nz, ny, nx = shape
    center = np.array([nz, ny, nx]) / 2.0
    scale = np.max([nz, ny, nx]) / 2.0

    z_coords = (np.arange(nz) - center[0]) / scale
    y_coords = (np.arange(ny) - center[1]) / scale
    x_coords = (np.arange(nx) - center[2]) / scale

    # Legendre Vander matrices
    Tz = legvander(z_coords, order).T
    Ty = legvander(y_coords, order).T
    Tx = legvander(x_coords, order).T

    C_tensor = np.zeros((order + 1, order + 1, order + 1))
    order_sq = order**2
    idx = 0
    for i, j, k in product(range(order + 1), repeat=3):
        if (i**2 + j**2 + k**2) <= order_sq:
            C_tensor[i, j, k] = coeffs[idx]
            idx += 1

    # Tensor Contraction
    recon_sdf = np.einsum('ijk,iz,jy,kx->zyx', C_tensor, Tz, Ty, Tx, optimize='optimal')
    recon_sdf = recon_sdf < 0

    # # Geometric Crop for high-order stability
    # zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    # valid_box = (np.abs(zz) <= 1.0) & (np.abs(yy) <= 1.0) & (np.abs(xx) <= 1.0)
    # recon_sdf = recon_sdf & valid_box

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
