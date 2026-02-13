import numpy as np
import gc
from scipy.ndimage import distance_transform_edt, label
from itertools import product

def get_rbf_basis(x, n_centers, epsilon=None):
    """
    Generates a 1D Gaussian RBF basis matrix.
    """
    centers = np.linspace(-1, 1, n_centers)
    if epsilon is None:
        # Standard heuristic for Gaussian overlap
        epsilon = 1.0 / (centers[1] - centers[0])
    
    # Compute Gaussian: exp(-(epsilon * dist)^2)
    dists_sq = (x[:, None] - centers[None, :])**2
    basis = np.exp(-epsilon**2 * dists_sq)
    return basis.astype(np.float32)

def features_from_mask(mask, order=12, epsilon=2.0, n_samples=60000):
    nz, ny, nx = mask.shape
    center = np.array([nz, ny, nx], dtype=np.float32) / 2.0
    scale = np.max([nz, ny, nx]).astype(np.float32) / 2.0

    sdf = (distance_transform_edt(1 - mask) - distance_transform_edt(mask)).astype(np.float32)
    
    # 1. Surface-Biased Sampling
    idx_boundary = np.argwhere(np.abs(sdf) < 3.0)
    idx_random = np.random.randint(0, [nz, ny, nx], size=(int(n_samples*0.2), 3))
    
    # 2. ANCHORING: Pin the outer shell of the volume to positive (Air)
    # We use the 8 corners and the centers of the 6 faces
    anchors = np.array(list(product([0, nz-1], [0, ny-1], [0, nx-1])))
    
    b_count = int(n_samples*0.8)
    idx_b_select = idx_boundary[np.random.choice(len(idx_boundary), b_count)]
    
    indices = np.vstack([idx_b_select, idx_random, anchors])
    values = sdf[indices[:, 0], indices[:, 1], indices[:, 2]]
    
    # 3. Weighting & Normalization
    weights = np.exp(-np.abs(values) / 2.0).astype(np.float32)
    weights[-len(anchors):] = 100.0 # Heavy penalty for matter at the corners
    
    pts = ((indices - center) / scale).astype(np.float32)

    # 1D Bases
    Bz = get_rbf_basis(pts[:, 0], order, epsilon)
    By = get_rbf_basis(pts[:, 1], order, epsilon)
    Bx = get_rbf_basis(pts[:, 2], order, epsilon)

    # Tensor Product Basis Construction
    A = (Bz[:, :, None, None] * By[:, None, :, None] * Bx[:, None, None, :]).reshape(len(values), -1)
    A *= weights[:, None]
    weighted_values = values * weights

    # 4. Solve via Normal Equations
    n_feat = order**3
    AtA = A.T @ A
    Atb = A.T @ weighted_values
    
    del A, Bz, By, Bx, sdf, indices
    gc.collect()

    # High alpha (1e-1) is vital for RBFs to prevent boundary noise
    AtA.flat[::n_feat + 1] += 1e-1 
    coeffs = np.linalg.solve(AtA, Atb)
    
    return coeffs.astype(np.float32)

def mask_from_features(coeffs, shape, order=12, epsilon=2.0):
    nz, ny, nx = shape
    center = np.array([nz, ny, nx], dtype=np.float32) / 2.0
    scale = np.max([nz, ny, nx]).astype(np.float32) / 2.0
    
    z_c = (np.arange(nz) - center[0]) / scale
    y_c = (np.arange(ny) - center[1]) / scale
    x_c = (np.arange(nx) - center[2]) / scale
    
    Bz = get_rbf_basis(z_c, order, epsilon)
    By = get_rbf_basis(y_c, order, epsilon)
    Bx = get_rbf_basis(x_c, order, epsilon)

    C_tensor = coeffs.reshape((order, order, order))
    recon_sdf = np.einsum('ijk,zi,yj,xk->zyx', C_tensor, Bz, By, Bx, optimize='optimal')
    
    # 5. Geometric Crop & Indexing fix
    zz, yy, xx = np.meshgrid(z_c, y_c, x_c, indexing='ij')
    valid_box = (np.abs(zz) < 0.98) & (np.abs(yy) < 0.98) & (np.abs(xx) < 0.98)
    
    mask = (recon_sdf < 0) & valid_box
    
    # 6. POST-PROCESS: Keep largest component
    if np.any(mask):
        labeled, num_features = label(mask)
        if num_features > 1:
            sizes = np.bincount(labeled.ravel())
            largest_label = sizes[1:].argmax() + 1
            mask = (labeled == largest_label)
            
    return mask.astype(bool)

def smooth_mask(mask:np.ndarray, order=16, epsilon=2.0):
    coeffs = features_from_mask(mask, order=order, epsilon=epsilon)
    mask_rec = mask_from_features(coeffs, mask.shape, order=order, epsilon=epsilon)
    return mask_rec

