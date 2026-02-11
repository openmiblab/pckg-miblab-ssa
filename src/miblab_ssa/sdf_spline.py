import numpy as np
import gc
from itertools import product
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import BSpline
from scipy.ndimage import label, center_of_mass

def get_bspline_basis(x, n_control, degree=3):
    knots = np.linspace(-1, 1, n_control + degree + 1)
    basis = np.zeros((len(x), n_control), dtype=np.float32)
    for i in range(n_control):
        c = np.zeros(n_control); c[i] = 1.0
        basis[:, i] = BSpline(knots, c, degree, extrapolate=False)(x)
    return np.nan_to_num(basis)

def features_from_mask(mask, n_grid=16, degree=3, n_samples=60000):
    nz, ny, nx = mask.shape
    center = np.array([nz, ny, nx], dtype=np.float32) / 2.0
    scale = np.max([nz, ny, nx]).astype(np.float32) / 2.0

    sdf = (distance_transform_edt(1 - mask) - distance_transform_edt(mask)).astype(np.float32)
    
    # 1. Surface-Biased Sampling
    idx_boundary = np.argwhere(np.abs(sdf) < 5.0)
    idx_random = np.random.randint(0, [nz, ny, nx], size=(int(n_samples*0.2), 3))
    indices = np.vstack([idx_boundary[np.random.choice(len(idx_boundary), int(n_samples*0.8))], idx_random])
    values = sdf[indices[:, 0], indices[:, 1], indices[:, 2]]
    
    weights = np.exp(-np.abs(values) / 2.0).astype(np.float32)
    weighted_values = values * weights
    pts = ((indices - center) / scale).astype(np.float32)

    # 2. Generate 1D Spline Bases
    Bz = get_bspline_basis(pts[:, 0], n_grid, degree) # (N, n_grid)
    By = get_bspline_basis(pts[:, 1], n_grid, degree) # (N, n_grid)
    Bx = get_bspline_basis(pts[:, 2], n_grid, degree) # (N, n_grid)

    # 3. Vectorized Basis Matrix Building
    # Instead of a 4096-iteration loop, we use broadcasting
    # We compute A[:, idx] = Bz * By * Bx via reshaping
    # (N, n, 1, 1) * (N, 1, n, 1) * (N, 1, 1, n) -> (N, n, n, n)
    # Then flatten the last three dims to (N, n^3)
    A = (Bz[:, :, None, None] * By[:, None, :, None] * Bx[:, None, None, :])
    A = A.reshape(len(values), -1) 
    
    # Apply weights across all features
    A *= weights[:, None]

    # 4. Normal Equations
    n_feat = n_grid**3
    AtA = A.T @ A
    Atb = A.T @ weighted_values

    # 5. Clean and Solve
    del A, Bz, By, Bx, sdf, indices
    gc.collect()

    AtA.flat[::n_feat + 1] += 1e-3
    coeffs = np.linalg.solve(AtA, Atb)
    
    return coeffs.astype(np.float32)


def mask_from_features(coeffs, shape, n_grid=12, degree=3):
    nz, ny, nx = shape
    center = np.array([nz, ny, nx], dtype=np.float32) / 2.0
    scale = np.max([nz, ny, nx]).astype(np.float32) / 2.0
    
    pad = int(scale * 1.0)
    z_min, z_max = max(0, int(center[0]-pad)), min(nz, int(center[0]+pad))
    y_min, y_max = max(0, int(center[1]-pad)), min(ny, int(center[1]+pad))
    x_min, x_max = max(0, int(center[2]-pad)), min(nx, int(center[2]+pad))

    z_c = (np.arange(z_min, z_max) - center[0]) / scale
    y_c = (np.arange(y_min, y_max) - center[1]) / scale
    x_c = (np.arange(x_min, x_max) - center[2]) / scale
    
    # 1D Bases for ROI
    Bz = get_bspline_basis(z_c, n_grid, degree)
    By = get_bspline_basis(y_c, n_grid, degree)
    Bx = get_bspline_basis(x_c, n_grid, degree)

    C_tensor = coeffs.reshape((n_grid, n_grid, n_grid))

    # Fast Tensor Contraction
    recon_roi = np.einsum('ijk,zi,yj,xk->zyx', C_tensor, Bz, By, Bx, optimize='optimal')
    
    mask_roi = (recon_roi < 0)
    
    if np.any(mask_roi):
        labeled, n_components = label(mask_roi)
        if n_components > 1:
            rc = np.array([(z_max-z_min)/2, (y_max-y_min)/2, (x_max-x_min)/2])
            centers = np.array(center_of_mass(mask_roi, labeled, range(1, n_components+1)))
            mask_roi = (labeled == (np.argmin(np.linalg.norm(centers - rc, axis=1)) + 1))
    
    vol = np.zeros(shape, dtype=np.uint8)
    vol[z_min:z_max, y_min:y_max, x_min:x_max] = mask_roi
    return vol

def smooth_mask(mask:np.ndarray, n_grid=12):
    coeffs = features_from_mask(mask, n_grid)
    mask_rec = mask_from_features(coeffs, mask.shape, n_grid)
    return mask_rec