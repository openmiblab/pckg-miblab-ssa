import numpy as np
import gc
from scipy.ndimage import distance_transform_edt, label
from itertools import product
from scipy.interpolate import BSpline

def get_bspline_basis(x, n_control, degree=3):
    knots = np.linspace(-1, 1, n_control - degree + 1)
    knots = np.pad(knots, (degree, degree), mode='constant', 
                   constant_values=(knots[0], knots[-1]))
    basis = np.zeros((len(x), n_control), dtype=np.float32)
    for i in range(n_control):
        c = np.zeros(n_control); c[i] = 1.0
        basis[:, i] = BSpline(knots, c, degree, extrapolate=False)(x)
    return np.nan_to_num(basis)

def solve_spline_system(pts, values, weights, order, degree, alpha=1e-3):
    Bz = get_bspline_basis(pts[:, 0], order, degree)
    By = get_bspline_basis(pts[:, 1], order, degree)
    Bx = get_bspline_basis(pts[:, 2], order, degree)

    # Tensor product construction
    A = (Bz[:, :, None, None] * By[:, None, :, None] * Bx[:, None, None, :]).reshape(len(values), -1)
    A_weighted = A * weights[:, None]
    weighted_values = values * weights

    n_feat = order**3
    AtA = A_weighted.T @ A_weighted
    Atb = A_weighted.T @ weighted_values
    
    AtA.flat[::n_feat + 1] += alpha
    coeffs = np.linalg.solve(AtA, Atb)
    
    return coeffs.astype(np.float32)

def eval_at_pts(coeffs, pts, order, degree):
    """Helper to evaluate spline at specific point cloud coordinates"""
    Bz = get_bspline_basis(pts[:, 0], order, degree)
    By = get_bspline_basis(pts[:, 1], order, degree)
    Bx = get_bspline_basis(pts[:, 2], order, degree)
    C_tensor = coeffs.reshape((order, order, order))
    return np.einsum('ijk,pi,pj,pk->p', C_tensor, Bz, By, Bx)

def features_from_mask(mask, orders=(6, 12, 18), degree=3, n_samples=80000):
    nz, ny, nx = mask.shape
    center = np.array([nz, ny, nx], dtype=np.float32) / 2.0
    scale = np.max([nz, ny, nx]).astype(np.float32) / 2.0

    sdf = (distance_transform_edt(1 - mask) - distance_transform_edt(mask)).astype(np.float32)
    
    # Sampling (Increase n_samples for the 3rd level detail)
    idx_boundary = np.argwhere(np.abs(sdf) < 3.0)
    idx_random = np.random.randint(0, [nz, ny, nx], size=(int(n_samples*0.2), 3))
    anchors = np.array(list(product([0, nz-1], [0, ny-1], [0, nx-1])))
    
    b_count = int(n_samples*0.8)
    idx_b_select = idx_boundary[np.random.choice(len(idx_boundary), b_count)]
    indices = np.vstack([idx_b_select, idx_random, anchors])
    
    values = sdf[indices[:, 0], indices[:, 1], indices[:, 2]]
    weights = np.exp(-np.abs(values) / 5.0).astype(np.float32)
    weights[-len(anchors):] = 100.0 
    pts = ((indices - center) / scale).astype(np.float32)

    # --- LEVEL 1: GLOBAL ---
    c1 = solve_spline_system(pts, values, weights, orders[0], degree, alpha=1e-5)
    res1 = values - eval_at_pts(c1, pts, orders[0], degree)

    # --- LEVEL 2: MEDIUM ---
    c2 = solve_spline_system(pts, res1, weights, orders[1], degree, alpha=1e-3)
    res2 = res1 - eval_at_pts(c2, pts, orders[1], degree)

    # --- LEVEL 3: FINE ---
    c3 = solve_spline_system(pts, res2, weights, orders[2], degree, alpha=1e-1)

    # Concatenate into one 1D array for PCA
    return np.concatenate([c1.flatten(), c2.flatten(), c3.flatten()])

def mask_from_features(combined_coeffs, shape, orders=(6, 12, 18), degree=3):
    nz, ny, nx = shape
    center = np.array([nz, ny, nx], dtype=np.float32) / 2.0
    scale = np.max([nz, ny, nx]).astype(np.float32) / 2.0
    
    z_c = (np.arange(nz) - center[0]) / scale
    y_c = (np.arange(ny) - center[1]) / scale
    x_c = (np.arange(nx) - center[2]) / scale
    
    # Split concatenated array
    n1, n2, n3 = [o**3 for o in orders]
    c1 = combined_coeffs[:n1]
    c2 = combined_coeffs[n1:n1+n2]
    c3 = combined_coeffs[n1+n2:]

    def get_layer_sdf(c, order):
        Bz = get_bspline_basis(z_c, order, degree)
        By = get_bspline_basis(y_c, order, degree)
        Bx = get_bspline_basis(x_c, order, degree)
        return np.einsum('ijk,zi,yj,xk->zyx', c.reshape((order,order,order)), Bz, By, Bx)

    recon_sdf = get_layer_sdf(c1, orders[0]) + \
                get_layer_sdf(c2, orders[1]) + \
                get_layer_sdf(c3, orders[2])
    
    # Geometric Crop
    zz, yy, xx = np.meshgrid(z_c, y_c, x_c, indexing='ij')
    valid_box = (np.abs(zz) < 0.95) & (np.abs(yy) < 0.95) & (np.abs(xx) < 0.95)
    mask = (recon_sdf < 0) & valid_box
    
    # POST-PROCESS: Keep only the largest component (The Kidney)
    if np.any(mask):
        labeled, num_features = label(mask)
        if num_features > 1:
            # Sort components by size
            sizes = np.bincount(labeled.ravel())
            largest_label = sizes[1:].argmax() + 1
            mask = (labeled == largest_label)
            
    return mask.astype(bool)


def smooth_mask(mask:np.ndarray, orders=(6, 12, 18), degree=3):
    coeffs = features_from_mask(mask, orders=orders, degree=degree)
    mask_rec = mask_from_features(coeffs, mask.shape, orders=orders, degree=degree)
    return mask_rec