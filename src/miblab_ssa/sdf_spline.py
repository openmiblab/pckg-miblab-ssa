import numpy as np
import gc
from scipy.ndimage import distance_transform_edt, label, center_of_mass
from itertools import product
from scipy.interpolate import BSpline

def get_bspline_basis(x, n_control, degree=3):
    # Clamped knots: standard practice for ensuring stability at -1 and 1
    knots = np.linspace(-1, 1, n_control - degree + 1)
    knots = np.pad(knots, (degree, degree), mode='constant', 
                   constant_values=(knots[0], knots[-1]))
    
    basis = np.zeros((len(x), n_control), dtype=np.float32)
    for i in range(n_control):
        c = np.zeros(n_control); c[i] = 1.0
        # extrapolate=False is critical to prevent the "box" expansion
        basis[:, i] = BSpline(knots, c, degree, extrapolate=False)(x)
    return np.nan_to_num(basis)

def features_from_mask(mask, order=12, degree=3, n_samples=60000):
    nz, ny, nx = mask.shape
    center = np.array([nz, ny, nx], dtype=np.float32) / 2.0
    scale = np.max([nz, ny, nx]).astype(np.float32) / 2.0

    sdf = (distance_transform_edt(1 - mask) - distance_transform_edt(mask)).astype(np.float32)
    
    # 1. Surface-Biased Sampling
    idx_boundary = np.argwhere(np.abs(sdf) < 3.0)
    idx_random = np.random.randint(0, [nz, ny, nx], size=(int(n_samples*0.2), 3))
    
    # 2. ANCHORING: Force the edges of the volume to be AIR
    # We create a grid of points on the outer faces of the volume
    edge_grid = np.linspace(0, 1, 10)
    anchors = np.array(list(product([0, nz-1], [0, ny-1], [0, nx-1]))) # Corners
    
    b_count = int(n_samples*0.8)
    idx_b_select = idx_boundary[np.random.choice(len(idx_boundary), b_count)]
    
    indices = np.vstack([idx_b_select, idx_random, anchors])
    values = sdf[indices[:, 0], indices[:, 1], indices[:, 2]]
    
    # 3. Weighting
    weights = np.exp(-np.abs(values) / 2.0).astype(np.float32)
    # Give anchors massive importance so the "cube" can't form
    weights[-len(anchors):] = 100.0 
    
    pts = ((indices - center) / scale).astype(np.float32)

    Bz = get_bspline_basis(pts[:, 0], order, degree)
    By = get_bspline_basis(pts[:, 1], order, degree)
    Bx = get_bspline_basis(pts[:, 2], order, degree)

    A = (Bz[:, :, None, None] * By[:, None, :, None] * Bx[:, None, None, :]).reshape(len(values), -1)
    A *= weights[:, None]
    weighted_values = values * weights

    # 4. Solve (Ridge alpha increased for stability)
    n_feat = order**3
    AtA = A.T @ A
    Atb = A.T @ weighted_values
    
    del A, Bz, By, Bx, sdf, indices
    gc.collect()

    AtA.flat[::n_feat + 1] += 1e-1 # High alpha to suppress edge noise
    coeffs = np.linalg.solve(AtA, Atb)
    
    return coeffs.astype(np.float32)

def mask_from_features(coeffs, shape, order=12, degree=3):
    nz, ny, nx = shape
    center = np.array([nz, ny, nx], dtype=np.float32) / 2.0
    scale = np.max([nz, ny, nx]).astype(np.float32) / 2.0
    
    z_c = (np.arange(nz) - center[0]) / scale
    y_c = (np.arange(ny) - center[1]) / scale
    x_c = (np.arange(nx) - center[2]) / scale
    
    Bz = get_bspline_basis(z_c, order, degree)
    By = get_bspline_basis(y_c, order, degree)
    Bx = get_bspline_basis(x_c, order, degree)

    C_tensor = coeffs.reshape((order, order, order))
    recon_sdf = np.einsum('ijk,zi,yj,xk->zyx', C_tensor, Bz, By, Bx, optimize='optimal')
    
    # Geometric Crop
    zz, yy, xx = np.meshgrid(z_c, y_c, x_c, indexing='ij')
    valid_box = (np.abs(zz) < 0.95) & (np.abs(yy) < 0.95) & (np.abs(xx) < 0.95)
    recon_sdf = (recon_sdf < 0) & valid_box

    # POST-PROCESS: Select component closest to the center
    if np.any(recon_sdf):
        labeled, num_features = label(recon_sdf)
        if num_features > 1:
            
            # Get sizes and centers of all components
            component_indices = np.arange(1, num_features + 1)
            centroids = center_of_mass(recon_sdf, labeled, component_indices)
            sizes = np.bincount(labeled.ravel())[1:]
            
            # Target center (in voxel coordinates)
            img_center = np.array(shape) / 2.0
            
            # Calculate distance from each component's center to image center
            distances = [np.linalg.norm(np.array(c) - img_center) for c in centroids]
            
            # HYBRID LOGIC: Pick the one that is both reasonably large AND central
            # Or simply the closest to center if clutter is very large:
            best_label = component_indices[np.argmin(distances)]
            
            recon_sdf = (labeled == best_label)

    return recon_sdf

def smooth_mask(mask:np.ndarray, order=16):
    coeffs = features_from_mask(mask, order=order)
    mask_rec = mask_from_features(coeffs, mask.shape, order)
    return mask_rec
