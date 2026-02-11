import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import euclidean_distances
from scipy.ndimage import label, center_of_mass
from tqdm import tqdm

def features_from_mask(mask, grid_size=12, epsilon=2.0, n_samples=50000, random_seed=42):
    """
    Extracts RBF weights as shape features.
    grid_size: number of RBF centers per axis (12^3 = 1728 features).
    epsilon: width of the Gaussian kernel.
    """
    np.random.seed(random_seed)

    # 1. Define Fixed Coordinate System
    nz, ny, nx = mask.shape
    center = np.array([nz, ny, nx]) / 2.0
    scale = np.max([nz, ny, nx]) / 2.0

    # 2. Compute SDF
    sdf = (5.0 * np.tanh(distance_transform_edt(~mask) - 
                         distance_transform_edt(mask)) / 5.0).astype(np.float32)
    
    # 3. Sampling (Same as your robust strategy)
    boundary_mask = np.abs(sdf) < 3.0
    idx_boundary = np.argwhere(boundary_mask)
    idx_random = np.random.randint(0, [nz, ny, nx], size=(n_samples, 3))
    
    n_bound = int(n_samples * 0.5)
    idx_b_select = idx_boundary[np.random.choice(len(idx_boundary), n_bound, replace=True)]
    indices = np.vstack([idx_b_select, idx_random])
    
    z_idx, y_idx, x_idx = indices[:, 0], indices[:, 1], indices[:, 2]
    Values = sdf[z_idx, y_idx, x_idx]
    
    # 4. Normalize Coords to [-1, 1]
    pts = (indices - center) / scale
    valid_mask = np.all(np.abs(pts) <= 1, axis=1)
    pts = pts[valid_mask]
    Values = Values[valid_mask]

    # 5. Generate RBF Basis
    # Define a fixed grid of centers across the volume
    grid = np.linspace(-1, 1, grid_size)
    centers = np.stack(np.meshgrid(grid, grid, grid, indexing='ij'), axis=-1).reshape(-1, 3)

    # Compute distances between sampled points and fixed centers
    # Matrix A size: (n_samples, n_centers)
    dists = euclidean_distances(pts, centers)
    A = np.exp(-(epsilon * dists)**2) # Gaussian Kernel

    # 6. Solve for Weights (these are your PCA features)
    clf = Ridge(alpha=1e-3, fit_intercept=False)
    clf.fit(A, Values)
    
    return clf.coef_.astype(np.float32)




def mask_from_features(coeffs, shape, grid_size=12, epsilon=2.0):
    """
    Ultra-fast RBF reconstruction using Gaussian separability.
    
    Complexity: O(N^3 * K) instead of O(N^3 * K^3)
    Where N is grid resolution and K is RBF grid size.
    """
    vol = np.zeros(shape, dtype=np.uint8)
    nz, ny, nx = shape
    
    # 1. Coordinate System Setup
    center = np.array([nz, ny, nx]) / 2.0
    scale = np.max([nz, ny, nx]) / 2.0

    # 2. Define ROI
    pad = int(scale * 1.0)
    z_min, z_max = max(0, int(center[0]-pad)), min(nz, int(center[0]+pad))
    y_min, y_max = max(0, int(center[1]-pad)), min(ny, int(center[1]+pad))
    x_min, x_max = max(0, int(center[2]-pad)), min(nx, int(center[2]+pad))

    # 3. Generate 1D Normalized Coordinates
    z_c = (np.arange(z_min, z_max) - center[0]) / scale
    y_c = (np.arange(y_min, y_max) - center[1]) / scale
    x_c = (np.arange(x_min, x_max) - center[2]) / scale
    
    # 4. Generate 1D RBF Centers
    grid_1d = np.linspace(-1, 1, grid_size)
    
    # 5. Build 1D Kernel Matrices
    # Shape: (ROI_dimension, grid_size)
    phi_z = np.exp(-(epsilon * (z_c[:, None] - grid_1d[None, :]))**2)
    phi_y = np.exp(-(epsilon * (y_c[:, None] - grid_1d[None, :]))**2)
    phi_x = np.exp(-(epsilon * (x_c[:, None] - grid_1d[None, :]))**2)

    # 6. Reshape Weights to 3D Tensor
    weight_tensor = coeffs.reshape((grid_size, grid_size, grid_size))

    # 7. Fast Tensor Contraction (The Speed Engine)
    # k, j, i are indices for weight_tensor centers
    # z, y, x are indices for the output volume voxels
    recon_roi = np.einsum('kji,zk,yj,xi->zyx', 
                          weight_tensor, phi_z, phi_y, phi_x, 
                          optimize='optimal')

    # 8. Threshold and Post-Process
    # We rebuild a small grid just for the geometric crop
    zz, yy, xx = np.meshgrid(z_c, y_c, x_c, indexing='ij')
    valid_box = (np.abs(zz) <= 1.0) & (np.abs(yy) <= 1.0) & (np.abs(xx) <= 1.0)
    
    mask_roi = (recon_roi < 0) & valid_box
    
    if np.any(mask_roi):
        labeled, n_components = label(mask_roi)
        if n_components > 1:
            # ROI center in local coordinates
            rc = np.array([(z_max-z_min)/2, (y_max-y_min)/2, (x_max-x_min)/2])
            comp_centers = np.array(center_of_mass(mask_roi, labeled, range(1, n_components+1)))
            dists = np.linalg.norm(comp_centers - rc, axis=1)
            mask_roi = (labeled == (np.argmin(dists) + 1))

    vol[z_min:z_max, y_min:y_max, x_min:x_max] = mask_roi
    return vol

def smooth_mask(mask:np.ndarray, grid_size=12, epsilon=2.0):
    coeffs = features_from_mask(mask, grid_size, epsilon)
    mask_recon = mask_from_features(coeffs, mask.shape, grid_size, epsilon)
    return mask_recon