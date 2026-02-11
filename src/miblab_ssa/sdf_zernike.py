import numpy as np
import gc
from itertools import product
from scipy.ndimage import distance_transform_edt
from numpy.polynomial.legendre import legvander # Used for the radial component
from scipy.ndimage import label, center_of_mass

def features_from_mask(mask, order=15, n_samples=50000, random_seed=42):
    np.random.seed(random_seed)
    nz, ny, nx = mask.shape
    
    # 1. Coordinate System
    center = np.array([nz, ny, nx], dtype=np.float32) / 2.0
    scale = np.max([nz, ny, nx]).astype(np.float32) / 2.0

    # 2. Compute SDF
    sdf = (distance_transform_edt(1 - mask) - distance_transform_edt(mask)).astype(np.float32)
    
    # 3. Sampling
    boundary_mask = np.abs(sdf) < 3.0
    idx_boundary = np.argwhere(boundary_mask)
    idx_random = np.random.randint(0, [nz, ny, nx], size=(n_samples, 3))
    
    n_bound = int(n_samples * 0.5)
    idx_b_select = idx_boundary[np.random.choice(len(idx_boundary), n_bound, replace=True)] if len(idx_boundary) > 0 else idx_random[:n_bound]
    indices = np.vstack([idx_b_select, idx_random])
    values = sdf[indices[:, 0], indices[:, 1], indices[:, 2]]

    # 4. Convert to Spherical-style Radial Basis
    pts = ((indices - center) / scale).astype(np.float32)
    
    # Radius r must be <= 1.0 for Zernike stability
    r = np.linalg.norm(pts, axis=1)
    # Important: Clip r to 1.0 to avoid polynomial explosion at corners
    r_norm = np.clip(r, 0, 1.0)
    
    # Unit vectors for angular components
    # (Simplified Zernike uses Cartesian projections to maintain orthogonality)
    z_unit, y_unit, x_unit = pts[:, 0]/(r+1e-6), pts[:, 1]/(r+1e-6), pts[:, 2]/(r+1e-6)

    # 5. Build Basis Matrix
    # We use Radial (r) x Cartesian Projections (x,y,z)
    poly_indices = [(n, l, m) for n, l, m in product(range(order + 1), repeat=3) if n + l + m <= order]
    A = np.zeros((len(values), len(poly_indices)), dtype=np.float32)
    
    Tr = legvander(r_norm, order).astype(np.float32)
    Tz = legvander(z_unit, order).astype(np.float32)
    Ty = legvander(y_unit, order).astype(np.float32)
    Tx = legvander(x_unit, order).astype(np.float32)

    for idx, (n, l, m) in enumerate(poly_indices):
        # Radial component n, Angular components l, m
        A[:, idx] = Tr[:, n] * Tz[:, l] * Ty[:, m] * Tx[:, 0] # Simplified 3D Zernike approximation

    # 6. Solve Normal Equations
    AtA = A.T @ A
    Atb = A.T @ values
    AtA.flat[::len(poly_indices) + 1] += 1e-3 
    coeffs = np.linalg.solve(AtA, Atb)

    return coeffs.astype(np.float32)




def mask_from_features(coeffs, shape, order, chunk_size=100000):
    """
    Reconstructs mask from Zernike-style coefficients using memory chunks.
    """
    vol = np.zeros(shape, dtype=np.uint8)
    nz, ny, nx = shape
    center = np.array([nz, ny, nx], dtype=np.float32) / 2.0
    scale = np.max([nz, ny, nx]).astype(np.float32) / 2.0
    
    # 1. ROI boundaries
    pad = int(scale * 1.0)
    z_min, z_max = max(0, int(center[0]-pad)), min(nz, int(center[0]+pad))
    y_min, y_max = max(0, int(center[1]-pad)), min(ny, int(center[1]+pad))
    x_min, x_max = max(0, int(center[2]-pad)), min(nx, int(center[2]+pad))

    # 2. Setup Evaluation Points
    z_c = (np.arange(z_min, z_max) - center[0]) / scale
    y_c = (np.arange(y_min, y_max) - center[1]) / scale
    x_c = (np.arange(x_min, x_max) - center[2]) / scale
    
    zz, yy, xx = np.meshgrid(z_c, y_c, x_c, indexing='ij')
    pts_flat = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=-1)
    
    # Pre-calculate poly indices to match extraction
    poly_indices = [(n, l, m) for n, l, m in product(range(order + 1), repeat=3) 
                    if n + l + m <= order]
    
    recon_flat = np.zeros(pts_flat.shape[0], dtype=np.float32)

    # 3. Chunked Loop
    # We process 'chunk_size' voxels at a time to stay under 8GB
    for start in range(0, pts_flat.shape[0], chunk_size):
        end = min(start + chunk_size, pts_flat.shape[0])
        chunk = pts_flat[start:end]
        
        # Calculate Radial and Angular components for the chunk
        r = np.linalg.norm(chunk, axis=1)
        r_norm = np.clip(r, 0, 1.0)
        
        # Unit projections (with epsilon to avoid div by zero)
        uz, uy, ux = chunk[:, 0]/(r+1e-6), chunk[:, 1]/(r+1e-6), chunk[:, 2]/(r+1e-6)
        
        # Vandermonde for the chunk
        Tr = legvander(r_norm, order).astype(np.float32)
        Tz = legvander(uz, order).astype(np.float32)
        Ty = legvander(uy, order).astype(np.float32)
        Tx = legvander(ux, order).astype(np.float32)
        
        # Evaluate the polynomial sum for this chunk
        chunk_val = np.zeros(end - start, dtype=np.float32)
        for idx, (n, l, m) in enumerate(poly_indices):
            if coeffs[idx] != 0: # Sparsity optimization
                chunk_val += coeffs[idx] * (Tr[:, n] * Tz[:, l] * Ty[:, m])
        
        recon_flat[start:end] = chunk_val

    # 4. Reshape and Filter
    recon_roi = recon_flat.reshape(zz.shape)
    rr = np.sqrt(zz**2 + yy**2 + xx**2)
    
    # Zernike is only valid inside the unit sphere (r <= 1)
    mask_roi = (recon_roi < 0) & (rr <= 1.0)
    
    # 5. Connectivity Cleanup
    if np.any(mask_roi):
        labeled, n_components = label(mask_roi)
        if n_components > 1:
            local_center = np.array([(z_max-z_min)/2, (y_max-y_min)/2, (x_max-x_min)/2])
            centers = np.array(center_of_mass(mask_roi, labeled, range(1, n_components+1)))
            winner = np.argmin(np.linalg.norm(centers - local_center, axis=1)) + 1
            mask_roi = (labeled == winner)

    vol[z_min:z_max, y_min:y_max, x_min:x_max] = mask_roi
    return vol


def smooth_mask(mask:np.ndarray, order=20):
    coeffs = features_from_mask(mask, order=order)
    mask_rec = mask_from_features(coeffs, mask.shape, order)
    return mask_rec