import numpy as np
from skimage import measure, morphology
from scipy.spatial import cKDTree
from scipy.ndimage import binary_fill_holes
from skimage import measure
import trimesh
import pyshtools as pysh
from skimage.measure import marching_cubes
from scipy.interpolate import griddata
from scipy.ndimage import binary_fill_holes


# ----------------------
# Utilities / preprocessing
# ----------------------

def preprocess_volume(vol, min_size=1000, closing_radius=2):
    """Simple morphological cleanup on binary volume."""
    vol = vol.astype(bool)
    vol = morphology.remove_small_objects(vol, min_size=min_size)
    if closing_radius > 0:
        vol = morphology.binary_closing(vol, morphology.ball(closing_radius))
    vol = binary_fill_holes(vol)
    return vol

def extract_mesh_from_volume(vol, spacing=(1.0,1.0,1.0), level=0.5):
    """
    Extract triangle mesh from binary volume using marching cubes.
    Returns verts (N,3) and faces (M,3).
    spacing: voxel spacing tuple (z,y,x) or (dx,dy,dz) according to skimage usage.
    """
    verts, faces, normals, values = measure.marching_cubes(vol.astype(np.uint8), level=level, spacing=spacing)
    return verts, faces



def dice_coefficient(vol_a, vol_b):
    """
    Compute Dice coefficient between two binary volumes.
    """
    vol_a = vol_a.astype(bool)
    vol_b = vol_b.astype(bool)
    intersection = np.logical_and(vol_a, vol_b).sum()
    size_a = vol_a.sum()
    size_b = vol_b.sum()
    if size_a + size_b == 0:
        return 1.0
    return 2.0 * intersection / (size_a + size_b)

def surface_distances(vol_a, vol_b, spacing=(1.0,1.0,1.0)):
    """
    Compute surface distances (Hausdorff and mean) between two binary volumes.
    Args:
      vol_a, vol_b: binary 3D arrays
      spacing: voxel spacing (dz,dy,dx)
    Returns:
      hausdorff, mean_dist
    """
    # extract meshes
    verts_a, faces_a, _, _ = measure.marching_cubes(vol_a.astype(np.uint8), level=0.5, spacing=spacing)
    verts_b, faces_b, _, _ = measure.marching_cubes(vol_b.astype(np.uint8), level=0.5, spacing=spacing)

    # build kd-trees
    tree_a = cKDTree(verts_a)
    tree_b = cKDTree(verts_b)

    # distances from A→B and B→A
    d_ab, _ = tree_b.query(verts_a, k=1)
    d_ba, _ = tree_a.query(verts_b, k=1)

    hausdorff = max(d_ab.max(), d_ba.max())
    mean_dist = 0.5 * (d_ab.mean() + d_ba.mean())
    return hausdorff, mean_dist








def sh_reconstruct_surface_from_coeffs(coeffs, lmax = 8):

    # Suppose coeffs is your SHCoeffs object (from pysh)
    coeffs_array = coeffs.to_array()[:, :lmax+1, :lmax+1]
    coeffs_trunc = pysh.SHCoeffs.from_array(coeffs_array)

    # Reconstruct the spherical function
    grid_recon = coeffs_trunc.expand(grid='DH')  # gives SHGrid
    radii_recon = grid_recon.to_array()

    # -----------------------
    # 2. Convert back to 3D surface
    # -----------------------
    # Define the same theta/phi grid
    n_theta, n_phi = radii_recon.shape
    theta = np.linspace(0, np.pi, n_theta)       # polar
    phi = np.linspace(0, 2*np.pi, n_phi)         # azimuth
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    # Convert spherical coords (r, theta, phi) -> Cartesian
    x = radii_recon * np.sin(theta_grid) * np.cos(phi_grid)
    y = radii_recon * np.sin(theta_grid) * np.sin(phi_grid)
    z = radii_recon * np.cos(theta_grid)

    # Flatten for point cloud
    points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    # -----------------------
    # 3. Create mesh from point cloud
    # -----------------------
    # Use alpha shape or ball pivoting; simplest: convex hull for now
    recon_mesh = trimesh.convex.convex_hull(points)

    # Save or show mesh
    recon_mesh.export("kidney_recon_lmax8.ply")






def power_spectrum(coeffs):
    # Real and rotationally invariant

    # Suppose we already computed coeffs from SH expansion (see previous example)
    # coeffs is an SHCoeffs object from pyshtools

    # Convert coefficients to array: shape (2, lmax+1, lmax+1)
    # axis 0: [0]=real part, [1]=imag part
    coeffs_array = coeffs.to_array()

    # -----------------------
    # 1. Compute rotation-invariant power spectrum
    # -----------------------
    lmax = coeffs.lmax
    power_spectrum = []

    for l in range(lmax + 1):
        # Get coefficients for this degree l across all orders m
        c_l = coeffs_array[:, l, :l+1]  # shape (2, l+1)
        # Combine real + imaginary into complex
        c_l_complex = c_l[0] + 1j * c_l[1]
        # Sum of squared magnitudes across m
        P_l = np.sqrt(np.sum(np.abs(c_l_complex)**2))
        power_spectrum.append(P_l)

    power_spectrum = np.array(power_spectrum)

    return power_spectrum


def cosine_similarity(a, b):
    # Simularity between two descriptor vectors - distance between shapes
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _sh_reconstruct_shape(vol_orig, spacing, lmax=200):

    # 1. Fit spherical harmonics
    coeffs, centroid, max_extent = sh_compute_coeffs(vol_orig, lmax=lmax)

    # coeffs contains complex SH coefficients = Fourier shape descriptors
    # -----------------------
    # 6. Use real descriptors for comparison
    # -----------------------
    descriptor_vector_real = coeffs.to_array().ravel().real
    descriptor_vector_power = power_spectrum(coeffs)

    vol_rec = sh_reconstruct_volume_from_coeffs(coeffs, vol_orig.shape, centroid, max_extent, lmax=lmax)

    # 4. Compute evaluation metrics
    dice = dice_coefficient(vol_orig, vol_rec)
    hausdorff, mean_dist = surface_distances(vol_orig, vol_rec, spacing=spacing)

    print('Dice: ', dice)

    return vol_rec, {'p1': [1, 'par1', '%', 'float'], 'p2': [1, 'par2', '%', 'float']}



def sh_compute_coeffs(volume, lmax=15):
    # -----------------------
    # 1. Original volume
    # -----------------------

    # Compute centroid and max extent
    coords = np.argwhere(volume)
    centroid = coords.mean(axis=0)
    max_extent = (coords.max(axis=0) - coords.min(axis=0)).max() / 2.0

    # -----------------------
    # 2. Normalize coordinates to unit sphere
    # -----------------------
    norm_coords = (coords - centroid) / max_extent
    Xc, Yc, Zc = norm_coords[:,0], norm_coords[:,1], norm_coords[:,2]
    R = np.sqrt(Xc**2 + Yc**2 + Zc**2)
    Theta = np.arccos(np.clip(Zc / R, -1, 1))       # polar angle
    Phi = np.arctan2(Yc, Xc) % (2*np.pi)            # azimuth

    # -----------------------
    # 3. Map points to 2D spherical grid
    # -----------------------
    n_theta, n_phi = 64, 128
    # n_theta, n_phi = 128, 256 # n_phi must be n_theta or 2*n_theta
    radii_grid = np.zeros((n_theta, n_phi))

    # Fill grid with max radius in each cell
    for i in range(norm_coords.shape[0]):
        t_idx = int(Theta[i] / np.pi * (n_theta-1))
        p_idx = int(Phi[i] / (2*np.pi) * (n_phi-1))
        radii_grid[t_idx, p_idx] = max(radii_grid[t_idx, p_idx], R[i])

    # -----------------------
    # 4. Create SHGrid and compute coefficients
    # -----------------------
    grid = pysh.SHGrid.from_array(radii_grid)
    coeffs = grid.expand()  # SHRealCoeffs
    
    coeffs_array = coeffs.to_array()[:, :lmax+1, :lmax+1]
    coeffs_trunc = pysh.SHCoeffs.from_array(coeffs_array)

    return coeffs_trunc, centroid, max_extent


def sh_reconstruct_volume_from_coeffs(coeffs, vol_shape, centroid, max_extent, lmax = 15):

    coeffs_array = coeffs.to_array()[:, :lmax+1, :lmax+1]
    coeffs_trunc = pysh.SHCoeffs.from_array(coeffs_array)

    # Assume coeffs is SHCoeffs object (from earlier steps)
    grid_recon = coeffs_trunc.expand(grid='DH')
    radii_recon = grid_recon.to_array()  # normalized unit-sphere radii

    # -----------------------
    # 4. Reconstruct 3D volume with original shape
    # -----------------------
    nx, ny, nz = vol_shape
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Coordinates relative to original centroid
    Xc = (X - centroid[0]) / max_extent
    Yc = (Y - centroid[1]) / max_extent
    Zc = (Z - centroid[2]) / max_extent

    R = np.sqrt(Xc**2 + Yc**2 + Zc**2)
    Theta = np.arccos(np.clip(Zc / (R + 1e-9), -1, 1))
    Phi = np.arctan2(Yc, Xc) % (2*np.pi)

    # Map spherical coordinates to SHGrid indices
    n_theta, n_phi = radii_recon.shape
    theta_idx = (Theta / np.pi * (n_theta-1)).astype(int)
    phi_idx   = (Phi / (2*np.pi) * (n_phi-1)).astype(int)
    r_boundary = radii_recon[theta_idx, phi_idx]

    # Binary reconstruction in original grid
    volume_recon = (R <= r_boundary).astype(np.uint8)

    return volume_recon


def sh_reconstruct_shape(mask_orig, lmax=200):

    coeffs, centroid, max_extent = sh_compute_coeffs(mask_orig, lmax=lmax)
    mask_rec = sh_reconstruct_volume_from_coeffs(coeffs, mask_orig.shape, centroid, max_extent, lmax=lmax)

    return mask_rec




import numpy as np
import pyshtools as pysh
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_dilation
from scipy.special import sph_harm

def get_inflated_mapping(verts, faces, iterations=80):
    """
    Inflates the mesh to a sphere to get unique (theta, phi) for every vertex.
    Preserves topology so we don't get overlapping rays in the hollow.
    """
    smooth_verts = verts.copy()
    # Simple adjacency graph
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    
    n_verts = len(verts)
    
    # Fast vectorized smoothing
    for _ in range(iterations):
        # Calculate face centers
        face_centers = smooth_verts[faces].mean(axis=1)
        
        # Accumulate neighbor pulls
        v_sums = np.zeros_like(smooth_verts)
        v_counts = np.zeros(n_verts)
        
        np.add.at(v_sums, faces, face_centers[:, None, :])
        np.add.at(v_counts, faces, 1)
        
        v_counts[v_counts==0] = 1
        v_avgs = v_sums / v_counts[:, None]
        
        # Move vertices
        smooth_verts = 0.5 * smooth_verts + 0.5 * v_avgs
        
        # Re-center and normalize to keep numerics healthy
        smooth_verts -= smooth_verts.mean(axis=0)
        rads = np.linalg.norm(smooth_verts, axis=1)
        smooth_verts /= (rads[:, None] + 1e-9)

    # Convert to Theta/Phi
    Xc, Yc, Zc = smooth_verts[:,0], smooth_verts[:,1], smooth_verts[:,2]
    Theta = np.arccos(np.clip(Zc, -1, 1))
    Phi = np.arctan2(Yc, Xc) % (2*np.pi)
    
    return Theta, Phi

def fit_coeffs_least_squares(verts, theta, phi, lmax):
    """
    Solves Y * C = V for C using Least Squares.
    Y is the matrix of spherical harmonics evaluated at (theta, phi).
    V is the vertex coordinates (x, y, or z).
    """
    # 1. Construct the Design Matrix Y (N_verts x N_coeffs)
    # Total coefficients = (lmax + 1)^2
    n_coeffs = (lmax + 1) ** 2
    n_verts = len(verts)
    
    # We construct Y column by column.
    # Mapping convention: pyshtools usually uses complex or real SH. 
    # For simplicity in reconstruction, we use real SH from scipy or manual.
    # BUT, to keep compatible with your pyshtools reconstruction, we should use pysh layouts.
    # However, building the pysh matrix manually is complex.
    # FAST PATH: We simply use standard Real Spherical Harmonics here.
    
    Y_matrix = np.zeros((n_verts, n_coeffs))
    
    col_idx = 0
    # Loop l from 0 to lmax
    for l in range(lmax + 1):
        # Loop m from -l to l
        for m in range(-l, l + 1):
            # Evaluate Real Spherical Harmonic
            # Scipy returns complex; we convert to Real Ylm
            harm = sph_harm(m, l, phi, theta)
            
            if m > 0:
                y_real = np.sqrt(2) * np.real(harm)
            elif m < 0:
                y_real = np.sqrt(2) * np.imag(harm)
            else: # m == 0
                y_real = np.real(harm)
                
            Y_matrix[:, col_idx] = y_real
            col_idx += 1
            
    # 2. Solve Least Squares: Y * c = x
    # We solve for X, Y, Z simultaneously
    # coeffs shape: (N_coeffs, 3)
    coeffs, residuals, rank, s = np.linalg.lstsq(Y_matrix, verts, rcond=None)
    
    return coeffs

def reconstruct_from_ls_coeffs(coeffs, lmax, vol_shape, centroid, max_extent, density=4):
    """
    Reconstructs volume from the Least Squares coefficients.
    """
    # 1. Create dense reconstruction grid
    recon_lmax = lmax * density
    theta_grid = np.linspace(0, np.pi, recon_lmax)
    phi_grid = np.linspace(0, 2*np.pi, recon_lmax * 2)
    T, P = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    
    # Flatten for matrix multiplication
    T_flat = T.flatten()
    P_flat = P.flatten()
    n_points = len(T_flat)
    n_coeffs = (lmax + 1) ** 2
    
    # 2. Build Reconstruction Matrix (reuse basis function logic)
    Y_recon = np.zeros((n_points, n_coeffs))
    col_idx = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            harm = sph_harm(m, l, P_flat, T_flat) # Note: scipy takes (m, l, phi, theta)
            if m > 0: y_r = np.sqrt(2) * np.real(harm)
            elif m < 0: y_r = np.sqrt(2) * np.imag(harm)
            else: y_r = np.real(harm)
            Y_recon[:, col_idx] = y_r
            col_idx += 1
            
    # 3. Compute Coordinates: X = Y_recon * c_x
    # coeffs is (N_coeffs, 3) -> X, Y, Z
    coords = Y_recon @ coeffs
    
    # Denormalize
    coords = (coords * max_extent) + centroid
    
    # 4. Voxelize
    ix = np.round(coords[:, 0]).astype(int)
    iy = np.round(coords[:, 1]).astype(int)
    iz = np.round(coords[:, 2]).astype(int)
    
    nx, ny, nz = vol_shape
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
    
    vol = np.zeros(vol_shape, dtype=np.uint8)
    vol[ix[valid], iy[valid], iz[valid]] = 1
    
    # Close gaps and fill
    vol = binary_dilation(vol, iterations=2)
    vol = binary_fill_holes(vol).astype(np.uint8)
    
    return vol

def reconstruct_shape_vsh(mask_orig, lmax=20):
    """
    Wrapper using Least Squares fitting to preserve concavities.
    """
    # 1. Mesh Extraction
    vol_float = gaussian_filter(mask_orig.astype(float), sigma=1.0)
    verts, faces, _, _ = marching_cubes(vol_float, level=0.5)
    
    centroid = verts.mean(axis=0)
    verts_centered = verts - centroid
    max_extent = np.max(np.linalg.norm(verts_centered, axis=1))
    verts_norm = verts_centered / max_extent
    
    # 2. Parameterization (Inflation)
    # This ensures unique angles for the hollow area
    theta, phi = get_inflated_mapping(verts_norm, faces, iterations=100)
    
    # 3. Least Squares Fitting
    # Fits the basis functions directly to the vertex positions.
    # This captures the "dent" much better than interpolation.
    coeffs = fit_coeffs_least_squares(verts_norm, theta, phi, lmax)
    
    # 4. Reconstruction
    mask_rec = reconstruct_from_ls_coeffs(coeffs, lmax, mask_orig.shape, centroid, max_extent)
    
    return mask_rec