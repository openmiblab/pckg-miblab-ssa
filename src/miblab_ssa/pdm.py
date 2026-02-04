import numpy as np
import trimesh
from skimage.measure import marching_cubes
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, binary_dilation, binary_fill_holes

# ---------------------------------------------------------
# 1. Template & Mapping Utils
# ---------------------------------------------------------

def get_template_mesh(radius=1.0, subdivisions=3):
    """
    Returns the fixed topology icosphere.
    """
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    
    # Pre-calculate the spherical coordinates (Theta, Phi) for this template ONCE.
    # We will query these angles on the target kidney.
    verts = mesh.vertices
    # Normalize just in case
    verts /= np.linalg.norm(verts, axis=1)[:, None]
    
    # Convert vector -> angles
    # Mapping (x, y, z) -> (x, z, y) so Z is poles is standard for many libs, 
    # but let's stick to standard math: Z is up.
    Xc, Yc, Zc = verts[:,0], verts[:,1], verts[:,2]
    
    # Clip z to [-1, 1] to avoid NaNs at poles
    Theta = np.arccos(np.clip(Zc, -1, 1))
    Phi = np.arctan2(Yc, Xc) % (2*np.pi)
    
    return mesh, Theta, Phi

def get_inflated_mapping(verts, faces, iterations=80):
    """
    Iteratively smoothes the target mesh into a sphere.
    This 'unwraps' the concave kidney so we can map it to the template.
    """
    smooth_verts = verts.copy()
    n_verts = len(verts)
    
    # 1. Build Adjacency (Edges)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    
    # 2. Laplacian Smoothing (Inflation)
    for _ in range(iterations):
        # Calculate Face centers
        face_centers = smooth_verts[faces].mean(axis=1)
        
        # Accumulate neighbor forces
        v_sums = np.zeros_like(smooth_verts)
        v_counts = np.zeros(n_verts)
        
        np.add.at(v_sums, faces, face_centers[:, None, :])
        np.add.at(v_counts, faces, 1)
        
        v_counts[v_counts==0] = 1
        v_avgs = v_sums / v_counts[:, None]
        
        # Update (Relax towards average)
        smooth_verts = 0.5 * smooth_verts + 0.5 * v_avgs
        
        # Re-project to unit sphere (The "Inflation" step)
        smooth_verts -= smooth_verts.mean(axis=0)
        norms = np.linalg.norm(smooth_verts, axis=1)
        smooth_verts /= (norms[:, None] + 1e-9)

    # 3. Get Angles of the inflated target
    Xc, Yc, Zc = smooth_verts[:,0], smooth_verts[:,1], smooth_verts[:,2]
    Theta = np.arccos(np.clip(Zc, -1, 1))
    Phi = np.arctan2(Yc, Xc) % (2*np.pi)
    
    return Theta, Phi

# ---------------------------------------------------------
# 2. Main API
# ---------------------------------------------------------

def features_from_mask(mask: np.ndarray, subdivisions=3):
    """
    Extracts PDM features by mapping the Template Mesh to the Target.
    Preserves Concavities (Hollows).
    """
    # 1. Extract Target Mesh
    # Blur slightly to get smooth normals
    mask_float = gaussian_filter(mask.astype(float), sigma=1.0)
    verts_target, faces_target, _, _ = marching_cubes(mask_float, level=0.5)
    
    # 2. Center Target (Crucial for PCA)
    centroid = verts_target.mean(axis=0)
    verts_centered = verts_target - centroid
    
    # 3. Inflate Target (Get its spherical address map)
    # This finds the unique (Theta, Phi) for every vertex on the kidney
    Theta_target, Phi_target = get_inflated_mapping(verts_centered, faces_target)
    
    # 4. Get Template Angles
    # We want to move the Template vertices to the corresponding locations
    template_mesh, Theta_template, Phi_template = get_template_mesh(subdivisions=subdivisions)
    
    # 5. Interpolate
    # We use the Target's (Theta, Phi) -> (X, Y, Z) mapping
    # to look up the (X, Y, Z) for the Template's (Theta, Phi).
    
    # Fix the Seam at 2pi: Augment data
    T_aug = np.concatenate([Theta_target, Theta_target, Theta_target])
    P_aug = np.concatenate([Phi_target, Phi_target - 2*np.pi, Phi_target + 2*np.pi])
    V_aug = np.concatenate([verts_centered, verts_centered, verts_centered])
    
    points_in = np.column_stack((T_aug, P_aug))
    
    # Look up X, Y, Z coordinates for the template vertices
    feat_x = griddata(points_in, V_aug[:,0], (Theta_template, Phi_template), method='linear')
    feat_y = griddata(points_in, V_aug[:,1], (Theta_template, Phi_template), method='linear')
    feat_z = griddata(points_in, V_aug[:,2], (Theta_template, Phi_template), method='linear')
    
    # Fill any interpolation gaps (poles or sparse areas)
    mask_nan = np.isnan(feat_x)
    if np.any(mask_nan):
        feat_x[mask_nan] = griddata(points_in, V_aug[:,0], (Theta_template[mask_nan], Phi_template[mask_nan]), method='nearest')
        feat_y[mask_nan] = griddata(points_in, V_aug[:,1], (Theta_template[mask_nan], Phi_template[mask_nan]), method='nearest')
        feat_z[mask_nan] = griddata(points_in, V_aug[:,2], (Theta_template[mask_nan], Phi_template[mask_nan]), method='nearest')
        
    # 6. Flatten to Feature Vector
    # These are the vertices of the TEMPLATE, but moved to the KIDNEY surface
    return np.column_stack((feat_x, feat_y, feat_z)).flatten()


def mask_from_features(features: np.ndarray, shape, subdivisions=3):
    """
    Reconstructs the mask. Since features are derived from a Template Mesh,
    we can use the Template's faces to create a guaranteed watertight surface.
    """
    vol = np.zeros(shape, dtype=np.uint8)
    
    # 1. Rebuild Mesh Object
    ref_template, _, _ = get_template_mesh(subdivisions=subdivisions)
    n_verts = len(ref_template.vertices)
    
    if features.size != n_verts * 3:
         raise ValueError(f"Feature mismatch! Expected {n_verts*3}, got {features.size}. Check subdivision level.")

    vertices = features.reshape((n_verts, 3))
    
    # Center back to volume
    vol_center = np.array(shape) / 2.0
    vertices += vol_center
    
    # Create the mesh using Fixed Topology
    mesh = trimesh.Trimesh(vertices=vertices, faces=ref_template.faces)
    
    # 2. Robust Voxelization (Sample & Fill)
    # We perform dense sampling on the mesh surface to handle any folding
    # 50,000 points covers a 300^3 volume well without gaps
    samples, _ = trimesh.sample.sample_surface(mesh, count=50000)
    
    ix = np.round(samples[:, 0]).astype(int)
    iy = np.round(samples[:, 1]).astype(int)
    iz = np.round(samples[:, 2]).astype(int)
    
    # Bounds check
    valid = (ix >= 0) & (ix < shape[0]) & \
            (iy >= 0) & (iy < shape[1]) & \
            (iz >= 0) & (iz < shape[2])
            
    # Paint surface
    vol[ix[valid], iy[valid], iz[valid]] = 1
    
    # 3. Fill
    vol = binary_dilation(vol, iterations=2)
    vol = binary_fill_holes(vol)
    
    return vol.astype(np.uint8)

def smooth_mask(mask, subdivisions=3):
    feats = features_from_mask(mask, subdivisions=subdivisions)
    return mask_from_features(feats, mask.shape, subdivisions=subdivisions)