import numpy as np
import vreg
from scipy.ndimage import rotate
from tqdm import tqdm


from miblab_ssa.metrics import dice_coefficient


def rotate_to_ref(
    mask_ref: np.ndarray,
    mask_to_rotate: np.ndarray,
    axis: int = 2, #rotation in Z axis, (X,Y,Z)
    angle_range=(0.0, 360.0),
    angle_step: float = 1.0,
    verbose: int = 0,
):
    """
    Rotate a 3D mask around a selected axis and compute the Dice score
    across all angles. By default returns only the maximum Dice score.

    Parameters
    ----------
    mask_ref : np.ndarray
        Reference 3D binary mask (fixed).
    mask_to_rotate : np.ndarray
        Moving 3D binary mask to be rotated.
    axis : int, default=2
        Axis of rotation (0, 1, or 2).
    angle_range : tuple, default=(0.0, 360.0)
        Range of rotation angles.
    angle_step : float, default=1.0
        Step size in degrees.
    verbose : int, default=0
        If 0, no progress bar is shown. Any other value shows a progress bar.

    Returns
    -------
    best_rotate : np.ndarray
        Rotated mask
    best_angle : float
        Rotation angle
    """

    if mask_ref.shape != mask_to_rotate.shape:
        raise ValueError("mask_ref and mask_to_rotate must have the same shape")

    # Select rotation plane
    if axis == 0:
        rot_axes = (1, 2)
    elif axis == 1:
        rot_axes = (0, 2)
    elif axis == 2:
        rot_axes = (0, 1)
    else:
        raise ValueError("axis must be 0, 1, or 2")

    start, end = angle_range
    angles = np.arange(start, end, angle_step)

    best_dice = -1.0
    best_angle = None
    best_rotated = None

    for angle in tqdm(angles, desc='Finding rotation..', disable=verbose==0):
        rotated = rotate(
            mask_to_rotate,
            angle=angle,
            axes=rot_axes,
            reshape=False,
            order=0,         # nearest-neighbor for binary masks
            mode='constant',
            cval=0.0
        )

        d = dice_coefficient(mask_ref, rotated)

        if d > best_dice:
            best_dice = d
            best_angle = angle
            best_rotated = rotated

    # --- return logic ---
    return best_rotated, best_angle






def pca_affine(original_affine, centroid, eigvecs):
    """
    Build a new affine aligned to PCA axes.

    Args:
        original_affine (4x4): Input affine (voxel -> world).
        centroid (3,): PCA centroid in world coords.
        eigvecs (3x3): PCA eigenvectors, columns = axes in world.

    Returns:
        new_affine (4x4): Voxel -> PCA-aligned coords.
        transform_world_to_pca (4x4): Extra transform applied in world space.
    """
    # World -> PCA coords
    R = eigvecs.T   # rotation
    T = -centroid   # translation

    # Build 4x4 homogeneous transform world->PCA
    W2P = np.eye(4)
    W2P[:3,:3] = R
    W2P[:3,3] = R @ T

    return W2P @ original_affine








def second_vector_cone_sweep(mask, centroid, eigvec1, eigvec2, voxel_size, cone_angle=30, angle_step=1):
    """
    Find a direction lying in the plane of eigvec1 & eigvec2 that points to
    the half-cone with the least mass.
    
    Robustness Fix: Uses vector averaging for the best candidates to avoid 
    cyclic discontinuity issues at 0/180 degrees.
    """
    # 1. Normalize eigenvectors
    eigvec1 = eigvec1 / np.linalg.norm(eigvec1)
    eigvec2 = eigvec2 / np.linalg.norm(eigvec2)

    # 2. Voxel coordinates in physical units, shifted to centroid
    # Optimization: Pre-calculate norms once if memory allows, or keep as is.
    coords = np.argwhere(mask > 0).astype(float) * voxel_size
    coords -= centroid

    if coords.shape[0] == 0:
        return eigvec1

    cos_cone = np.cos(np.deg2rad(cone_angle / 2))
    
    candidates = [] # Store (mass, direction_vector)

    # 3. Sweep candidate directions
    # We sweep 0 to 180. The logic handles the orientation (d vs -d).
    for angle in np.arange(0, 180, angle_step):
        rad = np.deg2rad(angle)
        d = np.cos(rad) * eigvec1 + np.sin(rad) * eigvec2
        d /= np.linalg.norm(d)

        # Optimization: fast dot product
        # Ideally, pre-normalized unit_coords would be faster if memory permits
        dots = coords @ d
        # We need normalized dot product for the angle check
        # But for 'left/right' mass, we just need the sign of the projection
        
        # Calculate angle check efficiently
        coord_norms = np.linalg.norm(coords, axis=1)
        valid = coord_norms > 0
        
        # Cosine similarity
        cos_sim = np.zeros_like(dots)
        cos_sim[valid] = np.abs(dots[valid]) / coord_norms[valid] # abs for double cone
        
        # Identify voxels in the double-cone
        mask_in_cone = cos_sim >= cos_cone
        
        if not np.any(mask_in_cone):
            continue

        # Project selected voxels onto the axis 'd' to check side (Left vs Right)
        proj = dots[mask_in_cone]
        
        left_mass = np.sum(proj < 0)
        right_mass = np.sum(proj >= 0)
        
        current_min_mass = min(left_mass, right_mass)
        
        # Determine correct orientation for this specific axis
        # If left (negative proj) is lighter, we want to point NEGATIVE.
        d_oriented = -d if left_mass < right_mass else d
        
        candidates.append((current_min_mass, d_oriented))

    if not candidates:
        return eigvec1

    # 4. Find global minimum mass
    min_mass = min(c[0] for c in candidates)

    # 5. Collect ALL vectors that achieved this minimum mass
    # This handles the "plateau" of equal values
    best_vectors = [c[1] for c in candidates if c[0] == min_mass]

    # 6. Compute the Mean Vector
    # Instead of sorting angles, we average the 3D vectors.
    # This correctly interpolates across the 0/180 degree wrap-around.
    mean_vector = np.mean(best_vectors, axis=0)
    
    # Handle rare cancellation case (e.g., vectors pointing exactly opposite)
    if np.linalg.norm(mean_vector) < 1e-6:
        return best_vectors[0] # Fallback to first candidate

    best_vec = mean_vector / np.linalg.norm(mean_vector)

    return best_vec



def orient_vector_with_cone(mask, centroid, vector, voxel_size, cone_angle=30):
    """
    Orients the given vector so that it points toward the 'lighter' end of the object,
    considering only the mass contained within a cone aligned with the vector.

    Parameters:
    -----------
    mask : np.ndarray
        3D binary mask.
    centroid : np.ndarray
        (3,) physical centroid.
    vector : np.ndarray
        (3,) unit vector (e.g., the principal axis) to orient.
    voxel_size : tuple/list
        (3,) physical voxel dimensions.
    cone_angle : float
        The aperture of the cone in degrees.

    Returns:
    --------
    np.ndarray
        The oriented unit vector pointing to the lighter side.
    float
        The mass the vector is pointing to
    """
    # 1. Normalize the input vector
    vec = vector / np.linalg.norm(vector)

    # 2. Get centered physical coordinates
    indices = np.argwhere(mask)
    if indices.shape[0] == 0:
        return vec
        
    points = indices * np.array(voxel_size)
    points_centered = points - centroid

    # 3. Calculate norms (distances from centroid)
    points_norm = np.linalg.norm(points_centered, axis=1)
    
    # Avoid division by zero for the voxel exactly at the centroid
    valid_points_mask = points_norm > 1e-9
    
    if not np.any(valid_points_mask):
        return vec

    # Filter arrays to exclude the center point
    p_centered = points_centered[valid_points_mask]
    p_norm = points_norm[valid_points_mask]

    # 4. Calculate Dot Products and Angles
    # Dot product of points with the axis vector
    dots = p_centered @ vec
    
    # Cosine of the angle between point and vector: (A . B) / (|A| |B|)
    # Since B is unit vector: (A . B) / |A|
    cos_angles = dots / p_norm

    # 5. Define the Cone Threshold
    # We use abs(cos_angles) to catch points in BOTH the forward cone 
    # and the backward cone (the "double cone").
    threshold = np.cos(np.deg2rad(cone_angle / 2))
    in_cone_mask = np.abs(cos_angles) >= threshold

    # If the cone is too narrow and catches nothing, return original
    if not np.any(in_cone_mask):
        return vec

    # 6. Count Mass in Forward vs Backward Cones
    # We look at the sign of the dot product for points inside the cone.
    # dot > 0 : Forward Cone (Direction of 'vec')
    # dot < 0 : Backward Cone (Direction of '-vec')
    valid_dots = dots[in_cone_mask]
    
    mass_forward = np.sum(valid_dots > 0)
    mass_backward = np.sum(valid_dots < 0)
    skew = np.abs(mass_forward - mass_backward) / (mass_forward + mass_backward)

    # 7. Orientation Logic
    # We want to point to the LEAST mass.
    if mass_forward > mass_backward:
        # The direction we are currently pointing to is heavier. Flip it.
        return -vec, skew
    
    # Otherwise, current direction is lighter (or equal). Keep it.
    return vec, skew


def kidney_canonical_axes(mask, centroid, eigvecs, voxel_size):
    """
    Returns orthogonal vectors in a canonical reference frame optimized 
    for normalization of kidney shapes.

    - The z-axis (kidney longitudinal) is the principal eigenvector oriented fro  head to foot
    - The x-axis (kidney sagittal) is perpendicular to the longitudinal axis, through the 
      centroid and pointing in the direction with least mass in a small cone around the axis
    - The y-axis (kidney transverse) is derived from the first two so as to make a right-handed 
      X,Y,Z Cartesian reference frame
    """

    # Find eigenvector along FH 
    foot_to_head = [0,-1,0]
    longitudinal_axis = eigvecs[:, 0].copy()
    if np.dot(longitudinal_axis, foot_to_head) < 0:
        longitudinal_axis *= -1

    # Decide direction of the secondary axes
    # sagittal_axis = second_vector_cone_sweep(mask, centroid, eigvecs[:,1], eigvecs[:,2], voxel_size)
    axis1, skew1 = orient_vector_with_cone(mask, centroid, eigvecs[:,1], voxel_size, cone_angle=90)
    axis2, skew2 = orient_vector_with_cone(mask, centroid, eigvecs[:,2], voxel_size, cone_angle=90)
    if skew1 > skew2:
        sagittal_axis = axis1
    else:
        sagittal_axis = axis2

    # Compute transversal axis
    transversal_axis = np.cross(longitudinal_axis, sagittal_axis)

    # Build new eigenvectors
    canonical_axes = np.zeros((3,3))  
    canonical_axes[:,0] = sagittal_axis 
    canonical_axes[:,1] = transversal_axis
    canonical_axes[:,2] = longitudinal_axis

    return canonical_axes


def inertia_principal_axes(mask, voxel_size=(1.0, 1.0, 1.0)):
    """
    Calculates the physical centroid and normalized, right-handed principal axes 
    of a 3D binary mask.

    Parameters:
    -----------
    mask : np.ndarray
        A 3D binary array (boolean or 0/1) where True indicates the object.
    voxel_size : tuple or list of 3 floats
        The voxel size in physical units (e.g., [dz, dy, dx] or [dx, dy, dz]).
        Must match the order of dimensions in the mask array.

    Returns:
    --------
    centroid : np.ndarray
        A (3,) array containing the center of mass in physical coordinates.
    axes : np.ndarray
        A (3, 3) matrix where each column represents a principal axis vector.
        - Column 0: Major axis (largest variance)
        - Column 1: Intermediate axis
        - Column 2: Minor axis (smallest variance)
        The system is guaranteed to be right-handed and normalized.
    """
    # 1. Get indices of the non-zero voxels (N, 3)
    indices = np.argwhere(mask)
    
    if indices.shape[0] < 3:
        raise ValueError("Mask must contain at least 3 points to define 3D axes.")

    # 2. Convert to physical coordinates
    # We multiply the indices by the voxel_size to handle anisotropic voxels correctly
    points = indices * np.array(voxel_size)

    # 3. Calculate Centroid
    centroid = np.mean(points, axis=0)

    # 4. Center the points (subtract centroid)
    points_centered = points - centroid

    # 5. Compute Covariance Matrix
    # rowvar=False means each column is a variable (x, y, z), each row is an observation
    cov_matrix = np.cov(points_centered, rowvar=False)

    # 6. Eigendecomposition
    # eigh is optimized for symmetric matrices (like covariance matrices)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 7. Sort axes by Eigenvalues (descending order)
    # eigh returns eigenvalues in ascending order, so we reverse them
    # We want: Index 0 = Major Axis (Largest eigenvalue)
    sort_indices = np.argsort(eigenvalues)[::-1]
    
    sorted_eigenvalues = eigenvalues[sort_indices]
    sorted_vectors = eigenvectors[:, sort_indices]

    # 8. Enforce Right-Handed Coordinate System
    # Calculate the determinant. If -1, the system is left-handed (reflection).
    # We flip the minor axis (last column) to fix this.
    if np.linalg.det(sorted_vectors) < 0:
        sorted_vectors[:, -1] *= -1

    return centroid, sorted_vectors, sorted_eigenvalues

# This works for any image but needed for masks
def _inertia_principal_axes(volume, voxel_size=(1.0,1.0,1.0), eps=1e-12):
    """
    Compute intensity-weighted center of mass and inertia (second-moment) tensor,
    and return principal axes (eigenvectors) and eigenvalues.

    Args:
        volume (ndarray): 3D numpy array (x,y,z) of intensities (non-negative typically).
        voxel_size (tuple[float] or np.ndarray): physical voxel spacing (dx, dy, dz).
        eps (float): small value to avoid division by zero.

    Returns:
        centroid_phys (ndarray shape (3,)): intensity-weighted centroid in physical coordinates (x,y,z).
        eigvals (ndarray shape (3,)): eigenvalues (descending).
        eigvecs (ndarray shape (3,3)): eigenvectors as columns; eigvecs[:,i] is eigenvector for eigvals[i].
        inertia (ndarray shape (3,3)): the computed inertia / second-moment matrix used.
    """
    voxel_size = np.asarray(voxel_size, dtype=float)
    if voxel_size.size != 3:
        raise ValueError("voxel_size must be length-3 (dx,dy,dz)")

    # Get indices and intensities for non-zero (or all) voxels
    # Using all voxels is fine; we can optionally ignore zeros if desired by the user upstream.
    coords_idx = np.argwhere(volume != 0)   # indices in (x,y,z) order
    if coords_idx.size == 0:
        raise ValueError("Volume contains no nonzero voxels.")

    intensities = volume[coords_idx[:,0], coords_idx[:,1], coords_idx[:,2]].astype(float)
    total_mass = intensities.sum()
    if total_mass <= eps:
        raise ValueError("Total intensity (mass) is zero or too small.")

    coords_phys = coords_idx.astype(float) * voxel_size  # shape (N,3) in (x,y,z)

    # Intensity-weighted centroid (center of mass)
    centroid_phys = (coords_phys * intensities[:,None]).sum(axis=0) / total_mass

    # Centralized coordinates
    delta = coords_phys - centroid_phys  # (N,3)

    # Compute the 3x3 inertia / second moment matrix (covariance-like, but with mass)
    # We compute the second central moments: M = sum_i w_i * (delta_i ⊗ delta_i)
    # This is essentially the (unnormalized) weighted covariance multiplied by total_mass.
    # Optionally normalize by total_mass to get weighted covariance; eigenvectors same either way.
    M = np.einsum('ni,nj->ij', delta * intensities[:,None], delta)  # shape (3,3)

    # If you prefer the covariance form: M_cov = M / total_mass
    # For axis extraction eigenvectors are identical up to eigenvalue scaling.
    # We'll compute eigen-decomposition of M (symmetric)
    # Use np.linalg.eigh (for symmetric matrices)
    eigvals_raw, eigvecs_raw = np.linalg.eigh(M)  # ascending order

    # Sort descending
    idx = np.argsort(eigvals_raw)[::-1]
    eigvals = eigvals_raw[idx]
    eigvecs = eigvecs_raw[:, idx]

    # Return inertia matrix as used (unnormalized). If user wants covariance: M / total_mass
    return centroid_phys, eigvecs, eigvals


def normalize_kidney_mask(mask, voxel_size, side, ref=None, verbose=1):
    """
    Normalize a 3D binary mask using mesh-based PCA alignment and scaling.
    Centers the mesh in the middle of the volume grid.
    """
    # TODO: return rotation angles versus patient reference frame (obliqueness).

    if side not in ['left', 'right']:
        raise ValueError(
            f"The side argument must be either 'left' or 'right'. "
            f"You have entered {side}. "
        )
    if np.size(voxel_size) != 3:
        raise ValueError("Voxel size must have 3 elements.")
    if np.ndim(mask) != 3:
        raise ValueError("mask must be 3D. ")
 
    # voxel size in mm
    # target_volume in mm3
    target_spacing = 1.0
    target_volume = 1e6

    # Optional mirroring
    if side == 'left':
        mask = np.flip(mask, 0)

    # Build volume with identity affine and a corner on the origina
    volume = vreg.volume(mask.astype(float), spacing=voxel_size)

    # Align principal axes to reference frame
    centroid, eigvecs, eigvals = inertia_principal_axes(mask, voxel_size)
    canonical_axes = kidney_canonical_axes(mask, centroid, eigvecs, voxel_size)
    canonical_affine = pca_affine(volume.affine, centroid, canonical_axes)
    volume.set_affine(canonical_affine)

    # Scale to target volume
    voxel_volume = np.prod(voxel_size)
    current_volume = mask.sum() * voxel_volume
    scale = (target_volume / current_volume) ** (1/3)
    volume = volume.stretch(scale)

    # Resample on standard isotropic volume
    target_length = 3 * (target_volume ** (1/3)) # length in mm
    target_dim = np.ceil(1 + target_length / target_spacing).astype(int) # length in mm
    target_shape = 3 * [target_dim]
    target_affine = np.diag(3 * [target_spacing] + [1.0])
    target_affine[:3,3] = 3 * [- target_spacing * (target_dim - 1) / 2]
    volume = volume.slice_like((target_shape, target_affine))

    # Convert back to mask 
    mask_norm = volume.values > 0.5

    params = { # Derive biomarkers on angulation vs ref axes
        "centroid": centroid,
        "eigvecs": eigvecs,
        "eigvals": eigvals,
        "scale": scale,
        "canonical_axes": canonical_axes,
        "canonical_affine": canonical_affine,
        "scale": scale,
        "output_affine": target_affine,
    }

    # If a reference is provided, rotate to match
    if ref is not None:
        mask_norm, ref_rotation_angle = rotate_to_ref(ref, mask_norm, angle_step=5, verbose=verbose) 
        params["ref_rotation"] = ref_rotation_angle

    return mask_norm, params

