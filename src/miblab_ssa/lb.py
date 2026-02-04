import numpy as np
from skimage import measure
import trimesh
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh

# -------------------------------
# Helper: convert trimesh to mask
# -------------------------------
def mesh_to_mask(mesh, shape):
    """
    Rasterize mesh into 3D binary mask
    """
    mask = np.zeros(shape, dtype=bool)
    # Use trimesh voxelization
    vox = mesh.voxelized(pitch=1.0)
    indices = vox.sparse_indices
    mask[indices[:,0], indices[:,1], indices[:,2]] = True
    return mask

# -------------------------------
# 1️⃣ Mask → Mesh
# -------------------------------
def mask_to_mesh(mask, spacing=(1.0,1.0,1.0)):
    """
    Convert 3D binary mask to triangular mesh.
    """
    verts, faces, normals, values = measure.marching_cubes(mask.astype(float), level=0.5, spacing=spacing)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return mesh


def mask_to_mesh_fixed_vertices(mask: np.ndarray, spacing: np.ndarray, target_vertices: int = 5000) -> trimesh.Trimesh:
    """
    Convert a 3D binary mask to a mesh with a fixed number of vertices.

    Parameters
    ----------
    center : bool
        If True, center the mesh at the origin.
    spacing : np.ndarray
        Voxel size

    Returns
    -------
    mesh_simplified : trimesh.Trimesh
        Mesh object with approximately target_vertices vertices.
    """
    # Step 1: extract surface using marching cubes
    verts, faces, normals, _ = measure.marching_cubes(mask.astype(float), level=0.5, spacing=spacing)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)

    # Step 2: simplify / resample to target number of vertices
    # Needs testing
    mesh_simplified = mesh.simplify_quadratic_decimation(target_vertices)

    return mesh_simplified


# -------------------------------
# 2️⃣ Preprocessing for invariance (FIXED)
# -------------------------------
def preprocess_mesh(mesh):
    """
    Apply translation, scaling, and PCA alignment.
    Returns processed mesh and preprocessing parameters for inverse mapping.
    """
    # Center
    centroid = mesh.vertices.mean(axis=0)
    mesh_c = mesh.copy()
    mesh_c.vertices = mesh.vertices - centroid

    # Scale
    scale = np.sqrt((mesh_c.vertices**2).sum(axis=1).mean())
    mesh_s = mesh_c.copy()
    mesh_s.vertices = mesh_c.vertices / scale

    # PCA alignment
    cov = np.cov(mesh_s.vertices.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    mesh_aligned = mesh_s.copy()
    mesh_aligned.vertices = mesh_s.vertices @ eigvecs

    # Save parameters for inverse transformation
    params = {"centroid": centroid, "scale": scale, "pca_eigvecs": eigvecs}
    return mesh_aligned, params

def inverse_preprocess_mesh(vertices, params):
    """
    Map reconstructed vertices back to original coordinates.
    """
    v = vertices @ params["pca_eigvecs"].T  # undo PCA
    v = v * params["scale"]                # undo scaling
    v = v + params["centroid"]             # undo translation
    return v

# -------------------------------
# 3️⃣ Laplace-Beltrami Eigenfunctions
# -------------------------------
def cotangent_laplacian(mesh):
    vertices = mesh.vertices
    faces = mesh.faces

    def cotangent(a, b, c):
        ba = b - a
        ca = c - a
        cos_angle = np.dot(ba, ca)
        sin_angle = np.linalg.norm(np.cross(ba, ca))
        return cos_angle / (sin_angle + 1e-10)

    I, J, V = [], [], []
    n = len(vertices)
    for face in faces:
        i, j, k = face
        vi, vj, vk = vertices[i], vertices[j], vertices[k]
        cot_alpha = cotangent(vj, vi, vk)
        cot_beta  = cotangent(vk, vj, vi)
        cot_gamma = cotangent(vi, vk, vj)
        for (p, q, w) in [(i,j,cot_gamma),(j,i,cot_gamma),
                          (j,k,cot_alpha),(k,j,cot_alpha),
                          (k,i,cot_beta),(i,k,cot_beta)]:
            I.append(p)
            J.append(q)
            V.append(w/2)

    L = coo_matrix((V, (I, J)), shape=(n, n))
    L = diags(L.sum(axis=1).A1) - L
    return L
        
def lb_eigen_decomposition(mesh, k=50):
    L = cotangent_laplacian(mesh)
    M = diags(np.ones(mesh.vertices.shape[0]))
    eigvals, eigvecs = eigsh(L, k=k, M=M, sigma=1e-8, which='LM')
    return eigvals, eigvecs

def surface_to_coefficients(mesh, k=50):
    eigvals, eigvecs = lb_eigen_decomposition(mesh, k=k)
    coords = mesh.vertices
    coeffs = eigvecs.T @ coords  # shape (k,3)
    return coeffs, eigvecs, eigvals


def rotationally_invariant_lb_coeffs(coeffs, eigvals, k=100):
    """
    Compute rotationally invariant Laplace–Beltrami spectral coefficients.

    Parameters
    ----------
    mesh : trimesh.Trimesh or similar
        Input surface mesh with vertices (N, 3)
    k : int
        Number of eigenmodes to use

    Returns
    -------
    eigvals : (k,) array
        Laplace–Beltrami eigenvalues
    invariants : (k,) array
        Rotationally invariant spectral coefficients
    """
    invariants = np.linalg.norm(coeffs, axis=1)  # sqrt(sum over x,y,z)
    invariants /= np.linalg.norm(invariants)

    # Optional: normalize eigenvalues by first non-zero eigenvalue
    eigvals = eigvals / eigvals[1] if eigvals[1] != 0 else eigvals

    # Optionally drop the first eigenvalue (zero mode) from descriptor since it's trivial
    eigvals = eigvals[1:]  # length k-1
    invariants = invariants[1:]  # skip first mode as it may be trivial

    descriptor = np.concatenate([eigvals[:k], invariants[:k]])
    descriptor /= np.linalg.norm(descriptor)  # normalize final vector

    return invariants, eigvals


# def coefficients_to_surface(coeffs, eigvecs):
#     reconstructed = eigvecs @ coeffs
#     return reconstructed

def coefficients_to_surface(coeffs, eigvecs, threshold=None):
    """
    Reconstruct surface vertices from coefficients and eigenvectors.
    
    Args:
        coeffs (np.ndarray): shape (k, 3), coefficients from surface_to_coefficients
        eigvecs (np.ndarray): shape (n, k), eigenvectors of Laplace-Beltrami
        threshold (float, optional): percentage (0-100).
            If given, only the top threshold% dominant modes (by coefficient norm)
            are kept in the reconstruction.
    
    Returns:
        np.ndarray: reconstructed vertices, shape (n, 3)
    """
    if threshold is not None:
        # Compute importance of each eigenfunction
        norms = np.linalg.norm(coeffs, axis=1)
        k = len(norms)

        # How many to keep
        keep = max(1, int(np.ceil(k * threshold / 100.0)))

        # Select indices of the most important modes
        idx_sorted = np.argsort(norms)[::-1]
        idx_keep = idx_sorted[:keep]

        # Zero out the others
        coeffs_filtered = np.zeros_like(coeffs)
        coeffs_filtered[idx_keep] = coeffs[idx_keep]

        reconstructed = eigvecs @ coeffs_filtered
    else:
        reconstructed = eigvecs @ coeffs

    return reconstructed


def pipeline(mask, k=50):
    # mesh = mask_to_mesh(mask)
    # Fixed number of vertices is necessary to achieve comparable coefficients
    mesh = mask_to_mesh_fixed_vertices(mask)
    mesh_proc, params = preprocess_mesh(mesh)
    coeffs, eigvecs, eigvals = surface_to_coefficients(mesh_proc, k=k)
    return coeffs, eigvecs, mesh_proc, params

def eigvals(mask, k=100, normalize=False):
    mesh = mask_to_mesh(mask)
    coeffs, eigvecs, eigvals = surface_to_coefficients(mesh, k=k)
    if normalize:
        # Normalize eigenvalues by first non-zero eigenvalue
        # eigvals = eigvals / eigvals[1] if eigvals[1] != 0 else eigvals
        eigvals = eigvals / np.max(eigvals)
        # Drop the first eigenvalue (zero mode) from descriptor since it's trivial
        eigvals = eigvals[1:]  # length k-1
    return eigvals


def process(mesh, k=10, threshold=None):
    mesh_proc, params = preprocess_mesh(mesh)

    # Compute LB coefficients (invariant)
    coeffs, eigvecs, eigvals = surface_to_coefficients(mesh_proc, k=k)

    # Reconstruct in normalized/aligned space
    reconstructed_vertices_proc = coefficients_to_surface(coeffs, eigvecs, threshold=threshold)

    # Map reconstruction back to original coordinates
    reconstructed_vertices_orig = inverse_preprocess_mesh(reconstructed_vertices_proc, params)

    # Build reconstructed mesh
    reconstructed_mesh = mesh.copy()
    reconstructed_mesh.vertices = reconstructed_vertices_orig  

    return coeffs, eigvals, reconstructed_mesh  



