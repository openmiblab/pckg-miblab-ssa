import numpy as np
from scipy.special import sph_harm, factorial
from tqdm import tqdm
from collections import defaultdict


def radial_poly(n, l, r):
    """
    Compute the radial polynomial R_nl(r) for 3D Zernike moments.

    Args:
        n (int): Radial order.
        l (int): Angular order.
        r (np.ndarray): Radial coordinates (0 <= r <= 1).

    Returns:
        np.ndarray: Radial polynomial evaluated at r.
    """
    rad = np.zeros_like(r, dtype=complex)
    for s in range((n - l) // 2 + 1):
        num = (-1)**s * factorial(n - s)
        den = (
            factorial(s)
            * factorial((n + l) // 2 - s)
            * factorial((n - l) // 2 - s)
        )
        rad += (num / den) * r**(n - 2 * s)
    return rad


def zernike_moments_3d(mask, n_max):
    """
    Computes 3D Zernike moments for a given 3D boolean mask.

    Args:
        mask (np.ndarray): 3D binary mask (dtype=bool).
        n_max (int): Maximum order of moments.

    Returns:
        dict: Dictionary of Zernike moments A_nml.
        tuple: (max_dist, centroid) used for normalization.
    """
    coords = np.argwhere(mask)
    if coords.size == 0:
        return {}, (0, np.zeros(3))

    centroid = np.mean(coords, axis=0)
    shifted_coords = coords - centroid
    max_dist = np.max(np.linalg.norm(shifted_coords, axis=1))

    if max_dist == 0:  # single point mask
        return {(0, 0, 0): np.sum(mask)}, (max_dist, centroid)

    # normalize coordinates into unit sphere
    normalized_coords = shifted_coords / max_dist
    x, y, z = normalized_coords.T
    r = np.sqrt(x**2 + y**2 + z**2)

    non_zero = r > 1e-9
    x, y, z, r = x[non_zero], y[non_zero], z[non_zero], r[non_zero]
    mask_values = mask[coords[:, 0], coords[:, 1], coords[:, 2]][non_zero]

    # spherical coords
    z_over_r = np.clip(z / r, -1.0, 1.0)
    theta = np.arccos(z_over_r)
    phi = np.arctan2(y, x)

    moments = {}
    for l in tqdm(range(n_max + 1), desc="Computing moments.."):
        for m in range(-l, l + 1):
            if (l - abs(m)) % 2 != 0:
                continue  # parity condition
            sph_h = sph_harm(m, l, phi, theta)  # computed once per (l,m)

            for n in range(l, n_max + 1, 2):  # n >= l, same parity
                rad = radial_poly(n, l, r)
                zernike_poly = rad * sph_h
                A_nml = np.sum(mask_values * np.conj(zernike_poly))
                moments[(n, m, l)] = A_nml

    return moments, (max_dist, centroid)




def dice_coefficient(a, b):
    """Compute Dice similarity coefficient between two boolean masks."""
    a = a.astype(bool)
    b = b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    return 2.0 * intersection / (a.sum() + b.sum() + 1e-9)




def reconstruct_volume_3d(moments, size, max_dist, centroid):
    """
    Reconstructs a 3D volume from Zernike moments.
    If reference_volume is provided, finds the threshold that maximizes similarity.

    Args:
        moments (dict): Zernike moments A_nml.
        size (tuple): Volume dimensions (z,y,x).
        max_dist (float): Normalization factor.
        centroid (np.ndarray): Centroid of mask.

    Returns:
        float: real reconstruction
    """
    zdim, ydim, xdim = size
    z_grid, y_grid, x_grid = np.ogrid[0:zdim, 0:ydim, 0:xdim]

    # Normalize coordinates
    x = (x_grid - centroid[2]) / max_dist
    y = (y_grid - centroid[1]) / max_dist
    z = (z_grid - centroid[0]) / max_dist
    r = np.sqrt(x**2 + y**2 + z**2)

    z_over_r = np.clip(z / (r + 1e-9), -1.0, 1.0)
    theta = np.arccos(z_over_r)
    phi = np.arctan2(y, x)

    # Reconstruct complex volume
    reconstructed_volume = np.zeros(size, dtype=complex)
    moments_by_lm = defaultdict(list)
    for (n, m, l), A_nml in moments.items():
        moments_by_lm[(l, m)].append((n, A_nml))

    for (l, m), nm_list in tqdm(moments_by_lm.items(), desc="Reconstructing.."):
        sph_h = sph_harm(m, l, phi, theta)
        for n, A_nml in nm_list:
            rad = radial_poly(n, l, r)
            reconstructed_volume += A_nml * rad * sph_h

    recon_real = np.real(reconstructed_volume)

    return recon_real







