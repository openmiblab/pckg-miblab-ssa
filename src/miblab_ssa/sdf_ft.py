import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.fft import dctn, idctn  




def get_spectral_mask(shape, radius):
    """
    Generates a boolean mask for spherical truncation.
    shape: The shape of the coefficient block (e.g., (16, 16, 16))
    radius: The spectral radius to keep (e.g., 16.0)
    """
    # 1. strictly use the 'shape' argument to determine grid dimensions
    ranges = [np.arange(n) for n in shape]
    
    # 2. Create the grid of frequencies (i, j, k)
    I, J, K = np.meshgrid(*ranges, indexing='ij')
    
    # 3. Calculate distance from DC component (0,0,0)
    # i^2 + j^2 + k^2 <= r^2
    mask = (I**2 + J**2 + K**2) <= radius**2
    
    return mask

def features_from_mask(mask: np.ndarray, order=16, norm="ortho", saturation_threshold=5.0):
    # 1. Compute SDF & Saturate
    mask = mask.astype(bool)
    # Using float32 saves memory/speed for DCT without losing precision relevant for shape
    sdf = (distance_transform_edt(~mask) - distance_transform_edt(mask)).astype(np.float32)
    sdf_saturated = saturation_threshold * np.tanh(sdf / saturation_threshold)

    # 2. Compute coefficients
    full_coeffs = dctn(sdf_saturated, norm=norm, workers=-1)
    
    # 3. Cubic Truncation first (Efficiency)
    # We slice out the corner cube first to avoid generating a mask for the massive 300^3 volume
    cube_coeffs = full_coeffs[:order, :order, :order]
    
    # 4. Spherical Masking
    # Now we generate the mask specifically for this cube shape
    mask = get_spectral_mask(cube_coeffs.shape, radius=order)
    
    # Flatten ONLY the valid spherical coefficients
    coeffs_flat = cube_coeffs[mask]

    return coeffs_flat.astype(np.float32)

def mask_from_features(coeffs_flat: np.ndarray, shape, order, norm="ortho"):
    # 1. Recreate the Mask
    # We must generate the exact same mask used during encoding
    # The container is the small corner cube of size (order, order, order)
    cube_shape = (order, order, order)
    mask = get_spectral_mask(cube_shape, radius=order)
    
    # 2. Rebuild the Cube
    # Validate that coefficient counts match
    expected_size = np.sum(mask)
    if coeffs_flat.size != expected_size:
        raise ValueError(f"Coefficient mismatch! Expected {expected_size} coeffs for order {order}, got {coeffs_flat.size}.")

    cube_coeffs = np.zeros(cube_shape, dtype=coeffs_flat.dtype)
    cube_coeffs[mask] = coeffs_flat
    
    # 3. Pad to Full Volume
    full_coeffs = np.zeros(shape, dtype=coeffs_flat.dtype)
    full_coeffs[:order, :order, :order] = cube_coeffs

    # 4. Inverse DCT
    sdf_recon = idctn(full_coeffs, norm=norm, workers=-1)
    
    return sdf_recon < 0


def smooth_mask(mask:np.ndarray, order=16, norm="ortho"):
    coeffs = features_from_mask(mask, order, norm)
    mask_recon = mask_from_features(coeffs, mask.shape, order, norm)
    return mask_recon