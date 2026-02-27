import numpy as np
from scipy.ndimage import distance_transform_edt, label
from scipy.fft import dctn, idctn  


def descriptors(coeffs, freq_vecs, low_freq_threshold=4.0):
    """
    Extracts physically meaningful descriptors from the DCT spectrum.
    
    Args:
        coeffs (np.ndarray): Flattened spectral coefficients.
        freq_vecs (np.ndarray): (3, N) array of (i, j, k) frequencies.
        low_freq_threshold (float): Boundary for 'low' frequency compaction.
        
    Returns:
        dict: Physical shape descriptors.

    Notes:
        Wavelength (in pixels) is 2L/k, with the matrix size and k the frequency
        For a matrix of 256 pixels and k=4 the wavelength cutoff is 128 pixels
    """
    # Precompute common terms
    powers = coeffs**2
    magnitudes = np.abs(coeffs)
    total_power = np.sum(powers)
    total_mag = np.sum(magnitudes)
    
    # Avoid division by zero for empty masks
    if total_power == 0 or total_mag == 0:
        return {k: 0.0 for k in ['total_power', 'compaction_ratio', 'spectral_entropy', 
                                 'spectral_centroid', 'anisotropy_index']}

    # 1. Power & Compaction
    radii = np.sqrt(np.sum(freq_vecs**2, axis=0))
    low_freq_power = np.sum(powers[radii <= low_freq_threshold])
    compaction_ratio = low_freq_power / total_power
    
    # 2. Spectral Entropy (Uncertainty/Complexity)
    psd_norm = powers / total_power
    # Use only non-zero values for Shannon Entropy
    psd_safe = psd_norm[psd_norm > 0]
    spectral_entropy = -np.sum(psd_safe * np.log2(psd_safe))
    
    # 3. Spectral Centroid (Weighted Mean Frequency)
    spectral_centroid = np.sum(radii * magnitudes) / total_mag
    
    # 4. Anisotropy (Directional Bias)
    # Weighted mean frequency per axis
    mean_freqs = np.sum(freq_vecs * magnitudes, axis=1) / total_mag
    # Standard deviation of axis means normalized by the global mean
    anisotropy_index = np.std(mean_freqs) / (np.mean(mean_freqs) + 1e-6)

    return {
        'total_power': float(total_power),
        'compaction_ratio': float(compaction_ratio),
        'spectral_entropy': float(spectral_entropy),
        'spectral_centroid': float(spectral_centroid),
        'anisotropy_index': float(anisotropy_index),
        'mean_freq_i': float(mean_freqs[0]),
        'mean_freq_j': float(mean_freqs[1]),
        'mean_freq_k': float(mean_freqs[2])
    }


def spectral_vectors(shape, order):
    """
    Returns the full frequency components (i, j, k) for the 
    flattened coefficients.
    
    Returns:
        np.ndarray: A (3, N) array where:
            row 0 = frequencies in I (depth/x)
            row 1 = frequencies in J (height/y)
            row 2 = frequencies in K (width/z)
    """
    # 1. Create the grid of indices
    ranges = [np.arange(n) for n in shape]
    I, J, K = np.meshgrid(*ranges, indexing='ij')
    
    # 2. Create the mask (matching features_from_mask logic)
    mask = (I**2 + J**2 + K**2) <= order**2
    
    # 3. Extract the components for the masked positions
    freq_i = I[mask]
    freq_j = J[mask]
    freq_k = K[mask]
    
    # 4. Stack into a (3, N) array
    # Using vstack results in a (3, N) shape
    return np.vstack((freq_i, freq_j, freq_k)).astype(np.float32)


def get_spectral_mask(shape, order):
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
    mask = (I**2 + J**2 + K**2) <= order**2
    
    return mask

def features_from_mask(mask: np.ndarray, order=16, norm="ortho", saturation_threshold=5.0):
    """Memory usage is about 10 x the size of mask"""
    
    # 1. Compute SDF & Saturate
    mask = mask.astype(bool)
    # Using float32 saves memory/speed for DCT without losing precision relevant for shape
    sdf = (distance_transform_edt(~mask) - distance_transform_edt(mask)).astype(np.float32)
    sdf_saturated = saturation_threshold * np.tanh(sdf / saturation_threshold)

    # 2. Compute coefficients
    full_coeffs = dctn(sdf_saturated, norm=norm, workers=1)
    
    # 3. Cubic Truncation first (Efficiency)
    # We slice out the corner cube first to avoid generating a mask for the massive 300^3 volume
    cube_coeffs = full_coeffs[:order, :order, :order]
    
    # 4. Spherical Masking
    # Now we generate the mask specifically for this cube shape
    mask = get_spectral_mask(cube_coeffs.shape, order)
    
    # Flatten ONLY the valid spherical coefficients
    coeffs_flat = cube_coeffs[mask]

    return coeffs_flat.astype(np.float32)

def mask_from_features(coeffs_flat: np.ndarray, shape, order, norm="ortho"):
    """Memory usage about 10 times mask size"""
    
    # 1. Recreate the Mask
    # We must generate the exact same mask used during encoding
    # The container is the small corner cube of size (order, order, order)
    cube_shape = (order, order, order)
    mask = get_spectral_mask(cube_shape, order)
    
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
    sdf_recon = idctn(full_coeffs, norm=norm, workers=1)
    sdf_recon = sdf_recon < 0

    # POST-PROCESS: Keep only the largest component (The Kidney)
    if np.any(sdf_recon):
        labeled, num_features = label(sdf_recon)
        if num_features > 1:
            # Sort components by size
            sizes = np.bincount(labeled.ravel())
            largest_label = sizes[1:].argmax() + 1
            sdf_recon = (labeled == largest_label)
    
    return sdf_recon


def smooth_mask(mask:np.ndarray, order=16, norm="ortho"):
    coeffs = features_from_mask(mask, order, norm)
    mask_recon = mask_from_features(coeffs, mask.shape, order, norm)
    return mask_recon