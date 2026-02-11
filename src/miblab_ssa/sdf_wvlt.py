import pywt
import numpy as np
from scipy.ndimage import zoom, distance_transform_edt


# def spectral_vector(wavelet='db4', min_level=1, order=9):
#     """
#     Returns a list of shapes and keys to allow manual reconstruction 
#     of a truncated feature vector.
#     """
#     target_res = 2**order
#     levels = list(range(min_level, order-2))

#     dummy_input = np.zeros((target_res, target_res, target_res), dtype=np.float32)
#     max_level = max(levels)
#     coeffs = pywt.wavedecn(dummy_input, wavelet=wavelet, level=max_level)
    
#     # We store metadata: (level_index, is_approximation, shape_info)
#     meta = []
    
#     # Level 0 (Approximation)
#     meta.append(('approx', coeffs[0].shape))
    
#     # Detail levels
#     for i in range(1, len(coeffs)):
#         current_level = max_level - i + 1
#         # Store shapes for all 7 components (aaa, aad, etc.)
#         detail_shapes = {k: v.shape for k, v in coeffs[i].items()}
#         meta.append((current_level, detail_shapes))
            
#     return meta

# import pywt
# import numpy as np

def spectral_vector(wavelet='db4', min_level=1, order=7):
    """
    Calculates metadata shapes mathematically without computing the transform.
    """
    target_res = 2**order
    levels = list(range(min_level, order-2))
    max_level = max(levels)
    
    # 3D sub-band keys
    keys = ['aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']
    
    meta = []
    
    # Calculate shapes level by level
    current_shape = (target_res, target_res, target_res)
    all_shapes = []
    
    # PyWavelets calculates lengths based on: floor((data_len + filter_len - 1) / 2)
    for _ in range(max_level):
        # Calculate the length of the coefficients for this level
        new_dim = pywt.dwt_coeff_len(data_len=current_shape[0], 
                                    filter_len=pywt.Wavelet(wavelet).dec_len, 
                                    mode='symmetric')
        current_shape = (new_dim, new_dim, new_dim)
        all_shapes.append(current_shape)

    # all_shapes[0] is L1, all_shapes[-1] is max_level.
    # We need to reverse this to match the 'coeffs' list structure: [Approx, L_max, ..., L1]
    all_shapes.reverse()
    
    # 1. Level 0 (Approximation) - This is the shape of the final decomposition
    meta.append(('approx', all_shapes[0]))
    
    # 2. Detail Levels
    for i, shape in enumerate(all_shapes):
        current_level = max_level - i
        # The detail dictionaries at level i have 7 components (excluding 'aaa')
        detail_shapes = {k: shape for k in keys if k != 'aaa'}
        meta.append((current_level, detail_shapes))
            
    return meta

def features_from_mask(mask, wavelet='db4', min_level=1, order=9, return_meta=False):
    """
    Returns a truly truncated feature vector containing only selected levels.
    """
    target_res = 2**order
    levels = list(range(min_level, order-2))
    zoom_factor = target_res / mask.shape[0]

    # mask_rescaled = zoom(mask.astype(np.float32), zoom_factor, order=0)
    # sdf = mask_rescaled
    
    mask_rescaled = zoom(mask.astype(np.float32), zoom_factor, order=0) > 0.5
    sdf = (5.0 * np.tanh(distance_transform_edt(~mask_rescaled) - 
                         distance_transform_edt(mask_rescaled)) / 5.0).astype(np.float32)

    max_level = max(levels)
    coeffs = pywt.wavedecn(sdf, wavelet=wavelet, level=max_level)
    
    # Build the truncated vector manually
    feature_list = []

    # Build meta on the fly (cheap)
    meta = []
    
    # Always include Approximation (Level 0)
    feature_list.append(coeffs[0].flatten())
    meta.append(('approx', coeffs[0].shape))
    
    # Include all details in meta
    for i in range(1, len(coeffs)):
        current_level = max_level - i + 1
        # We ALWAYS store the shape in meta, even if we skip the data
        detail_shapes = {k: v.shape for k, v in coeffs[i].items()}
        meta.append((current_level, detail_shapes))

        if current_level in levels:
            for key in sorted(coeffs[i].keys()):
                feature_list.append(coeffs[i][key].flatten())

                # # count small values
                # f = np.sum(coeffs[i][key] < 1e-6 * np.max(coeffs[i][key])) / np.size(coeffs[i][key])
                # print(f"{i}, {key}: {100*f} %")

    feature_list = np.concatenate(feature_list).astype(np.float32)      
    if return_meta:
        return feature_list, meta
    else:
        return feature_list


def mask_from_features(coeffs_flat, original_shape, wavelet='db4', min_level=1, order=9, meta=None):
    """
    Reconstructs using the metadata to 'un-flatten' the truncated vector.
    """
    if meta is None:
        meta = spectral_vector(wavelet, min_level, order)
    levels = list(range(min_level, order-2))
    max_level = max(levels)
    reconstructed_coeffs = [None] * (max_level + 1)
    
    idx = 0
    # 1. Extract Approximation
    approx_shape = meta[0][1]
    approx_size = np.prod(approx_shape)
    reconstructed_coeffs[0] = coeffs_flat[idx:idx+approx_size].reshape(approx_shape)
    idx += approx_size
    
    # 2. Extract Details
    meta_idx = 1
    for i in range(1, max_level + 1):
        current_level = max_level - i + 1
        level_meta = meta[meta_idx][1]
        level_dict = {}

        for key in sorted(level_meta.keys()):
            shape = level_meta[key]
            if current_level in levels:
                size = np.prod(shape)
                level_dict[key] = coeffs_flat[idx:idx+size].reshape(shape)
                idx += size
            else:
                # LEVEL SKIPPED: Create zeros based on the metadata shape
                level_dict[key] = np.zeros(shape, dtype=np.float32)
        reconstructed_coeffs[i] = level_dict
        meta_idx += 1

    sdf_recon = pywt.waverecn(reconstructed_coeffs, wavelet=wavelet)
    zoom_factor = original_shape[0] / sdf_recon.shape[0]
    sdf_original = zoom(sdf_recon, zoom_factor, order=1)

    # return sdf_original > 0.5
    
    return sdf_original < 0


def smooth_mask(mask:np.ndarray, wavelet='db4', min_level=1, order=9):
    coeffs, meta = features_from_mask(mask, wavelet, min_level, order, return_meta=True)
    mask_recon = mask_from_features(coeffs, mask.shape, wavelet, min_level, order, meta)
    print(f"{coeffs.size} features at {coeffs.size * 4 / 1024 / 1024} MB")
    return mask_recon
