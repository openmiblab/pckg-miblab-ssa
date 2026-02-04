import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.fftpack import dctn, idctn


def features_from_mask(mask:np.ndarray, order=16, norm="ortho"):

    # Compute sdf
    mask = mask.astype(bool)
    dist_out = distance_transform_edt(~mask)
    dist_in  = distance_transform_edt(mask)
    sdf = dist_out - dist_in 

    # Compute coefficients, truncate and flatten
    coeffs = dctn(sdf, norm=norm)
    coeffs = coeffs[:order, :order, :order]
    coeffs = coeffs.flatten()

    return coeffs

def mask_from_features(coeffs_trunc:np.ndarray, shape, norm="ortho"):

    # Rebuild coefficients at required shape
    order = int(np.cbrt(coeffs_trunc.size))
    coeffs_trunc = coeffs_trunc.reshape(3 * [order])
    coeffs = np.zeros(shape, dtype=coeffs_trunc.dtype)
    coeffs[:order, :order, :order] = coeffs_trunc

    # Compute mask from sdf
    sdf = idctn(coeffs, norm=norm)
    mask = sdf < 0

    return mask

def smooth_mask(mask:np.ndarray, order=16, norm="ortho"):
    coeffs = features_from_mask(mask, order, norm)
    mask_recon = mask_from_features(coeffs, mask.shape, norm)
    return mask_recon

def features_from_dataset(masks, order=16, norm="ortho"):
    features = [features_from_mask(mask, order, norm) for mask in masks]
    return np.array(features)





