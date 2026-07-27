import os
from typing import Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import logging
from tqdm import tqdm
import zarr
import dask.array as da
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dask_ml.decomposition import PCA as daskPCA


def _elbow_index(x: np.ndarray, y: np.ndarray) -> int:
    """
    Finds the elbow/knee of a curve as the point of maximum perpendicular
    distance from the chord connecting its first and last point (the
    standard "distance-to-chord" knee-detection method).
    """
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    start = np.array([x_norm[0], y_norm[0]])
    end = np.array([x_norm[-1], y_norm[-1]])
    line_vec = end - start
    line_vec /= np.linalg.norm(line_vec)

    points = np.stack([x_norm, y_norm], axis=1) - start
    proj_len = points @ line_vec
    proj = np.outer(proj_len, line_vec)
    perp_dist = np.linalg.norm(points - proj, axis=1)

    return int(np.argmax(perp_dist))


def pca_cv_from_features_joao(
    features_zarr_path: str = None,
    output_plot_path: str = None,
    max_components: int = None,
    n_folds: int = 5,
    step: int = 1,
    random_state: int = 42,
):
    """
    Cross-validated Scikit-Learn PCA: fits PCA on each training fold and
    measures held-out reconstruction error across a range of component
    counts, to help verify linear PCA is behaving sensibly and to pick the
    number of components that best generalizes (rather than overfitting to
    the same data PCA was fit on). Loads the entire feature matrix into RAM
    (see pca_from_features).
    """
    logging.info(f"PCA CV: Loading data from {os.path.basename(features_zarr_path)} into RAM...")

    # 1. Load Data from Zarr directly into NumPy
    feat_root = zarr.open(features_zarr_path, mode='r')
    data = feat_root['features'][:]

    n_samples, n_features = data.shape
    # Each fold trains on fewer than n_samples rows (KFold gives some folds one
    # extra test sample), so n_components is capped by the smallest training
    # fold's rank, not the full dataset's.
    max_test_size = -(-n_samples // n_folds)  # ceil division
    min_train_size = n_samples - max_test_size
    max_k = min(max_components or (min_train_size - 1), min_train_size - 1, n_features)
    components_range = np.arange(1, max_k + 1, step)

    logging.info(f"PCA CV: Matrix shape {n_samples}x{n_features}. Running {n_folds}-fold CV over k=1..{max_k}...")

    # 2. Cross-Validate: fit once per fold at max_k, score reconstruction at each k
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_errors = np.full((n_folds, len(components_range)), np.nan)

    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        logging.info(f"PCA CV: Fold {fold + 1}/{n_folds}")

        pca = PCA(n_components=max_k, svd_solver='auto', random_state=random_state)
        pca.fit(data[train_idx])

        val_centered = data[val_idx] - pca.mean_

        for i, k in enumerate(components_range):
            basis = pca.components_[:k]
            scores = val_centered @ basis.T
            recon = scores @ basis
            fold_errors[fold, i] = np.mean((val_centered - recon) ** 2)

    mean_error = fold_errors.mean(axis=0)
    std_error = fold_errors.std(axis=0)

    # Held-out reconstruction MSE is (almost) always monotonically decreasing in k,
    # so argmin trivially lands on max_k and is not a useful "how many components
    # do I need" answer. Use the elbow of the curve instead.
    min_idx = int(np.argmin(mean_error))
    min_k = int(components_range[min_idx])
    elbow_idx = _elbow_index(components_range.astype(float), mean_error)
    optimal_k = int(components_range[elbow_idx])

    logging.info(
        f"PCA CV: Elbow (recommended) number of components = {optimal_k} "
        f"(mean val MSE = {mean_error[elbow_idx]:.6f}); "
        f"lowest mean val MSE reached at k={min_k} (MSE = {mean_error[min_idx]:.6f}, "
        f"the search ceiling)"
    )

    # 3. Plot: full-range view (with per-fold curves) + zoomed-in view around the elbow
    zoom_k = min(max_k, max(optimal_k * 3, 10))
    zoom_mask = components_range <= zoom_k

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = cm.tab10(np.linspace(0, 1, n_folds))

    for ax, mask, title in [
        (axes[0], slice(None), f"Full range (k=1..{max_k})"),
        (axes[1], zoom_mask, f"Zoomed near elbow (k=1..{zoom_k})"),
    ]:
        for fold_idx in range(n_folds):
            ax.plot(
                components_range[mask], fold_errors[fold_idx][mask],
                color=colors[fold_idx], alpha=0.35, linewidth=1, label=f"Fold {fold_idx + 1}"
            )
        ax.plot(
            components_range[mask], mean_error[mask],
            color='black', linewidth=2, label='Mean across folds'
        )
        ax.fill_between(
            components_range[mask],
            (mean_error - std_error)[mask],
            (mean_error + std_error)[mask],
            color='black', alpha=0.12, label='+/- 1 std'
        )
        ax.axvline(optimal_k, color='red', linestyle='--', linewidth=1.5, label=f"Elbow (k={optimal_k})")
        ax.set_xlabel("Number of components")
        ax.set_ylabel("Validation reconstruction MSE")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # Single shared legend (fold lines repeat across panels)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=min(len(by_label), 7), bbox_to_anchor=(0.5, 1.05))

    fig.suptitle(f"{n_folds}-Fold Cross-Validated PCA Reconstruction Error", y=1.12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_plot_path) or '.', exist_ok=True)
    plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"PCA CV: Plot saved to {output_plot_path}")

    return optimal_k, components_range, mean_error, std_error


def pca_from_features(
    features_zarr_path: str = None, 
    output_zarr_path: str = None, 
    n_components: int = None,
):
    """
    Standard Scikit-Learn PCA: Loads the entire feature matrix into RAM.
    Best for cases where N_samples * N_features * 4 bytes < Available RAM.
    """
    logging.info(f"PCA: Loading data from {os.path.basename(features_zarr_path)} into RAM...")

    # 1. Load Data from Zarr directly into NumPy
    feat_root = zarr.open(features_zarr_path, mode='r')
    data = feat_root['features'][:]  # The [:] loads the whole array into memory
    labels = feat_root['labels'][:]
    
    n_samples, n_features = data.shape
    logging.info(f"PCA: Matrix shape {n_samples}x{n_features}. Fitting Sklearn PCA...")

    # 2. Fit Scikit-Learn PCA
    # 'auto' or 'full' works best here since we are in-memory
    pca = PCA(n_components=n_components, svd_solver='auto')
    pca.fit(data)

    # 3. Prepare Output Zarr
    if not output_zarr_path.endswith('.zarr'):
        output_zarr_path += '.zarr'
    
    root = zarr.open_group(output_zarr_path, mode='w')
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)

    # 4. Save Results
    logging.info("PCA: Saving model attributes to Zarr...")

    # Save the basis (components)
    root.create_dataset('components', 
                        data=pca.components_.astype(np.float32), 
                        chunks=(1, n_features),
                        compressor=compressor)

    # Save metadata and model parameters
    root.create_dataset('mean', 
                        data=pca.mean_.astype(np.float32), 
                        compressor=compressor)
    
    root.create_dataset('variance', 
                        data=pca.explained_variance_.astype(np.float32))
    
    root.create_dataset('variance_ratio', 
                        data=pca.explained_variance_ratio_.astype(np.float32))
    
    root.create_dataset('labels', data=labels)

    # 5. Transfer Attributes (Essential for reconstruction)
    root.attrs['original_shape'] = feat_root.attrs.get('original_shape', None)
    root.attrs['kwargs'] = feat_root.attrs.get('kwargs')

    logging.info("PCA: Finished.")


def direct_pca_from_features(
    features_zarr_path: str, 
    output_zarr_path: str, 
    n_components=50,
    chunk_size=10000  # Number of features per chunk
):
    """
    Direct SVD implementation to bypass dask_ml shuffle overhead.
    Optimized for Wide Data (N_samples << N_features).
    """
    logging.info(f"PCA: Connecting to {os.path.basename(features_zarr_path)}...")

    # 1. Open Zarr safely
    feat_root = zarr.open(features_zarr_path, mode='r')
    features_ds = feat_root['features']
    labels = feat_root['labels'][:]
    
    # Rechunking: All samples (-1) for a slice of features.
    # This is the "Tall and Skinny" slice that dask.linalg.svd prefers.
    data = da.from_zarr(features_ds).rechunk({0: -1, 1: chunk_size})
    n_samples, n_features = data.shape
    k = n_components or 50

    # 2. Manual Centering & Variance calculation
    # We compute the mean and total variance first to calculate ratios later
    logging.info("PCA: Computing column means and total variance...")
    mean_dask = data.mean(axis=0)
    
    # We need the sum of individual feature variances for the ratio denominator
    # We use a map_blocks approach or standard var to keep it lazy
    total_var_dask = data.var(axis=0).sum()
    
    # Trigger compute for these small scalars/vectors
    mean_np, total_var = da.compute(mean_dask, total_var_dask)
    
    # 3. Running Compressed SVD
    logging.info(f"PCA: Running svd_compressed (k={k}) on {n_samples}x{n_features}...")
    # data_centered is a lazy operation
    data_centered = data - mean_np
    
    # u: Scores, s: Singular values, v: Components (Principal Axes)
    u, s, v = da.linalg.svd_compressed(data_centered, k=k, n_power_iter=3)

    # 4. Prepare Output Zarr
    if not output_zarr_path.endswith('.zarr'):
        output_zarr_path += '.zarr'
    
    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)

    # 5. Compute and Save Results
    # We compute s (singular values) first to derive variance metrics
    s_np = s.compute()
    
    # Eigenvalues (Explained Variance) = s^2 / (n-1)
    explained_variance = (s_np**2) / (n_samples - 1)
    explained_variance_ratio = explained_variance / total_var

    logging.info("PCA: Saving components (streaming dask -> zarr)...")
    # Stream the massive V matrix (components) directly to disk
    v.astype(np.float32).to_zarr(
        url=root.store, 
        component='components', 
        compressor=compressor,
        overwrite=True
    )

    logging.info("PCA: Saving metadata and variance metrics...")
    root.create_dataset('mean', data=mean_np.astype(np.float32), compressor=compressor)
    root.create_dataset('variance', data=explained_variance.astype(np.float32))
    root.create_dataset('variance_ratio', data=explained_variance_ratio.astype(np.float32))
    root.create_dataset('labels', data=labels)

    # 6. Transfer Attributes
    root.attrs['original_shape'] = feat_root.attrs.get('original_shape', None)
    root.attrs['kwargs'] = feat_root.attrs.get('kwargs')
    root.attrs['n_samples'] = n_samples
    root.attrs['n_features'] = n_features

    logging.info(f"PCA: Finished successfully.")


def dask_pca_from_features(
    features_zarr_path: str, 
    output_zarr_path: str, 
    n_components=None,
    chunk_size=100  # Number of samples per chunk
):
    """
    Dask PCA: Streams (N, F) features from Zarr. 
    Optimized for N_samples << N_features (Wide Data).

    Memory usage (bytes): max of 
    - n_samples x n_features x 4
    - chunksize x n_features x 8
    """
    logging.info(f"PCA: Connecting to {os.path.basename(features_zarr_path)}...")

    # 1. Open Zarr as a Dask Array (Lazy)
    feat_root = zarr.open(features_zarr_path, mode='r')
    features_ds = feat_root['features']
    labels = feat_root['labels'][:]
    
    # Rechunking: (chunk_size, all_features)
    # This ensures each worker gets a manageable batch of full wavelet vectors
    data = da.from_zarr(features_ds).rechunk({0: chunk_size, 1: -1})
    
    n_samples, n_features = data.shape

    # 2. Fit Dask PCA
    # svd_solver='full' is efficient when one dimension is significantly smaller
    logging.info(f"PCA: Fitting Dask PCA on {n_samples}x{n_features} matrix...")
    pca = daskPCA(n_components=n_components, svd_solver='full') 
    pca.fit(data)

    # 3. Prepare Output Zarr
    if not output_zarr_path.endswith('.zarr'):
        output_zarr_path += '.zarr'
    
    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)

    # 4. Save Results
    logging.info("PCA: Saving model attributes...")

    # A. Save the massive components matrix
    # If it's already a numpy array, we use create_dataset. 
    # If it's dask, we use to_zarr to save RAM.
    if hasattr(pca.components_, 'compute'):
        logging.info("Components are Dask-backed. Streaming to Zarr...")
        pca.components_.astype(np.float32).to_zarr(
            url=root.store, 
            component='components', 
            compressor=compressor,
            overwrite=True
        )
    else:
        logging.info("Components are NumPy-backed. Writing to Zarr...")
        root.create_dataset('components', 
                            data=pca.components_.astype(np.float32), 
                            chunks=(1, n_features),
                            compressor=compressor)
        
    # helper to handle both dask and numpy arrays safely
    # daskPCA can return either numpy arrays or dask arrays
    def safe_compute(arr):
        return arr.compute() if hasattr(arr, 'compute') else arr

    # B. Save smaller datasets safely
    root.create_dataset('mean', 
                        data=safe_compute(pca.mean_).astype(np.float32), 
                        compressor=compressor)
    
    root.create_dataset('variance', 
                        data=safe_compute(pca.explained_variance_).astype(np.float32))
    
    root.create_dataset('variance_ratio', 
                        data=safe_compute(pca.explained_variance_ratio_).astype(np.float32))
    
    root.create_dataset('labels', data=labels)

    # 5. Transfer Attributes
    root.attrs['original_shape'] = feat_root.attrs.get('original_shape', None)
    root.attrs['kwargs'] = feat_root.attrs.get('kwargs')

    logging.info(f"PCA: Finished.")




def scores_from_features(
    features_zarr_path: str = None, 
    pca_zarr_path: str = None, 
    scores_csv_path: str = None,
    normalized_scores_csv_path: str = None, 
):
    """
    Computes PCA scores in memory and saves results to CSV files.
    output_csv_base_path: e.g., 'results/pca_output' 
    (will create results/pca_output_scores.csv and results/pca_output_norm.csv)
    """
    logging.info(f"Loading data from {os.path.basename(features_zarr_path)} into RAM...")
    
    # 1. Load Input Data into RAM
    feat_root = zarr.open(features_zarr_path, mode='r')
    features = feat_root['features'][:] # (N, F)
    labels = feat_root['labels'][:]
    
    # 2. Load PCA Basis into RAM
    pca_root = zarr.open(pca_zarr_path, mode='r')
    mu = pca_root['mean'][:]               # (F,)
    eig_vecs = pca_root['components'][:]   # (K, F)
    var = pca_root['variance'][:]          # (K,)

    # 3. Compute Projections
    logging.info("Computing projections...")
    centered = features - mu
    scores = centered @ eig_vecs.T
    
    logging.info("Normalizing scores...")
    normalized_scores = scores / np.sqrt(var)

    # 4. Convert to Pandas for CSV Export
    # Create column names (PC_1, PC_2, ..., PC_K)
    n_components = scores.shape[1]
    col_names = [f"PC_{i+1}" for i in range(n_components)]

    # Create Raw Scores Dataframe
    df_scores = pd.DataFrame(scores, columns=col_names)
    df_scores.insert(0, 'label', labels) # Add label at start
    
    # Create Normalized Scores Dataframe
    df_norm = pd.DataFrame(normalized_scores, columns=col_names)
    df_norm.insert(0, 'label', labels)

    # 5. Save to CSV
    logging.info(f"Saving CSVs...")
    df_scores.to_csv(scores_csv_path, index=False)
    df_norm.to_csv(normalized_scores_csv_path, index=False)

    logging.info("Finished PCA projection and CSV export.")
    return scores, normalized_scores


def dask_scores_from_features(
    features_zarr_path, 
    pca_zarr_path, 
    output_zarr_path, 
):
    """
    Memory usage:

    n_features x (chunk_size + n_components) x 8
    """
    logging.info("Connecting to Zarr stores (Lazy)...")
    
    # 1. Open Zarr as Dask Arrays
    features = da.from_zarr(features_zarr_path, component='features')
    labels = da.from_zarr(features_zarr_path, component='labels')
    pca_root = zarr.open(pca_zarr_path, mode='r')
    
    # PCA attributes are small enough to compute/load once, 
    # EXCEPT for 'components' if they are 10M wide.
    # We keep components as a dask array to stream the dot product.
    mu = da.from_zarr(pca_zarr_path, component='mean')
    eig_vecs = da.from_zarr(pca_zarr_path, component='components') # (n_comp, n_features)
    var = da.from_zarr(pca_zarr_path, component='variance')
    var_ratio = pca_root['variance_ratio'][:] # Small enough for numpy

    # 2. Dask Math (Symbolic/Lazy)
    # This creates a task graph, it does NOT execute yet.
    logging.info("Building projection graph...")
    
    # Centering (Broadcasting mu over features)
    centered = features - mu
    
    # Projection: (N, F) @ (F, K) -> (N, K)
    # We use eig_vecs.T to align dimensions for the dot product
    scores = da.matmul(centered, eig_vecs.T)
    
    # Normalization by sqrt of variance
    normalized_scores = scores / da.sqrt(var)

    # 3. Prepare Output Zarr
    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)

    # 4. Execute and Save
    # We use .to_zarr() to trigger the computation chunk-by-chunk
    logging.info(f"Computing and streaming scores to {output_zarr_path}...")
    
    # Compute and save scores
    scores.astype(np.float32).to_zarr(
        url=root.store, 
        component='scores', 
        compressor=compressor,
        overwrite=True
    )
    
    # Compute and save normalized scores
    normalized_scores.astype(np.float32).to_zarr(
        url=root.store, 
        component='normalized_scores', 
        compressor=compressor,
        overwrite=True
    )

    # Save metadata (Small arrays)
    root.create_dataset('labels', data=labels.compute())
    root.create_dataset('variance', data=var.compute())
    root.create_dataset('variance_ratio', data=var_ratio)

    # Copy essential attributes for reconstruction
    feat_root = zarr.open(features_zarr_path, mode='r')
    root.attrs['original_shape'] = feat_root.attrs.get('original_shape')
    root.attrs['kwargs'] = feat_root.attrs.get('kwargs')
    
    logging.info("Finished PCA projection.")
    return True


def modes_from_pca(
    pca_zarr_path: str = None, 
    modes_zarr_path: str = None, 
    n_components: int = 8, 
    n_coeffs: int = 11, 
    max_coeff: float = 5  # Typically 2 or 3 standard deviations
):
    """
    Generates 3D shape modes by varying PCA coefficients and reconstructing masks.
    Output: 5D Zarr (Coeff_Steps, Mode_Index, Depth, Height, Width)

    Memory usage (1 mask + 1 feature vector) x overhead of mask_from_features
    """
    logging.info(f"Modes: Opening PCA store at {os.path.basename(pca_zarr_path)}...")
    
    # 1. Connect to PCA (Lazy)
    z_pca = zarr.open(pca_zarr_path, mode='r')
    
    # Check bounds of stored components
    n_features = z_pca['components'].shape[1]
    stored_k = z_pca['components'].shape[0]
    limit_k = min(n_components, stored_k)
    
    # Load Mean and Metadata and create coeffs
    avr = z_pca['mean'][:]
    sdev = np.sqrt(z_pca['variance'][:limit_k])
    coeffs_range = np.linspace(-max_coeff, max_coeff, n_coeffs)

    # 3. Setup 5D Output Store
    if not modes_zarr_path.endswith('.zarr'):
        modes_zarr_path += '.zarr'
        
    store = zarr.DirectoryStore(modes_zarr_path)
    root = zarr.group(store=store, overwrite=True)

    z_modes = root.create_dataset(
        'features',
        shape=(n_coeffs, limit_k, n_features) ,
        chunks=(1, 1, n_features),
        dtype=np.float32,
        compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2),
    )

    # 4. Generate Modes (The "Safe" Loop)
    logging.info(f"Modes: Building {n_coeffs * limit_k} feature vectors...")

    for i in tqdm(range(limit_k), desc="Processing Modes"):
    
        current_comp = z_pca['components'][i, :]
        for j in range(n_coeffs):
            feat_vec = avr + (coeffs_range[j] * sdev[i] * current_comp)
            z_modes[j, i, ...] = feat_vec

    # 5. Save attributes for the visualization tool
    root.attrs['coeffs'] = coeffs_range.tolist()
    root.attrs['original_shape'] = z_pca.attrs.get('original_shape')
    root.attrs['kwargs'] = z_pca.attrs.get('kwargs')
    
    logging.info(f"Modes: Successfully saved to {modes_zarr_path}")



def features_from_scores(
    pca_zarr_path: str = None, 
    scores_csv_path: str = None,
    output_zarr_path: str = None,
    target_labels: list = None,
    n_components: int = None,
):
    """
    Reconstructs features (e.g. spectral representations) from PCA scores stored in a CSV.
    Reconstruction formula: features = mean + (scores @ components)
    """
    # 1. Load PCA Basis
    z_pca = zarr.open(pca_zarr_path, mode='r')
    avr = z_pca['mean'][:] 
    all_components = z_pca['components'][:] # (Total_K, F)
    n_features = all_components.shape[1]
    
    # 2. Load Scores from CSV
    logging.info(f"Reading scores from {scores_csv_path}...")
    df = pd.read_csv(scores_csv_path)
    
    # 3. Filter by labels if requested
    if target_labels is not None:
        # Ensure labels are compared as strings to be safe
        df['label_str'] = df['label'].astype(str)
        target_labels_str = [str(l) for l in target_labels]
        df = df[df['label_str'].isin(target_labels_str)]
        if df.empty:
            logging.error("No matching labels found in CSV.")
            return
    
    found_labels = df['label'].values
    
    # 4. Extract score matrix (exclude the label column)
    # We look for columns starting with 'PC_' or 'Feat_'
    score_cols = [c for c in df.columns if c.startswith(('PC_', 'Feat_'))]
    
    if n_components is not None:
        score_cols = score_cols[:n_components]
        eig_vecs = all_components[:n_components]
    else:
        # Match components to available columns in CSV
        n_comp_in_csv = len(score_cols)
        eig_vecs = all_components[:n_comp_in_csv]
        
    scores_matrix = df[score_cols].values # (N, K)

    # 5. Perform Reconstruction
    logging.info(f"Reconstructing {len(found_labels)} samples...")
    # Math: (N, K) @ (K, F) + (F,) -> (N, F)
    reconstructed_features = (scores_matrix @ eig_vecs) + avr

    # 6. Save to Zarr
    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    
    root.create_dataset(
        'features',
        data=reconstructed_features.astype(np.float32),
        chunks=(1, n_features),
        compressor=zarr.Blosc(cname='zstd', clevel=3)
    )
    
    # Save labels and copy attributes from PCA root if they exist
    root.array('labels', data=np.array(found_labels, dtype='U'))
    
    # Metadata recovery
    root.attrs['original_shape'] = z_pca.attrs.get('original_shape')
    root.attrs['kwargs'] = z_pca.attrs.get('kwargs')

    logging.info(f"Reconstruction finished. Output saved to {output_zarr_path}")
    return reconstructed_features

# def features_from_scores(
#     pca_zarr_path, 
#     scores_zarr_path,
#     output_zarr_path,
#     target_labels=None,
#     n_components=None,
# ):
#     z_pca = zarr.open(pca_zarr_path, mode='r')
#     z_scores = zarr.open(scores_zarr_path, mode='r')
    
#     # Use .get() or check keys to prevent KeyErrors
#     avr = z_pca['mean'][:] 
#     n_features = z_pca['components'].shape[1]
    
#     # Optimization: Pre-load components once
#     if n_components is None:
#         eig_vecs = z_pca['components'][:]
#     else:
#         eig_vecs = z_pca['components'][:n_components]

#     # Label indexing
#     all_labels = z_scores['labels'][:].astype(str)
#     label_to_idx = {l: i for i, l in enumerate(all_labels)}
    
#     if target_labels is not None:
#         valid_indices = [label_to_idx[str(l)] for l in target_labels if str(l) in label_to_idx]
#         found_labels = [str(l) for l in target_labels if str(l) in label_to_idx]
#     else:
#         valid_indices = list(range(len(all_labels)))
#         found_labels = all_labels

#     store = zarr.DirectoryStore(output_zarr_path)
#     root = zarr.group(store=store, overwrite=True)
    
#     z_recons = root.create_dataset(
#         'features',
#         shape=(len(valid_indices), n_features),
#         chunks=(1, n_features),
#         dtype='float32',
#         compressor=zarr.Blosc(cname='zstd', clevel=3)
#     )
#     # Map to fixed-length string or object to avoid Zarr string errors
#     root.array('labels', data=found_labels)

#     for i, idx in enumerate(valid_indices):
        
#         if n_components is None:
#             score = z_scores['scores'][idx]
#         else:
#             score = z_scores['scores'][idx,:n_components]
            
#         # Standard reconstruction: mean + (scores @ components)
#         z_recons[i] = (avr + np.dot(score, eig_vecs)).astype(np.float32)

#     # Copy essential attributes for reconstruction
#     root.attrs['original_shape'] = z_scores.attrs.get('original_shape')
#     root.attrs['kwargs'] = z_scores.attrs.get('kwargs')

#     logging.info("Reconstruction finished.")


def cumulative_features_from_scores(
    pca_zarr_path: str = None, 
    scores_csv_path: str = None,
    gt_features_zarr_path: str = None, 
    output_zarr_path: str = None,
    target_labels: list = None,
    step_size: int = 1,
    max_components: int = 10
):
    # 1. Open Stores
    z_pca = zarr.open(pca_zarr_path, mode='r')
    feat_gt = zarr.open(gt_features_zarr_path, mode='r')
    
    avr = z_pca['mean'][:].astype(np.float32) 
    n_features = z_pca['components'].shape[1]
    
    # Load and Filter CSV
    df = pd.read_csv(scores_csv_path)
    label_col = df.columns[0]
    
    # Explicitly grab only PC columns to avoid metadata interference
    pc_cols = [c for c in df.columns if 'PC' in c][:max_components]
    max_components = len(pc_cols) 

    if target_labels is not None:
        target_str = [str(l) for l in target_labels]
        df = df[df[label_col].astype(str).isin(target_str)]

    # 2. Sync Labels with Ground Truth Zarr
    labels_orig = {str(l): i for i, l in enumerate(feat_gt['labels'][:])}
    
    # Only keep subjects that exist in BOTH the CSV and the GT Zarr
    df = df[df[label_col].astype(str).isin(labels_orig.keys())]

    if df.empty:
        logging.error("No matching labels found across CSV and GT Zarr.")
        return False
    
    found_labels = df[label_col].astype(str).tolist()
    scores_matrix = df[pc_cols].values.astype('float32') # Use explicit PC columns

    # Calculate indices
    n_samples = len(found_labels)
    save_indices = np.arange(step_size - 1, max_components, step_size)
    n_steps = len(save_indices) + 2
    step_names = ["Mean"] + (save_indices + 1).tolist() + ["GT"]

    # 3. Setup Output
    root = zarr.open(output_zarr_path, mode='w')
    root.attrs['saved_steps'] = step_names
    
    # Fix: Safely copy attributes from PCA store
    for attr in ['original_shape', 'kwargs']:
        if attr in z_pca.attrs:
            root.attrs[attr] = z_pca.attrs[attr]

    root.create_dataset('labels', data=np.array(found_labels, dtype='U'), overwrite=True)

    z_recons = root.create_dataset(
        'features',
        shape=(n_samples, n_steps, n_features),
        chunks=(1, 1, n_features),
        dtype='float32',
        compressor=zarr.Blosc(cname='zstd', clevel=3),
        overwrite=True
    )

    # 4. Loop
    logging.info(f"Cumulative Reconstruction: {n_samples} samples, {n_steps} steps.")
    eig_vecs = z_pca['components'][:max_components].astype(np.float32)

    for i, label in enumerate(tqdm(found_labels, desc="Cumulative Steps")):
        # Step 0: Mean
        z_recons[i, 0, :] = avr

        # Step -1: GT (Safe now because of filtering at start)
        gt_idx = labels_orig[label]
        z_recons[i, -1, :] = feat_gt['features'][gt_idx].astype(np.float32)

        # Cumulative Summation
        scores = scores_matrix[i, :]
        current_reconstruction = avr.copy()
        step_counter = 1 
        
        for k in range(max_components):
            current_reconstruction += (scores[k] * eig_vecs[k])
            
            if k in save_indices:
                z_recons[i, step_counter, :] = current_reconstruction
                step_counter += 1

    return True


def pca_performance(
    pca_zarr_path: str = None, 
    scores_csv_path: str = None, 
    gt_features_zarr_path: str = None, 
    marginal_mse_csv_path: str = None,
    cumulative_mse_csv_path: str = None,
    n_components: int = 50,
):
    # 1. Load Data
    z_pca = zarr.open(pca_zarr_path, mode='r')
    z_orig = zarr.open(gt_features_zarr_path, mode='r')
    df_scores = pd.read_csv(scores_csv_path)
    
    # Setup Columns and Index
    pc_cols = [c for c in df_scores.columns if c.startswith('PC')][:n_components]
    labels_scores = df_scores['label'].astype(str).tolist()
    labels_orig = [str(l) for l in z_orig['labels'][:]]
    orig_label_to_idx = {l: i for i, l in enumerate(labels_orig)}

    # 2. Initialize DataFrames with NaNs
    # This ensures the CSV has the full structure from the start
    df_marginal = pd.DataFrame(np.nan, index=labels_scores, columns=pc_cols)
    df_cumulative = pd.DataFrame(np.nan, index=labels_scores, columns=pc_cols)
    
    # 3. Load Basis Arrays
    avr = z_pca['mean'][:]
    eig_vecs = z_pca['components'][:n_components, :]
    
    logging.info(f"Starting evaluation. Saving to CSV every subject.")

    # 4. Processing Loop
    for i, label in enumerate(tqdm(labels_scores)):
        if label not in orig_label_to_idx:
            continue
        
        orig_idx = orig_label_to_idx[label]
        orig_features = z_orig['features'][orig_idx]
        # orig_features_norm = np.linalg.norm(orig_features)
        subject_scores = df_scores.loc[i, pc_cols].values
        
        current_cumulative_vec = avr.copy()
        
        # Inner loop for components
        for k in range(len(pc_cols)):
            score_k = subject_scores[k]
            vec_k = eig_vecs[k, :]
            
            # Marginal calculation
            marginal_feat_vec = avr + (score_k * vec_k)
            err_m = np.linalg.norm(marginal_feat_vec - orig_features) #/ orig_features_norm
            df_marginal.iloc[i, k] = err_m
            
            # Cumulative calculation
            current_cumulative_vec += (score_k * vec_k)
            err_c = np.linalg.norm(current_cumulative_vec - orig_features) #/ orig_features_norm
            df_cumulative.iloc[i, k] = err_c
            
        # --- Save in Loop ---
        # We save the whole dataframe so the file stays structured and readable
        if i % 5 == 0 or i == len(labels_scores) - 1: # Save every 5 subjects to reduce I/O lag
            df_marginal.to_csv(marginal_mse_csv_path)
            df_cumulative.to_csv(cumulative_mse_csv_path)

    return df_marginal, df_cumulative
