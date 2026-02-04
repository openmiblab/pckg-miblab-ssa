from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np


def run_pca(features, n_components=10):
    """
    Run PCA on feature matrix.
    Returns reduced features and fitted PCA object.
    """
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(features)
    return reduced, pca

def save_pca_model(feature_file, pca_file, n_components=None):
    """
    Saves the essential math attributes of a fitted PCA object.
    """
    features = np.load(feature_file)['features']
    pca = PCA(n_components=n_components)
    pca.fit(features)
    np.savez(
        pca_file,
        mean=pca.mean_,
        components=pca.components_,
        variance=pca.explained_variance_,
        variance_ratio=pca.explained_variance_ratio_,
    )

def decompose_new_point(new_feature_vector, model_path="pca_kidney_model.npz"):
    # 1. Load the matrices
    data = np.load(model_path)
    mean_vec = data['mean']        # Shape: (n_features,)
    components = data['components'] # Shape: (n_modes, n_features)
    variance = data['variance']    # Shape: (n_modes,)
    
    # 2. Pre-processing: Center the data
    # We subtract the "Average Kidney" from our new sample
    centered_vector = new_feature_vector - mean_vec
    
    # 3. Projection (The "Transform" step)
    # Dot product with the transposed components matrix
    # Shape: (n_features,) dot (n_features, n_modes) -> (n_modes,)
    scores = np.dot(centered_vector, components.T)
    
    # 4. Calculate Sigma (Z-Score)
    # How many standard deviations is this from the mean?
    sigmas = scores / np.sqrt(variance)
    
    return scores, sigmas


def reconstruct_from_scores(scores, model_path="pca_kidney_model.npz"):
    # 1. Load Model
    data = np.load(model_path)
    mean_vec = data['mean']
    components = data['components']
    
    # 2. Reverse Projection
    # Scores (1, n_modes) dot Components (n_modes, n_features) -> (1, n_features)
    # This rebuilds the shape variation from the origin
    shape_variation = np.dot(scores, components)
    
    # 3. Add Mean
    # Move the shape from the origin back to the "Average Kidney" location
    reconstructed_vector = shape_variation + mean_vec
    
    return reconstructed_vector


def classify_shapes(features_reduced, n_clusters=2, random_state=0):
    """
    Cluster shapes.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features_reduced)
    return labels, kmeans


# --- Usage Example ---
# 1. Decompose
# scores, _ = decompose_new_point(original_vector)

# 2. Reconstruct
# recon_vector = reconstruct_from_scores(scores)

# 3. Visualize (using the mesh function from before)
# mesh = mesh_from_features(recon_vector, original_shape=(300,300,300), grid_size=48)
# mesh.show()

# --- Example Usage ---
# new_vector = features_from_mask(new_mask, grid_size=64)
# scores, sigmas = decompose_new_point(new_vector)

# print(f"PC1 Score: {scores[0]:.4f} ({sigmas[0]:.2f} Sigmas)")

# Usage:
# save_pca_model(pca)