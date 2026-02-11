import numpy as np
import zarr
import matplotlib.pyplot as plt
import pandas as pd


def plot_pca_performance(
    pca_zarr_path: str, 
    output_image_path: str,
    marginal_dice_path: str = None, 
    cumulative_dice_path: str = None, 
    n_components: int = None
):
    """
    Visualizes PCA performance. 
    Switches to scatter plots and dynamic row count based on Dice availability.
    """
    # 1. Load Variance Data
    z_pca = zarr.open(pca_zarr_path, mode='r')
    var_ratio = z_pca['variance_ratio'][:] 
    
    # Apply n_components limit if provided
    limit = n_components if n_components else var_ratio.size
    var_ratio = var_ratio[:limit]
    cum_var_ratio = np.cumsum(var_ratio)
    x = np.arange(1, len(var_ratio) + 1)

    # 2. Determine Layout
    # If no dice paths are provided, we only need 1 row
    show_dice = marginal_dice_path is not None or cumulative_dice_path is not None
    nrows = 2 if show_dice else 1
    
    fig, axes = plt.subplots(nrows, 2, figsize=(16, 5 * nrows), sharex=True, squeeze=False)
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    # --- TOP ROW: Variance (Scatter) ---
    axes[0, 0].scatter(x, var_ratio, color='steelblue', s=15, alpha=0.8)
    axes[0, 0].set_title("Marginal Explained Variance", fontsize=14)
    axes[0, 0].set_ylabel("Variance Ratio")
    axes[0, 0].grid(linestyle='--', alpha=0.5)

    axes[0, 1].scatter(x, cum_var_ratio, color='firebrick', s=15, alpha=0.8)
    axes[0, 1].set_title("Cumulative Explained Variance", fontsize=14)
    axes[0, 1].set_ylabel("Total Variance Ratio")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].grid(linestyle='--', alpha=0.5)

    # --- DICE PLOTTING HELPER ---
    def plot_dice_distribution(ax, data_path, title):
        df = pd.read_csv(data_path)
        # Ensure we only plot up to the same n_components limit
        df = df.iloc[:, :limit]
        
        # Calculate Stats
        median = df.median(axis=0)
        q1 = df.quantile(0.25, axis=0)
        q3 = df.quantile(0.75, axis=0)
        d_min = df.min(axis=0)
        d_max = df.max(axis=0)
        
        # Scatter for Median
        ax.scatter(x, median, color='indigo', s=20, label='Median Dice', zorder=3)
        # Fill ranges to maintain context of population spread
        ax.fill_between(x, d_min, d_max, color='gray', alpha=0.1, label='Min-Max')
        ax.fill_between(x, q1, q3, color='mediumpurple', alpha=0.3, label='IQR')
        
        ax.set_title(title, fontsize=14)
        ax.set_ylabel("Dice Coefficient")
        ax.set_xlabel("Principal Component Index")
        ax.set_ylim(0, 1.0)
        ax.grid(linestyle='--', alpha=0.5)
        ax.legend(loc='lower right', fontsize='small')

    # --- BOTTOM ROW: Dice (Optional) ---
    if show_dice:
        if marginal_dice_path:
            plot_dice_distribution(axes[1, 0], marginal_dice_path, "Marginal Accuracy")
        else:
            axes[1, 0].set_axis_off()

        if cumulative_dice_path:
            plot_dice_distribution(axes[1, 1], cumulative_dice_path, "Cumulative Accuracy")
        else:
            axes[1, 1].set_axis_off()
    else:
        # Add x-labels to the variance row if dice row is missing
        axes[0, 0].set_xlabel("Principal Component Index")
        axes[0, 1].set_xlabel("Principal Component Index")

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.show()
    plt.close()