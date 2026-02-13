import numpy as np
import zarr
import matplotlib.pyplot as plt
import pandas as pd


def plot_pca_performance(
    pca_zarr_path: str, 
    output_image_path: str,
    marginal_mse_path: str = None, 
    cumulative_mse_path: str = None, 
    n_components: int = None
):
    # 1. Load Variance Data
    z_pca = zarr.open(pca_zarr_path, mode='r')
    var_ratio = z_pca['variance_ratio'][:] 
    limit = n_components if n_components else var_ratio.size
    var_ratio = var_ratio[:limit]
    cum_var_ratio = np.cumsum(var_ratio)
    x_var = np.arange(1, len(var_ratio) + 1)

    # 2. Determine Global Y-Max for MSE plots
    global_mse_max = 0
    mse_dfs = {}
    
    for path, key in zip([marginal_mse_path, cumulative_mse_path], ['marginal', 'cumulative']):
        if path:
            df = pd.read_csv(path, index_col=0)
            if limit is not None:
                # Assuming the first column is 'Average', limit + 1 columns are kept
                df = df.iloc[:, :limit + 1] 
            mse_dfs[key] = df
            global_mse_max = max(global_mse_max, df.values.max())

    # 3. Determine Layout
    show_mse = len(mse_dfs) > 0
    nrows = 2 if show_mse else 1
    
    fig, axes = plt.subplots(nrows, 2, figsize=(16, 5 * nrows), squeeze=False)
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    # --- TOP ROW: Variance ---
    axes[0, 0].scatter(x_var, var_ratio, color='steelblue', s=15, alpha=0.8)
    axes[0, 0].set_title("Marginal Explained Variance", fontsize=14)
    axes[0, 0].set_ylabel("Variance Ratio")
    axes[0, 0].set_ylim(0, None) # TWEAK: Start at zero
    axes[0, 0].grid(linestyle='--', alpha=0.5)

    axes[0, 1].scatter(x_var, cum_var_ratio, color='firebrick', s=15, alpha=0.8)
    axes[0, 1].set_title("Cumulative Explained Variance", fontsize=14)
    axes[0, 1].set_ylabel("Total Variance Ratio")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].grid(linestyle='--', alpha=0.5)

    # --- MSE PLOTTING HELPER ---
    def render_mse_plot(ax, df, title, y_limit):
        x_steps = df.columns.astype(float)
        
        median = df.median(axis=0)
        q1 = df.quantile(0.25, axis=0)
        q3 = df.quantile(0.75, axis=0)
        d_min = df.min(axis=0)
        d_max = df.max(axis=0)
        
        # TWEAK: Darker Min-Max shading (alpha=0.2 instead of 0.1)
        ax.fill_between(x_steps, d_min, d_max, color='gray', alpha=0.2, label='Min-Max Range')
        ax.fill_between(x_steps, q1, q3, color='mediumpurple', alpha=0.4, label='Interquartile Range')
        
        ax.scatter(x_steps, median, color='indigo', s=25, label='Median MSE', zorder=3)
        ax.plot(x_steps, median, color='indigo', alpha=0.7, linestyle='-', lw=1.5, zorder=4)
        
        ax.set_title(title, fontsize=14)
        ax.set_ylabel("Mean Squared Error")
        ax.set_xlabel("Principal Components")
        
        ax.set_ylim(0, y_limit * 1.1) # Unified scale with headroom
        ax.set_xlim(0, limit)
        
        ax.grid(linestyle='--', alpha=0.5)
        ax.legend(loc='best', fontsize='small')

    # --- BOTTOM ROW: MSE ---
    if show_mse:
        if 'marginal' in mse_dfs:
            render_mse_plot(axes[1, 0], mse_dfs['marginal'], "Marginal Reconstruction Error", global_mse_max)
        else:
            axes[1, 0].set_axis_off()

        if 'cumulative' in mse_dfs:
            render_mse_plot(axes[1, 1], mse_dfs['cumulative'], "Cumulative Reconstruction Error", global_mse_max)
        else:
            axes[1, 1].set_axis_off()
    else:
        axes[0, 0].set_xlabel("Principal Component Index")
        axes[0, 1].set_xlabel("Principal Component Index")

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.show()
    plt.close()