import numpy as np
import zarr
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
import os
import logging
import seaborn as sns
import glob

from .pca import _elbow_index


def plot_deep_pca_performance(
    model_pth_path: str = None, 
    output_image_path: str = None,
    marginal_mse_path: str = None, 
    cumulative_mse_path: str = None, 
    n_components: int = None
):
    # 1. Load Variance Data
    checkpoint = torch.load(model_pth_path, map_location='cpu', weights_only=False)
    latent_std = checkpoint['latent_std'][:]
    variance = latent_std ** 2
    var_ratio = variance / np.sum(variance)

    # 2. Create plot
    return _plot_pca_performance(var_ratio, output_image_path, marginal_mse_path, cumulative_mse_path, n_components)


def plot_pca_performance(
    pca_zarr_path: str = None, 
    output_image_path: str = None,
    marginal_mse_path: str = None, 
    cumulative_mse_path: str = None, 
    n_components: int = None
):
    # 1. Load Variance Data
    z_pca = zarr.open(pca_zarr_path, mode='r')
    var_ratio = z_pca['variance_ratio'][:] 

    # 2. Create plot
    return _plot_pca_performance(var_ratio, output_image_path, marginal_mse_path, cumulative_mse_path, n_components)


def _plot_pca_performance(var_ratio, output_image_path, marginal_mse_path, cumulative_mse_path, n_components):
    limit = n_components if n_components else var_ratio.size
    var_ratio = var_ratio[:limit]
    cum_var_ratio = np.cumsum(var_ratio)
    x_var = np.arange(1, len(var_ratio) + 1)

    # Elbow of the cumulative explained variance curve: the point where the
    # curve bends furthest from a straight line connecting its endpoints,
    # i.e. where additional components stop meaningfully adding variance.
    elbow_idx = _elbow_index(x_var.astype(float), cum_var_ratio)
    elbow_k = int(x_var[elbow_idx])
    logging.info(
        f"PCA performance: Elbow of cumulative explained variance at "
        f"n_components={elbow_k} ({cum_var_ratio[elbow_idx] * 100:.1f}% variance explained)"
    )

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
    axes[0, 1].axvline(
        elbow_k, color='red', linestyle='--', linewidth=1.5,
        label=f"Elbow (k={elbow_k}, {cum_var_ratio[elbow_idx] * 100:.1f}%)"
    )
    axes[0, 1].set_title("Cumulative Explained Variance", fontsize=14)
    axes[0, 1].set_ylabel("Total Variance Ratio")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].grid(linestyle='--', alpha=0.5)
    axes[0, 1].legend(loc='lower right', fontsize='small')

    # --- MSE PLOTTING HELPER ---
    def render_mse_plot(ax, df, title, y_limit):
        x_steps = df.columns.astype(str)
        x_steps = [float(x[3:]) for x in x_steps] # remove 'PC_'
        
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

    return elbow_k


def plot_pca_comparison(
    models: list = None,
    output_image_path: str = None,
    n_components: int = None,
):
    """
    Overlays linear vs. deep PCA reconstruction error (median +/- IQR) for one
    or more representations, one row per representation, for a direct,
    scale-consistent comparison of the two methods (and, with multiple rows,
    across representations) in a single figure.

    `models` is a list of dicts, one per representation, each with keys:
        'name' (str, used in panel titles),
        'linear_marginal_mse_path', 'linear_cumulative_mse_path',
        'deep_marginal_mse_path', 'deep_cumulative_mse_path'
    Any path may be omitted/None to skip that curve.

    Requires pca_performance() and deep_pca_performance() to have been run:
    both normalize error by ||mean feature vector|| (see pca_performance),
    so the two curves are on the same scale and safe to overlay.
    """
    n_rows = len(models)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 6 * n_rows), squeeze=False)

    for row, model_info in enumerate(models):
        name = model_info.get('name', f'Model {row + 1}')
        panels = [
            (axes[row, 0], model_info.get('linear_marginal_mse_path'), model_info.get('deep_marginal_mse_path'), f"{name}: Marginal Reconstruction Error"),
            (axes[row, 1], model_info.get('linear_cumulative_mse_path'), model_info.get('deep_cumulative_mse_path'), f"{name}: Cumulative Reconstruction Error"),
        ]

        for ax, linear_path, deep_path, title in panels:
            for path, label, line_color, band_color in [
                (linear_path, 'Linear PCA', 'steelblue', 'lightskyblue'),
                (deep_path, 'Deep PCA', 'indigo', 'mediumpurple'),
            ]:
                if not path:
                    continue

                df = pd.read_csv(path, index_col=0)
                if n_components is not None:
                    df = df.iloc[:, :n_components]

                x_steps = [float(str(c)[3:]) for c in df.columns]  # strip 'PC_' prefix
                median = df.median(axis=0)
                q1 = df.quantile(0.25, axis=0)
                q3 = df.quantile(0.75, axis=0)

                ax.fill_between(x_steps, q1, q3, color=band_color, alpha=0.3)
                ax.plot(x_steps, median, color=line_color, lw=2, label=f'{label} (median)')

                elbow_idx = _elbow_index(np.asarray(x_steps, dtype=float), median.values)
                elbow_k = x_steps[elbow_idx]
                ax.axvline(
                    elbow_k, color=line_color, linestyle='--', linewidth=1.2, alpha=0.8,
                    label=f'{label} elbow (k={int(elbow_k)})'
                )

            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Number of components")
            ax.set_ylabel("Reconstruction Error (relative to ||mean||)")
            ax.grid(linestyle='--', alpha=0.5)
            ax.legend(loc='best', fontsize='small')

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_pca_reconstruction_performance(
    dice_csv_path: str = None, 
    hausdorff_csv_path: str = None, 
    output_image_path: str = None
):
    """
    Updated to handle the "Mean" index at index 0 and "GT" at the end.
    Produces side-by-side plots for Dice-based Error and Hausdorff Distance.
    """
    # 1. Load Data
    df_dice = pd.read_csv(dice_csv_path, index_col=0)
    df_haus = pd.read_csv(hausdorff_csv_path, index_col=0)

    # 2. Filter Logic: 
    # We want to plot everything except the "GT" column (which is just zero error).
    def get_plot_data(df):
        # Keeps "Mean" and numeric steps, drops "GT"
        valid_cols = [c for c in df.columns if str(c).upper() != 'GT']
        return df[valid_cols], valid_cols

    df_dice_plot, steps = get_plot_data(df_dice)
    df_haus_plot, _ = get_plot_data(df_haus)
    
    # x_numeric will be 0 (Mean), 1 (First K), 2 (Second K), etc.
    x_numeric = np.arange(len(steps))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plt.subplots_adjust(wspace=0.25)

    # --- PLOTTING HELPER ---
    def render_panel(ax, df, title, ylabel, color, line_color):
        median = df.median(axis=0)
        q1 = df.quantile(0.25, axis=0)
        q3 = df.quantile(0.75, axis=0)
        d_min = df.min(axis=0)
        d_max = df.max(axis=0)
        
        # Area shading
        ax.fill_between(x_numeric, d_min, d_max, color='gray', alpha=0.1, label='Min-Max Range')
        ax.fill_between(x_numeric, q1, q3, color=color, alpha=0.3, label='IQR (25th-75th)')
        
        # Median line
        ax.plot(x_numeric, median, color=line_color, alpha=0.9, linestyle='-', lw=2.5, zorder=4)
        ax.scatter(x_numeric, median, color=line_color, s=60, label='Median', edgecolors='white', zorder=5)

        # 1. Ensure a tick exists for every single data point
        ax.set_xticks(x_numeric)
        
        # 2. Create a list of labels where only every 5th one is shown
        # We use an empty string '' for the ones we want to hide
        n = 5
        custom_labels = [
            str(label) if i % n == 0 else '' 
            for i, label in enumerate(steps)
        ]
        
        # 3. Apply the custom labels
        ax.set_xticklabels(custom_labels, rotation=0)
        
        # Aesthetics
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
        ax.set_ylabel(ylabel, fontsize=12, fontweight='semibold')
        ax.set_xlabel("Reconstruction Progress (Components)", fontsize=12)
        
        ax.set_ylim(0, df.values.max() * 1.15)
        ax.grid(axis='x', linestyle=':', alpha=0.2) # Optional: light vertical grid
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend(loc='upper right', fontsize='small', frameon=True)

    # --- Panel 1: Dice Error ---
    render_panel(
        axes[0], df_dice_plot, 
        "Overlap Accuracy (1 - Dice)", 
        "Error (Lower is Better)",
        color='lightskyblue', line_color='steelblue'
    )

    # --- Panel 2: Hausdorff Distance ---
    render_panel(
        axes[1], df_haus_plot, 
        "Boundary Accuracy (Hausdorff)", 
        "Distance (mm)",
        color='mediumpurple', line_color='indigo'
    )

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return True




def plot_reconstruction_fidelity(
    dice_csv_path: str = None,
    hausdorff_csv_path: str = None,
    output_image_path: str = None,
):
    """
    Plots the distribution of reconstruction accuracy across spectral orders.
    Calculates 1 - Dice for the error plot to show convergence toward zero.
    """
    # 1. Load Data
    df_dice = pd.read_csv(dice_csv_path, index_col=0)
    df_haus = pd.read_csv(hausdorff_csv_path, index_col=0)
    
    # If Dice is provided, convert to Error (1 - Dice)
    df_dice_error = 1 - df_dice

    # 2. Setup Plotting Grid
    ncols = 2
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 7))

    # --- PLOTTING HELPER ---
    def render_panel(ax, df, title, ylabel, color, line_color):
        # Ensure columns are sorted numerically (1, 2, 3...)
        cols = sorted(df.columns, key=lambda x: int(x) if str(x).isdigit() else 0)
        df_plot = df[cols]
        
        x_numeric = np.arange(len(cols))
        
        median = df_plot.median(axis=0)
        q1 = df_plot.quantile(0.25, axis=0)
        q3 = df_plot.quantile(0.75, axis=0)
        d_min = df_plot.min(axis=0)
        d_max = df_plot.max(axis=0)
        
        # Area shading
        ax.fill_between(x_numeric, d_min, d_max, color='gray', alpha=0.1, label='Min-Max Range')
        ax.fill_between(x_numeric, q1, q3, color=color, alpha=0.3, label='IQR (25th-75th)')
        
        # Median line
        ax.plot(x_numeric, median, color=line_color, alpha=0.9, lw=2.5, zorder=4)
        ax.scatter(x_numeric, median, color=line_color, s=40, label='Median', edgecolors='white', zorder=5)
        
        # Aesthetics
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel(ylabel, fontsize=11, fontweight='semibold')
        ax.set_xlabel("Spectral Reconstruction Order", fontsize=11)
        
        # Thinned out X-labels for 1-36 range (show every 5th order)
        ax.set_xticks(x_numeric[::5])
        ax.set_xticklabels(cols[::5])
        
        ax.set_ylim(0, df_plot.values.max() * 1.1)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend(loc='upper right', fontsize='small')
        sns.despine(ax=ax)

    # --- Panel 1: Dice Error ---
    render_panel(
        axes[0], df_dice_error, 
        "Spectral Convergence (Overlap)", 
        "1 - Dice Score (Lower is Better)",
        color='lightskyblue', line_color='steelblue'
    )

    # --- Panel 2: Hausdorff (Optional) ---
    render_panel(
        axes[1], df_haus, 
        "Boundary Convergence (Distance)", 
        "Hausdorff Distance (mm)",
        color='mediumpurple', line_color='indigo'
    )

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.show()



def plot_mask_sections(
    dataset_zarr_path: str = None, 
    dir_png: str = None,
    n_components: int = None
):
    # 1. Load the dataset
    # Expected shape: (n_cols, n_rows, x, y, z)
    root = zarr.open(dataset_zarr_path, mode='r')
    data = root['masks']
    n_cols, n_rows, sx, sy, sz = data.shape
    
    centers = (sx // 2, sy // 2, sz // 2)
    os.makedirs(dir_png, exist_ok=True)

    # Define the planes to extract
    # Mapping plane name to the slicing logic
    if n_components is not None:
        planes = {
            "axial":     lambda d: d[:n_components, :, :, :, centers[2]],  # (C, R, X, Y)
            "sagittal":  lambda d: d[:n_components, :, :, centers[1], :].transpose(0, 1, 3, 2),  # (C, R, X, Z)
            "coronal":   lambda d: d[:n_components, :, centers[0], :, :].transpose(0, 1, 3, 2),  # (C, R, Y, Z)
        }
    else:
        planes = {
            "axial":     lambda d: d[:, :, :, :, centers[2]],  # (C, R, X, Y)
            "sagittal":  lambda d: d[:, :, :, centers[1], :].transpose(0, 1, 3, 2),  # (C, R, X, Z)
            "coronal":   lambda d: d[:, :, centers[0], :, :].transpose(0, 1, 3, 2),  # (C, R, Y, Z)
        }

    for name, slice_func in planes.items():
        # Extract the 4D grid of 2D slices
        grid_4d = slice_func(data) 
        
        # 2. Arrange into a 2D Mosaic
        # np.block expects a list of lists: [[row1_img1, row1_img2], [row2_img1...]]
        # We iterate rows first, then columns
        rows_list = []
        for r in range(n_rows):
            cols_list = [grid_4d[c, r] for c in range(n_cols)]
            rows_list.append(cols_list)
        
        mosaic = np.block(rows_list)
        
        # 3. Convert to Image
        mosaic = (mosaic * 255).astype(np.uint8)
            
        img = Image.fromarray(mosaic)
        img.save(os.path.join(dir_png, f"mosaic_{name}.png"))
        print(f"Saved {name} mosaic to {dir_png}")




def plot_shape_fingerprints(
    dir_csv: str = None, 
    dir_png: str = None,
):
    """
    Reads all shape CSVs and creates a vertical stack of 'fingerprint' plots for each.
    The y-axis range is fixed across all rows within a feature to the global min-max.
    """
    os.makedirs(dir_png, exist_ok=True)
    csv_files = glob.glob(os.path.join(dir_csv, "*.csv"))

    for csv_path in csv_files:
        feat_name = os.path.basename(csv_path).replace(".csv", "")
        df = pd.read_csv(csv_path, index_col=0) 
        
        # Calculate global bounds for this specific feature
        y_min = df.values.min()
        y_max = df.values.max()
        
        # Add a small margin so the peaks/valleys don't touch the edges
        margin = (y_max - y_min) * 0.05
        y_lims = (y_min - margin, y_max + margin)

        n_rows = len(df)
        fig, axes = plt.subplots(
            nrows=n_rows, 
            ncols=1, 
            figsize=(12, 1.2 * n_rows), # Slightly tighter height
            sharex=True
        )
        
        if n_rows == 1:
            axes = [axes]
            
        for i, (idx, row_data) in enumerate(df.iterrows()):
            ax = axes[i]
            
            x_vals = range(len(row_data))
            y_vals = row_data.values
            
            # Plot fingerprint
            ax.fill_between(x_vals, y_vals, alpha=0.3, color='teal')
            ax.plot(x_vals, y_vals, color='teal', lw=1.5)
            
            # Fix the scale to the global range
            ax.set_ylim(y_lims)
            
            # Formatting
            ax.set_ylabel(f"Row {idx}", rotation=0, labelpad=40, va='center')
            ax.set_yticks([])  
            
            # Remove top, right, and left spines
            sns.despine(ax=ax, left=True)

        # Only label the bottom x-axis
        plt.xticks(range(len(df.columns)), df.columns, rotation=45, ha='right', fontsize=8)
        
        fig.suptitle(f"Shape Fingerprints: {feat_name}", fontsize=16, y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(dir_png, f"{feat_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"Fingerprint plots saved to {dir_png}")