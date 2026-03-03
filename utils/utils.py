import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
from scipy.stats import beta


def to_bool(x):
    """
    Safe boolean conversion:
    - True/False stays True/False
    - "true"/"false" strings are correctly parsed
    - numeric values (0/1) are handled
    """
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float, np.integer, np.floating)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "t", "yes", "y")
    return bool(x)


def plot_distribution(adv_beta, save_path="advbeta_evolution.pdf", param_names=None):
    history = adv_beta.history
    episodes = np.array(history['episode'])
    if len(episodes) == 0: return

    alphas = np.array(history['alphas']) 
    betas  = np.array(history['betas'])   
    n_dims = alphas.shape[1]

    timestamps = [0, int(episodes[-1]*0.2), int(episodes[-1]*0.5), episodes[-1]]
    colors = cm.viridis(np.linspace(0.3, 0.9, len(timestamps)))

    fig, axes = plt.subplots(1, n_dims, figsize=(6*n_dims, 5))
    if n_dims == 1:
        axes = [axes]

    x = np.linspace(0, 1, 500)

    for dim in range(n_dims):
        ax = axes[dim]
        x_phys = adv_beta.lower[dim] + x * (adv_beta.upper[dim] - adv_beta.lower[dim])

        ax.plot(x_phys, np.ones_like(x), 'k--', alpha=0.5, label='Uniform Baseline')

        for idx, t in enumerate(timestamps):
            actual_idx = np.argmin(np.abs(episodes - t))
            y = beta.pdf(x, alphas[actual_idx, dim], betas[actual_idx, dim])
            label = "Init" if episodes[actual_idx] == 0 else f"Ep {episodes[actual_idx]}"
            ax.plot(x_phys, y, color=colors[idx], linewidth=2, label=label)
            if idx == len(timestamps)-1:
                ax.fill_between(x_phys, y, color=colors[idx], alpha=0.1)

        dim_label = param_names[dim] if param_names and dim < len(param_names) else f"Dim {dim}"
        ax.set_title(f"Evolution of sampling distribution\n({dim_label})")
        ax.set_xlabel("Kg")
        ax.set_ylabel("Probability density")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
