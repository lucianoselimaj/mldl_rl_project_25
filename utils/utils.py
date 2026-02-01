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


def plot_distribution(adv_beta, save_path="advbeta_evolution.pdf"):
    history = adv_beta.history
    episodes = np.array(history['episode'])
    if len(episodes) == 0: return 

    alphas = np.array(history['alphas'])[:, 0]
    betas = np.array(history['betas'])[:, 0]
    
    timestamps = [0, int(episodes[-1]*0.2), int(episodes[-1]*0.5), episodes[-1]]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(0, 1, 500)

    x_phys = adv_beta.lower[0] + x * (adv_beta.upper[0] - adv_beta.lower[0])
    
    colors = cm.viridis(np.linspace(0.3, 0.9, len(timestamps)))
    
    # Baseline
    ax.plot(x_phys, np.ones_like(x), 'k--', alpha=0.5, label='Uniform Baseline')

    for idx, t in enumerate(timestamps):
        actual_idx = np.argmin(np.abs(episodes - t))
        y = beta.pdf(x, alphas[actual_idx], betas[actual_idx])
        label = "Init" if episodes[actual_idx] == 0 else f"Ep {episodes[actual_idx]}"
        ax.plot(x_phys, y, color=colors[idx], linewidth=2, label=label)
        if idx == len(timestamps)-1: ax.fill_between(x_phys, y, color=colors[idx], alpha=0.1)

    ax.set_title("Evolution of sampling distribution")
    ax.set_xlabel("Physical parameter value")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
