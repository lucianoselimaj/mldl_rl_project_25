import os
import torch
import wandb
import random

from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from wandb.integration.sb3 import WandbCallback

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist

def set_seed(seed):
    """Minimal seeding: enough for fair comparisons."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)       # policy init + action sampling
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_curriculum_dynamics(adv_beta_instance, param_names=["Thigh", "Leg", "Foot"]):
    """
    Visualizza l'evoluzione completa dei parametri Beta avversari.
    """
    history = adv_beta_instance.history
    
    if len(history['episode']) == 0:
        print("Nessun dato storico trovato (forse Warmup non ancora finito?)")
        return

    episodes = history['episode']
    alphas = np.array(history['alphas']) # Shape: (N_Updates, N_Dims)
    betas = np.array(history['betas'])
    means = np.array(history['means'])
    
    n_dims = alphas.shape[1]
    
    # --- PLOT 1: Evoluzione della Media (La "Strategia" dell'Avversario) ---
    plt.figure(figsize=(12, 5))
    for dim in range(n_dims):
        label = param_names[dim] if dim < len(param_names) else f"Dim {dim}"
        plt.plot(episodes, means[:, dim], label=label, linewidth=2)
    
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label="Uniforme/Neutro")
    plt.title("Strategia dell'Avversario: Dove sta spingendo le masse?", fontsize=14)
    plt.ylabel("Media Normalizzata (0=Min Mass, 1=Max Mass)")
    plt.xlabel("Episodi")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- PLOT 2: Forma della Distribuzione (PDF) nel Tempo ---
    # Mostriamo 3 istantanee: Inizio, Metà, Fine
    indices_to_show = [0, len(episodes)//2, -1]
    labels_time = ["Inizio Curriculum", "Metà Training", "Fine Training"]
    
    x = np.linspace(0, 1, 100)
    
    fig, axes = plt.subplots(1, n_dims, figsize=(15, 4), sharey=True)
    if n_dims == 1: axes = [axes] # Gestione caso 1 dimensione

    for dim in range(n_dims):
        ax = axes[dim]
        param_name = param_names[dim] if dim < len(param_names) else f"Dim {dim}"
        
        for i, idx in enumerate(indices_to_show):
            a = alphas[idx, dim]
            b = betas[idx, dim]
            
            # Calcola PDF Beta
            y = beta_dist.pdf(x, a, b)
            
            ax.plot(x, y, label=f"{labels_time[i]}\n($\\alpha$={a:.1f}, $\\beta$={b:.1f})", linewidth=2)
            ax.fill_between(x, y, alpha=0.1)

        ax.set_title(f"Distribuzione Massa: {param_name}")
        ax.set_xlabel("Valore Massa Normalizzato")
        if dim == 0: ax.set_ylabel("Densità di Probabilità")
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)

    plt.suptitle("Evoluzione della 'Cattiveria' dell'Avversario (Forma della Distribuzione)", y=1.05, fontsize=16)
    plt.tight_layout()
    plt.show()

    # --- PLOT 3 (Tecnico): Alpha e Beta grezzi ---
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 8), sharex=True)
    if n_dims == 1: axes = [axes]
    
    for dim in range(n_dims):
        ax = axes[dim]
        param_name = param_names[dim] if dim < len(param_names) else f"Dim {dim}"
        
        ax.plot(episodes, alphas[:, dim], label="Alpha (Spinge a Destra)", color='tab:blue')
        ax.plot(episodes, betas[:, dim], label="Beta (Spinge a Sinistra)", color='tab:orange')
        
        ax.set_ylabel(f"{param_name}\nValore Param")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.xlabel("Episodi")
    plt.suptitle("Dettaglio Tecnico: Valori Alpha/Beta Grezzi", y=1.02)
    plt.tight_layout()
    plt.show()

class PerEpisodeWandbCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:  # requires Monitor
                self.ep += 1
                log_dict = {
                    "episode": self.ep,  
                    "episode_return": info["episode"]["r"],
                    "episode_length": info["episode"]["l"],
                }

                env = self.model.env.envs[0].unwrapped
                masses = env.get_parameters()
                    
                log_dict.update({
                    "curriculum/mass_thigh": masses[1],
                    "curriculum/mass_leg":   masses[2],
                    "curriculum/mass_foot":  masses[3],
                })
                wandb.log(log_dict, step=self.num_timesteps)
        return True


def train_sac(config, run_name):
    # reproducibility
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    set_seed(config["seed"])

    train_env = Monitor(gym.make(config["env_id"], use_ext=False, gmm_seed=config["seed"]))
    train_env.seed(config["seed"])
    train_env.action_space.seed(config["seed"])

    policy_kwargs = {
        "net_arch": {
            "pi": [config["policy_pi_layer1"], config["policy_pi_layer2"]],
            "qf": [config["policy_qf_layer1"], config["policy_qf_layer2"]],
        },
        "activation_fn": getattr(torch.nn, config["policy_activation_fn"]),
    }

    model = SAC(
        policy=config["policy"],
        env=train_env,
        seed=config["seed"],
        learning_starts=config["learning_starts"],
        train_freq=config["train_freq"],
        gradient_steps=config["gradient_steps"],
        batch_size=config["batch_size"],
        buffer_size=config["buffer_size"],
        gamma=config["gamma"],
        tau=config["tau"],
        target_update_interval=config["target_update_interval"],
        ent_coef=config["ent_coef"],
        target_entropy=config["target_entropy"],
        policy_kwargs=policy_kwargs,
        device=config["device"],
        verbose=config["verbose"],
        tensorboard_log="./tb_logs",  # will sync into W&B because sync_tensorboard=True
    )

    callbacks = CallbackList([
        WandbCallback(verbose=1),
        PerEpisodeWandbCallback(),
    ])

    model.learn(
        total_timesteps=config["total_timesteps"],
        log_interval=1,
        callback=callbacks,
        tb_log_name=run_name,  # makes tb_logs folder clearer
    )

    save_dir = os.path.join("saved_models")
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, f"sac_final_seed{config['seed']}_id_{run_name}"))

    curriculum_instance = train_env.get_wrapper_attr('curriculum') 
    plot_curriculum_dynamics(curriculum_instance)

    train_env.close()
