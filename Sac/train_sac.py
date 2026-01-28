import os
import torch
import wandb
import numpy as np

from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from wandb.integration.sb3 import WandbCallback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import beta

def plot_distribution(adv_beta, save_path="advbeta_evolution.pdf", param_names=None, start_ep=750, step_perc=0.30):
    history = adv_beta.history
    episodes = np.array(history['episode'])
    
    if len(episodes) == 0: 
        return 

    alphas_hist = np.array(history['alphas'])
    betas_hist = np.array(history['betas'])
    
    if alphas_hist.ndim == 1:
        n_dims = 1
        alphas_hist = alphas_hist.reshape(-1, 1)
        betas_hist = betas_hist.reshape(-1, 1)
    else:
        n_dims = alphas_hist.shape[1]

    if param_names is None:
        param_names = [f"Param {i}" for i in range(n_dims)]
    elif len(param_names) != n_dims:
        if len(param_names) < n_dims:
            param_names += [f"Param {i}" for i in range(len(param_names), n_dims)]
        else:
            param_names = param_names[:n_dims]

    fig, axes = plt.subplots(1, n_dims, figsize=(5 * n_dims, 5), constrained_layout=True)
    if n_dims == 1: 
        axes = [axes]

    last_ep = episodes[-1]

    if last_ep > start_ep:
        duration = last_ep - start_ep
        targets = [start_ep]
        
        targets.append(int(start_ep + duration * step_perc))
        targets.append(int(start_ep + duration * (step_perc * 2)))
        targets.append(last_ep)
    else:
        targets = [last_ep]

    colors = cm.viridis(np.linspace(0.2, 0.9, len(targets)))
    x = np.linspace(0, 1, 500)

    for dim in range(n_dims):
        ax = axes[dim]
        
        low = adv_beta.lower[dim]
        high = adv_beta.upper[dim]
        x_phys = low + x * (high - low)

        ax.plot(x_phys, np.ones_like(x), 'k--', alpha=0.4, label='Uniform (Start)')

        plotted_indices = set()
        
        for idx, t in enumerate(targets):
            actual_idx = np.argmin(np.abs(episodes - t))
            
            if actual_idx in plotted_indices and idx != len(targets)-1:
                continue
            
            plotted_indices.add(actual_idx)

            curr_ep = episodes[actual_idx]
            a = alphas_hist[actual_idx, dim]
            b = betas_hist[actual_idx, dim]
            y = beta.pdf(x, a, b)
            
            if idx == len(targets) - 1:
                lbl = f"Final (Ep {curr_ep})"
                ls = '-' 
                lw = 2.5
            elif idx == 0:
                lbl = f"Start (Ep {curr_ep})"
                ls = '-'
                lw = 2
            else:
                # Calcola la percentuale per la label in base all'indice e al passo
                perc_val = int(step_perc * idx * 100)
                lbl = f"Ep {curr_ep} (~{perc_val}%)"
                ls = '-'
                lw = 1.5

            ax.plot(x_phys, y, color=colors[idx], linestyle=ls, linewidth=lw, label=lbl)
            
            if idx == len(targets)-1: 
                ax.fill_between(x_phys, y, color=colors[idx], alpha=0.1)

        ax.set_title(f"Evolution: {param_names[dim]}")
        ax.set_xlabel("Mass (kg)")
        ax.grid(True, alpha=0.3)
        
        if dim == 0:
            ax.set_ylabel("Probability Density")
            ax.legend()

    plt.savefig(save_path)
    plt.close()

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

    train_env = Monitor(
        gym.make(
            config["env_id"],
            use_ext=bool(config.get("use_ext", False)),
            curriculum_seed=int(config["seed"]),
            randomize_on_reset=bool(config.get("randomize_on_reset", True)),
        )
    )

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

    # Plot beta distribution in a pdf file
    use_ext = getattr(train_env.unwrapped, "use_ext", False)
    if use_ext:
        adv_beta = train_env.unwrapped.curriculum
        my_params = ["Thigh Mass", "Leg Mass", "Foot Mass"]
        plot_distribution(
            adv_beta, 
            save_path=f"evolution_{run_name}.pdf", 
            param_names=my_params,
            start_ep=750,    # Start of curriculum
            step_perc=0.30   # Plot's pace
        )

    train_env.close()
