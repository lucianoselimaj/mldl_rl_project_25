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
import matplotlib.cm as cm
from scipy.stats import beta

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
    ax.set_xlabel("Physical parameter vValue")
    ax.legend()
    plt.tight_layout()
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
        plot_distribution(adv_beta, save_path=f"evolution_{run_name}.pdf")

    train_env.close()
