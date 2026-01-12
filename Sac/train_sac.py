import os
import torch
import wandb

from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from wandb.integration.sb3 import WandbCallback


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
    train_env = Monitor(gym.make(config["env_id"], use_ext=True, gmm_seed=config["seed"]))

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

    train_env.close()
