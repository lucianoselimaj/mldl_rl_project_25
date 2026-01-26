"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *
import yaml
from stable_baselines3 import SAC
import wandb
from wandb.integration.sb3 import WandbCallback
import torch
import os


def train_sac(config, run_name):
    """Train and test policies on the Hopper env with stable-baselines3"""
    
    # Create enviroment
    train_env = gym.make(config['env_id'])

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper


    policy_kwargs = {
        "net_arch": {
            "pi": [wandb.config.policy_pi_layer1, wandb.config.policy_pi_layer2],
            "qf": [wandb.config.policy_qf_layer1, wandb.config.policy_qf_layer2],
        },
        "activation_fn": getattr(torch.nn, wandb.config.policy_activation_fn)
    }
    
    # Model initialization
    model = SAC(
    policy=config["policy"],
    env=train_env,
    seed=config["seed"],  # Set seed for reproducibility
    learning_starts=config["learning_starts"],
    train_freq=config["train_freq"],
    gradient_steps=config["gradient_steps"],
    batch_size=config["batch_size"],
    gamma=config["gamma"],
    tau=config["tau"],
    target_update_interval=config["target_update_interval"],
    ent_coef=config["ent_coef"],
    policy_kwargs=policy_kwargs,
    device=config["device"],
    verbose = config["verbose"],
    tensorboard_log="./tb_logs",
    )


    # Model training
    model.learn(config["total_timesteps"],
        log_interval=4,   # This is the number of episodes before logging
        callback=WandbCallback(
            model_save_path=None,
            verbose=1,
        ),
    )
    
    os.makedirs("models", exist_ok=True)
    model.save(f"models/sac_final_seed{config['seed']}_id_{run_name}")

    return
