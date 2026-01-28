import argparse
import os
import random
import torch
import wandb
import numpy as np
import gym



from env.custom_hopper import *   # registers envs
from agent import Agent, Policy


def set_seed(seed: int):
    """Minimal seeding: enough for fair comparisons."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)       # policy init + action sampling
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_actor_critic(config=None, run_name=None):
    if wandb.run is None:
        wandb.init(project="actor-critic", config=config, name=run_name)

    cfg = wandb.config

    # ---- SEED (minimal) ----
    seed = int(getattr(cfg, "seed", 0))
    set_seed(seed)
    # ------------------------

    env_id = getattr(cfg, "env_id", "CustomHopper-source-v0")
    env = gym.make(
        env_id,
        randomize_on_reset=bool(getattr(cfg, "randomize_on_reset", True))
    )

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)


    # optional: seed env if available (older gym)
    if hasattr(env, "seed"):
        env.seed(seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)

    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]

    policy = Policy(obs_dim, act_dim)
    agent = Agent(
        policy,
        device=cfg.device,
        actor_critic=cfg.actor_critic,
        baseline=cfg.baseline
    )

    for episode in range(cfg.n_episodes):
        done = False
        train_reward = 0.0
        state = env.reset()

        while not done:
            action, action_logprob = agent.get_action(state)
            prev_state = state

            state, reward, done, info = env.step(action.detach().cpu().numpy())
            agent.store_outcome(prev_state, state, action_logprob, reward, done)
            train_reward += reward

        agent.update_policy()

        # --- WANDB LOGGING ---
        # Log metrics every episode (or every N episodes to reduce traffic)
        wandb.log({
            "training/episode_return": train_reward,
            "training/episode": episode,
            "parameters/baseline": cfg.baseline,
            "parameters/seed": seed,
        })

        if (episode + 1) % 1000 == 0:
            print(f'Training episode: {episode} | Return: {train_reward}')

    save_dir = os.path.join("saved_models")
    os.makedirs(save_dir, exist_ok=True)

    # We use the run_name passed to the function (or wandb.run.name if available)
    final_name = run_name if run_name else wandb.run.name
    torch.save(agent.policy.state_dict(), os.path.join(save_dir, f"{final_name}.mdl"))

    env.close()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="CustomHopper-source-v0", type=str)
    parser.add_argument("--n-episodes", default=100000, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--actor-critic", action="store_true")
    parser.add_argument("--baseline", default=0.0, type=float)
    parser.add_argument("--seed", default=0, type=int)   # ✅ ADDED

    args = parser.parse_args()
    train_actor_critic(config=vars(args), run_name="manual_run")
