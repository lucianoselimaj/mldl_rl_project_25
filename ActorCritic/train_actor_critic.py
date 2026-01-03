import argparse
import torch
import wandb
import os

from env.custom_hopper import *
from agent import Agent, Policy


# We wrap the main logic in a function that accepts 'config' and 'run_name'
def train_actor_critic(config=None, run_name=None):
    # Initialize wandb if it hasn't been initialized
    if wandb.run is None:
        run = wandb.init(project="actor-critic", config=config, name=run_name)

    # Use wandb.config, which merges defaults with sweep params
    cfg = wandb.config

    env = gym.make('CustomHopper-source-v0')

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)

    # -------------------------------------------------------
    # Setup Agent using configuration
    # -------------------------------------------------------
    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)


    agent = Agent(
        policy,
        device=cfg.device,
        actor_critic=cfg.actor_critic,
        baseline=cfg.baseline
    )

    # -------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------
    for episode in range(cfg.n_episodes):
        done = False
        train_reward = 0
        state = env.reset()

        while not done:
            action, action_probabilities = agent.get_action(state)
            previous_state = state

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            agent.store_outcome(previous_state, state, action_probabilities, reward, done)

            train_reward += reward

        agent.update_policy()

        # --- WANDB LOGGING ---
        # Log metrics every episode (or every N episodes to reduce traffic)
        wandb.log({
            "training/episode_return": train_reward,
            "training/episode": episode,
            "parameters/baseline": cfg.baseline
        })
        # ---------------------

        if (episode + 1) % 1000 == 0:
            print(f'Training episode: {episode} | Return: {train_reward}')

    # Save model using the run name
    # Save to ActorCritic/saved_models
    save_dir = os.path.join("saved_models")
    os.makedirs(save_dir, exist_ok=True)

    # We use the run_name passed to the function (or wandb.run.name if available)
    final_name = run_name if run_name else wandb.run.name
    torch.save(agent.policy.state_dict(), os.path.join(save_dir, f"{final_name}.mdl"))

    wandb.finish()


# Keep this for execution via command line (sweeps often call this file directly)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--actor-critic', action='store_true')
    parser.add_argument('--baseline', default=0, type=float)
    args = parser.parse_args()

    # Convert args to dictionary for wandb
    # Assuming the sweep agent passes arguments or sets env vars
    train_actor_critic(config=vars(args), run_name="sweep_run")