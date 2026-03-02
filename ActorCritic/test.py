"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import numpy as np
import torch
import gym
import os

from env.custom_hopper import *
from ActorCritic.agent import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=10, type=int, help='Number of test episodes')
    parser.add_argument('--env-id', default='CustomHopper-target-v0', type=str, help='Environment id')
    return parser.parse_args()


args = parse_args()


def main():

    env = gym.make(args.env_id)

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)

    # Load from ActorCritic/saved_models
    # Assumes args.model is just the filename (e.g., "sweep_run.mdl")
    model_path = os.path.join("ActorCritic", "saved_models", args.model)
    policy.load_state_dict(torch.load(model_path, map_location=args.device), strict=True)

    agent = Agent(policy, device=args.device)
    all_rewards = []
    for episode in range(args.episodes):
        done = False
        test_reward = 0
        state = env.reset()

        while not done:

            action, _ = agent.get_action(state, evaluation=True)

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            if args.render:
                env.render()

            test_reward += reward

        all_rewards.append(test_reward)
        print(f"Episode: {episode} | Return: {test_reward}")
    print(f"Mean_rewards: {np.mean(all_rewards)}")
    print(f"Std_rewards: {np.std(all_rewards)}")


if __name__ == '__main__':
    main()