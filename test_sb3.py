"""Test SAC's sb3 models on the OpenAI Gym Hopper environment"""
import argparse

import torch
import gym
import os

from env.custom_hopper import *
from stable_baselines3 import SAC


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=10, type=int, help='Number of test episodes')

    return parser.parse_args()

args = parse_args()


def main():

    env = gym.make('CustomHopper-source-v0')
    #env = gym.make('CustomHopper-target-v0')

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    # Load from Sac/saved_models
    # Assumes args.model is the filename (e.g. "sac_final_seed0_id_xyz")
    model_path = os.path.join("saved_models", args.model)
    model = SAC.load(model_path, device=args.device)
    all_rewards = []
    for episode in range(args.episodes):
       done = False
       test_reward = 0
       state = env.reset()


       while not done:

          action, _ = model.predict(state, deterministic=True)

          state, reward, done, info = env.step(action)

          if args.render:
             env.render()



          test_reward += reward
       all_rewards.append(test_reward)
       print(f"Episode: {episode} | Return: {test_reward}")
    print(f"Mean_rewards: {np.mean(all_rewards)}")
    print(f"Std_rewards: {np.std(all_rewards)}")



if __name__ == '__main__':
    main()