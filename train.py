"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse

import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--actor-critic', action='store_true', help='Use Actor-Critic algorithm') # If it is present, it becomes True 
    parser.add_argument('--baseline', default = 0, type=float, help='Value of the baseline')


    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())


	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	agent = Agent(policy, device=args.device, actor_critic=args.actor_critic, baseline=args.baseline)

    #
    # TASK 2 and 3: interleave data collection to policy updates
    #

	for episode in range(args.n_episodes):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state

		while not done:  # Loop until the episode is over

			# Action, log_prob and value associated with the actual state
			action, action_probabilities = agent.get_action(state)
			previous_state = state # Save actual state

			state, reward, done, info = env.step(action.detach().cpu().numpy()) 

			agent.store_outcome(previous_state, state, action_probabilities, reward, done)

			train_reward += reward

		agent.update_policy()

		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode)
			print('Episode return:', train_reward)
		""""""""""
		NO BASELINE
		
		Training episode: 19999
		Episode return: 127.88130140910782
		Training episode: 39999
		Episode return: 50.03305632919815
		Training episode: 59999
		Episode return: 197.72080436009026
		Training episode: 79999
		Episode return: 148.54096615044057
		Training episode: 99999
		Episode return: 153.94403360808414
		
		BASELINE = 20
		Training episode: 19999
		Episode return: 181.28481805191194
		Training episode: 39999
		Episode return: 211.72094164361536
		Training episode: 59999
		Episode return: 178.94370237672163
		Training episode: 79999
		Episode return: 170.94069108691951
		Training episode: 99999
		Episode return: 234.29935339953536

		
		"""""""""

	torch.save(agent.policy.state_dict(), "model.mdl")

	

if __name__ == '__main__':
	main()