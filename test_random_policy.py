"""Test a random policy on the OpenAI Gym Hopper environment.

    
    TASK 1: Play around with this code to get familiar with the
            Hopper environment.

            For example:
                - What is the state space in the Hopper environment? Is it discrete or continuous?
                - What is the action space in the Hopper environment? Is it discrete or continuous?
                - What is the mass value of each link of the Hopper environment, in the source and target variants respectively?
                - what happens if you don't reset the environment even after the episode is over?
                - When exactly is the episode over?
                - What is an action here?
"""
import pdb

import gym

from env.custom_hopper import *


def main():
	#env = gym.make('CustomHopper-source-v0')
	env = gym.make('CustomHopper-target-v0')

	print('State space:', env.observation_space) # state-space
	print('Action space:', env.action_space) # action-space
	print('Dynamics parameters:', env.get_parameters()) # masses of each link of the Hopper
	#state-space: it is an 11-dimensional continuous vector from -inf to inf
	#action-space: 3-dimensional continuous vector from -1 to 1. It represents the torque to each joint
	#source-mass: Dynamics parameters: [2.565634   4.05789051 2.7813567  5.31557477]
	#target-mass: Dynamics parameters: [3.66519143 4.05789051 2.7813567  5.31557477]



	n_episodes = 500
	render = False

	for episode in range(n_episodes):
		done = False
		state = env.reset()	# Reset environment to initial state

		while not done:  # Until the episode is over

			action = env.action_space.sample()	# Sample random action
		
			state, reward, done, info = env.step(action)	# Step the simulator to the next timestep

			if render:
				env.render()

	

if __name__ == '__main__':
	main()