import numpy as np
import torch
import torch.nn.functional as F
#from matplotlib.tests.test_backend_pgf import baseline_dir
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

        """
            Actor network
        """
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(state_space, self.hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden, self.hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden, action_space) 
        )
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(state_space, self.hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, self.hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, 1) 
        )

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        action_mean = self.actor(x)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

    
        # TASK 3: forward in the critic network
        state_value = self.critic(x)
        
        return normal_dist, state_value


class Agent(object):
    def __init__(self, policy, device='cpu', actor_critic=False, baseline=0):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer_actor = torch.optim.Adam(policy.actor.parameters(), lr=1e-3)
        self.optimizer_critic = torch.optim.Adam(policy.critic.parameters(), lr=1e-3)
        self.actor_critic = actor_critic
        self.baseline = baseline

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []



    def update_policy(self):
        
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        
        # Clean buffers for next episode
        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        #
        # TASK 2:
        # REINFORCE
        if not self.actor_critic:
        #   - compute discounted returns
            returns = discount_rewards(rewards, self.gamma).detach()
        #   - compute policy gradient loss function given actions and returns
            advantages = returns - self.baseline
            loss = -(action_log_probs * advantages).mean()
        #   - compute gradients and step the optimizer
            self.optimizer_actor.zero_grad()
            loss.backward()
            self.optimizer_actor.step()

        else:
        #
        # TASK 3:
        # ACTOR-CRITIC
        #   - compute boostrapped discounted return estimates
            with torch.no_grad():
                next_state_value = self.policy.critic(next_states).squeeze(-1)

            state_value = self.policy.critic(states).squeeze(-1)
            returns = rewards + self.gamma * next_state_value * (1 - done) # if the next state is terminal next_state value is 0

        #   - compute advantage terms
            advantages = (returns - state_value).detach() # advanatges params detached since are not relevant in actor backprop
        #   - compute actor loss and critic loss
            actor_loss = -(action_log_probs * advantages).mean()
            critic_loss = F.mse_loss(returns.detach(), state_value) # returns params detached since are not relevant in actor backprop
        #   - compute gradients and step the optimizer
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()


        return        


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, _ = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob
        

    def store_outcome(self, state, next_state, action_log_prob, reward, done):

        self.states.append(torch.from_numpy(state).float())

        next_state = torch.from_numpy(next_state).float()
        self.next_states.append(next_state)

        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(float(done))



        

