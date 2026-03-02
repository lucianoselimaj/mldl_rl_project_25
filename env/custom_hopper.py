"""Implementation of the Hopper environment supporting
domain randomization optimization.
    
    See more at: https://www.gymlibrary.dev/environments/mujoco/hopper/
"""
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
from Sac.adversarial_beta import AdversarialBeta
from utils.utils import to_bool

class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None, use_beta=False, curriculum_seed=42, randomize_on_reset=False):
        
        self.randomize_on_reset = to_bool(randomize_on_reset)
        self.use_beta = use_beta
        self.curriculum_seed = curriculum_seed # Seed for AdvBeta
        self.current_active_params = None
        self.cumulative_reward = 0
        self.episode_count = 0
        self.curriculum = None

        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.seed(self.curriculum_seed)
        self.domain = domain
        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses

        if domain == 'source':  # Source environment has an imprecise torso mass (-30% shift)
            self.sim.model.body_mass[1] *= 0.7
        
        if self.use_beta:
            # Beta initialization
            active_masses = self.original_masses[1:] 
            self.curriculum = AdversarialBeta(
                nominal_masses=active_masses,
                buffer_size=300,
                warmup_episodes=750,
                limit_percentage=0.3, # +/- 30%
                mix_ratio=0.5,          # 50% uniform / 50% curriculum
                tau=0.1,                # Soft update rate
                max_alpha_beta=80.0,    # Upper bound on alpha and beta values
                seed=self.curriculum_seed,        
            )
        else:
            self.curriculum = None
        
        
    def set_random_parameters(self):
        """Set random masses"""
        self.set_parameters(self.sample_parameters(ext=self.use_beta))


    def sample_parameters(self, ext=False):
        """
        Sample masses according to a Uniform Domain Randomization (UDR)
        distribution.

        - Torso mass (index 0 in this vector, body_mass[1] in Mujoco) is NOT randomized.
        - Thigh, leg, and foot are randomized at each episode.
        """
        if ext:
            # Sampling
            self.current_active_params = self.curriculum.sample_task()

            # Fix torso
            current_torso_mass = self.sim.model.body_mass[1]

            return np.concatenate(([current_torso_mass], self.current_active_params))
        else: 
            # Run UDR
            # Current masses in the simulator for this domain:
            current_masses = np.copy(self.sim.model.body_mass[1:])

            # Reference masses 
            torso_ref, thigh_ref, leg_ref, foot_ref = self.original_masses

            # Define UDR ranges 
            thigh_min, thigh_max = 0.7 * thigh_ref, 1.3 * thigh_ref
            leg_min,   leg_max = 0.7 * leg_ref,   1.3 * leg_ref
            foot_min,  foot_max = 0.7 * foot_ref,  1.3 * foot_ref

            # Sample new randomized masses 
            m_thigh = self.np_random.uniform(thigh_min, thigh_max)
            m_leg = self.np_random.uniform(leg_min,   leg_max)
            m_foot = self.np_random.uniform(foot_min,  foot_max)

            # Randomize thigh, leg, foot
            randomized_masses = np.array([
                current_masses[0],  # torso not randomized
                m_thigh,
                m_leg,
                m_foot
            ])

            return randomized_masses



    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses


    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[1:] = task


    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        # -----------------------------
        # Step customized for curriculum
        self.cumulative_reward += reward # Accumulate reward

    
        if done:
            # Map current_active_params to cumulative_reward
            if self.use_beta and (self.curriculum is not None):
                if self.current_active_params is not None:
                    self.curriculum.add_experience(self.current_active_params, self.cumulative_reward)
                    self.episode_count += 1
                    
                    # Every 30 episodes update strategy
                    if self.episode_count % 30 == 0:
                        update = self.curriculum.fit_model()
                        if update:
                            print("Adverarial Betas updated")
                            # Optional:
                            diag = self.curriculum.get_diagnostics()
                            print(f"Curriculum Update Ep {diag['ep']}: Alphas={diag['alphas']}")
            
            # Reset for next episode
            self.cumulative_reward = 0
    # -----------------------------
        return ob, reward, done, {}


    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])


    def reset_model(self):
        """Reset the environment to a random initial state"""
        self.cumulative_reward = 0
        # Security reset at each episode for curriculum
        self.current_active_params = None

        if self.randomize_on_reset:
            self.set_random_parameters()

        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


    def set_mujoco_state(self, state):
        """Set the simulator to a specific state

        Parameters:
        ----------
        state: ndarray,
               desired state
        """
        mjstate = deepcopy(self.get_mujoco_state())

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]

        self.set_sim_state(mjstate)


    def set_sim_state(self, mjstate):
        """Set internal mujoco state"""
        return self.sim.set_state(mjstate)


    def get_mujoco_state(self):
        """Returns current mjstate"""
        return self.sim.get_state()



"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

