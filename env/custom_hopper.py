"""Implementation of the Hopper environment supporting
domain randomization optimization.
    
    See more at: https://www.gymlibrary.dev/environments/mujoco/hopper/
"""
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv


class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.domain = domain

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses

        if domain == 'source':  # Source environment has an imprecise torso mass (-30% shift)
            self.sim.model.body_mass[1] *= 0.7

    def set_random_parameters(self):
        """Set random masses"""
        self.set_parameters(self.sample_parameters())


    def sample_parameters(self):
        """
        Sample masses according to a Uniform Domain Randomization (UDR)
        distribution.

        - Torso mass (index 0 in this vector, body_mass[1] in Mujoco) is NOT randomized.
        - Thigh, leg, and foot are randomized at each episode.
        """

        # Current masses in the simulator for this domain:
        # [torso, thigh, leg, foot]
        current_masses = np.copy(self.sim.model.body_mass[1:])

        # Reference masses from hopper.xml: [torso_ref, thigh_ref, leg_ref, foot_ref]
        torso_ref, thigh_ref, leg_ref, foot_ref = self.original_masses

        # --- Define UDR ranges (hyperparameters) ---
        # Here we choose ±30% around the reference masses.
        # You can adjust these factors later if needed.
        thigh_min, thigh_max = 0.7 * thigh_ref, 1.3 * thigh_ref
        leg_min,   leg_max = 0.7 * leg_ref,   1.3 * leg_ref
        foot_min,  foot_max = 0.7 * foot_ref,  1.3 * foot_ref

        # --- Sample new randomized masses ---
        m_thigh = np.random.uniform(thigh_min, thigh_max)
        m_leg = np.random.uniform(leg_min,   leg_max)
        m_foot = np.random.uniform(foot_min,  foot_max)

        # - Keep the CURRENT torso mass fixed (already scaled in 'source' if needed).
        # - Randomize only thigh, leg, foot.
        randomized_masses = np.array([
            current_masses[0],  # torso: do NOT randomize
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

        return ob, reward, done, {}


    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])


    def reset_model(self):
        """Reset the environment to a random initial state"""

        # --- FOR PHASE 1: COMMENT THIS OUT ---
        # if self.domain == 'source':
        #     self.set_random_parameters()
        # -------------------------------------

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

