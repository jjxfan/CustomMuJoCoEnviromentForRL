import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os


# you can completely modify this class for your MuJoCo environment by following the directions
class BallBalanceEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    # set default episode_len for truncate episodes
    def __init__(self, episode_len=4000, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        # change shape of observation to your observation space size
        observation_space = Box(low=-np.inf, high=np.inf, shape=(18 ,), dtype=np.float64)
        # load your MJCF model with env and choose frames count between actions
        MujocoEnv.__init__(
            self,
            os.path.abspath("assets/mjmodel.xml"),
            5,
            observation_space=observation_space,
            **kwargs
        )
        self.step_number = 0
        self.episode_len = episode_len

    def control_cost(self, action):
        control_cost = 0.1 * np.sum(np.square(self.data.ctrl))
        return control_cost

    # determine the reward depending on observation or other properties of the simulation
    def step(self, a):
        reward = 0.4
        self.do_simulation(a, self.frame_skip)
        self.step_number += 1
        ctrl_cost = 0.1 * self.control_cost(a)

        obs = self._get_obs()
        done = bool(not np.isfinite(obs).all() or (obs[2] < 0))
        truncated = self.step_number > self.episode_len
        if obs[3] < -0.10:
            reward = reward + 1
        if obs[3] > 0:
            reward = reward - 0.5
        reward = reward - ctrl_cost
        return obs, reward - ctrl_cost, done, truncated, {}

    # define what should happen when the model is reset (at the beginning of each episode)
    def reset_model(self):
        self.step_number = 0

        # for example, noise is added to positions and velocities
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    # determine what should be added to the observation
    # for example, the velocities and positions of various joints can be obtained through their names, as stated here
    def _get_obs(self):

        # obs = np.concatenate((np.array(self.data.joint("left_foot").qpos[:3]),
        #                       np.array(self.data.joint("left_foot").qvel[:3]),
        #                       np.array(self.data.joint("right_foot").qpos),
        #                       np.array(self.data.joint("right_foot").qvel),
        #                       np.array(self.data.joint("left_knee").qpos),
        #                       np.array(self.data.joint("left_knee").qvel)), axis=0)
        obs = np.concatenate((np.array(self.data.joint("root_joint").qpos[:3]),
                              np.array(self.data.joint("root_joint").qvel[:3]),
                              np.array(self.data.joint("left_foot_joint").qpos),
                              np.array(self.data.joint("left_foot_joint").qvel),
                              np.array(self.data.joint("right_foot_joint").qpos),
                              np.array(self.data.joint("right_foot_joint").qvel),
                              np.array(self.data.joint("left_knee_joint").qpos),
                              np.array(self.data.joint("left_knee_joint").qvel),
                              np.array(self.data.joint("right_knee_joint").qpos),
                              np.array(self.data.joint("right_knee_joint").qvel),
                              np.array(self.data.joint("left_hip_joint").qpos),
                              np.array(self.data.joint("left_hip_joint").qvel),
                              np.array(self.data.joint("right_hip_joint").qpos),
                              np.array(self.data.joint("right_hip_joint").qvel)), axis=0)
        return obs
