import mujoco.gl_context
import mujoco.viewer
import numpy as np
from gymnasium import utils
import mujoco
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco import mujoco_rendering
from gymnasium.spaces import Box
from noise import pnoise2, snoise2
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
    def __init__(self, episode_len=1000, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        # change shape of observation to your observation space size
        observation_space = Box(low=-np.inf, high=np.inf, shape=(49, ), dtype=np.float64)
        # load your MJCF model with env and choose frames count between actions
        MujocoEnv.__init__(
            self,
            os.path.abspath("assets/ramp.xml"),
            5,
            observation_space=observation_space,
            **kwargs
        )
        self.step_number = 0
        self.episode_len = episode_len

    def control_cost(self, action):
        control_cost = 0.1 * np.sum(np.square(self.data.ctrl))
        return control_cost
    
    def rot_change_rew(self, past, cur):
        reward = 0
        bonus = 0.1
        for i in range(len(past)):
            if past[i] * cur[i] >= 0:
                reward = reward + bonus
            else:
                reward = reward - bonus
        return reward 

    # determine the reward depending on observation or other properties of the simulation
    def step(self, a):
        past_obs = self._get_obs()
        reward = 0
        past_x = past_obs[0]
        past_rot_vel = np.concatenate((
        np.array(self.data.joint("left_foot_joint").qvel),
        np.array(self.data.joint("right_foot_joint").qvel),
        np.array(self.data.joint("left_knee_joint").qvel),
        np.array(self.data.joint("right_knee_joint").qvel),
        np.array(self.data.joint("left_hip_joint").qvel),
        np.array(self.data.joint("right_hip_joint").qvel)), axis=0)
        

        self.do_simulation(a, self.frame_skip)
        self.step_number += 1
        ctrl_cost = 0.025 * self.control_cost(a)

        obs = self._get_obs()
        done = bool(not np.isfinite(obs).all())
        leftfootID = 10
        rightfootID = 11
        leftfootID2 = 15
        rightfootID2 = 16
        leftfootID3 = 9
        rightfootID3 = 14
        for contact in self.data.contact:
            # print("Contact 1:")
            # print(contact.geom1)
            # print("Contact 2:")
            # print(contact.geom2)
            if (contact.geom1 != leftfootID) and (contact.geom2 != leftfootID):
                if (contact.geom1 != rightfootID) and (contact.geom2 != rightfootID):
                    if (contact.geom1 != leftfootID2) and (contact.geom2 != leftfootID2):
                        if (contact.geom1 != rightfootID2) and (contact.geom2 != rightfootID2):
                            if (contact.geom1 != leftfootID3) and (contact.geom2 != leftfootID3):
                                if (contact.geom1 != rightfootID3) and (contact.geom2 != rightfootID3):
                                    done = True


        cur_rot_vel = np.concatenate((
        np.array(self.data.joint("left_foot_joint").qvel),
        np.array(self.data.joint("right_foot_joint").qvel),
        np.array(self.data.joint("left_knee_joint").qvel),
        np.array(self.data.joint("right_knee_joint").qvel),
        np.array(self.data.joint("left_hip_joint").qvel),
        np.array(self.data.joint("right_hip_joint").qvel)), axis=0)
        reward = reward + self.rot_change_rew(past_rot_vel, cur_rot_vel)
        # print(self.data.joint("root_joint").qpos[2] * 0.5)
                            
        truncated = self.step_number > self.episode_len
        reward = reward + (past_x - obs[0]) * 200
        reward = reward + self.data.joint("root_joint").qpos[2] * 0.75
        reward = reward - ctrl_cost
        return obs, reward, done, truncated, {}
    

     

    def gen_hfield_perlin(self, num_rows, num_cols, octaves):
        freq = 16.0 * octaves
        hfield = np.zeros((num_rows, num_cols))
        for y in range(num_rows):
            for x in range(num_cols):
                hfield[x, y] = int(snoise2(x / freq, y / freq, octaves) * 127.0 + 128.0)
        hfield = (hfield-np.min(hfield))/(np.max(hfield)-np.min(hfield))
        return hfield

    # define what should happen when the model is reset (at the beginning of each episode)
    def reset_model(self):
        
        self.step_number = 0

        # for example, noise is added to positions and velocities
        qpos = self.init_qpos 
        # + self.np_random.uniform(
        #     size=self.model.nq, low=-0.01, high=0.01
        # )
        qvel = self.init_qvel
        # + self.np_random.uniform(
        #     size=self.model.nv, low=-0.01, high=0.01
        # )
        self.set_state(qpos, qvel)
        num_rows = self.model.hfield_nrow[0]
        num_cols = self.model.hfield_ncol[0]
        new_hfield = self.gen_hfield_perlin(num_rows, num_cols, 10)
        self.model.hfield_data = new_hfield.flatten()
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
        base_orient = np.array(self.data.joint("root_joint").qpos[-4:])

        obs = np.concatenate((np.array(self.data.joint("root_joint").qpos),
                              np.array(self.data.joint("root_joint").qvel),
                              np.array(self.data.qfrc_actuator),
                              base_orient - np.array(self.data.joint("left_foot_joint").qpos[-4:]),
                            #   np.array(self.data.joint("left_foot_joint").qvel),
                              base_orient - np.array(self.data.joint("right_foot_joint").qpos[-4:]),
                            #   np.array(self.data.joint("right_foot_joint").qvel),
                              base_orient - np.array(self.data.joint("left_knee_joint").qpos[-4:]),
                            #   np.array(self.data.joint("left_knee_joint").qvel),
                              base_orient - np.array(self.data.joint("right_knee_joint").qpos[-4:]),
                            #   np.array(self.data.joint("right_knee_joint").qvel),
                              base_orient - np.array(self.data.joint("left_hip_joint").qpos[-4:]),
                            #   np.array(self.data.joint("left_hip_joint").qvel),
                              base_orient - np.array(self.data.joint("right_hip_joint").qpos[-4:]),
                            #   np.array(self.data.joint("right_hip_joint").qvel)
                              ), axis=0)
        return obs
