import mujoco.glfw
from stable_baselines3 import SAC
from rigid_walk import BallBalanceEnv
import mujoco
import cv2
import time
import imageio

env = BallBalanceEnv(render_mode="human")
model = SAC.load("BestModel/walking_7.16.2024.zip")


for _ in range(1000):

    obs = # Enter in realtime data

    action, _states = model.predict(obs, deterministic=False)

    # Apply action

    if done:
        break