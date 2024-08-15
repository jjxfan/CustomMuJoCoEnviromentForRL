import os
import numpy as np
from stable_baselines3.common.env_checker import check_env
from rigid_walk import BallBalanceEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
import torch as th
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
#th.autograd.set_detect_anomaly(True


class RenderCallback(BaseCallback):
    """
    Custom callback for rendering the environment during training.
    """
    def __init__(self, render_freq: int, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        # Render the environment every `render_freq` steps
        if self.n_calls % self.render_freq == 0:
            self.training_env.render()
        return True

log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)


env_id = 'bipedal-v0'  # Replace with your specific environment ID
n_envs = 16

gym.envs.registration.register(
    id=env_id,
    entry_point=BallBalanceEnv,
    max_episode_steps=1000, # Customize to your needs.
    reward_threshold=10000 # Customize to your needs.
)

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            all_rewards = []

            # Iterate over each monitor file in the log_dir
            for monitor_file in os.listdir(self.log_dir):
                if monitor_file.endswith(".monitor.csv"):
                    # monitor_path = os.path.join(self.log_dir, monitor_file)
                    try:
                        print("search")
                        x, y = ts2xy(load_results(self.log_dir), 'timesteps')
                        all_rewards.extend(y)
                    except ValueError:
                        # Handle cases where the file is empty or invalid
                        continue

            if len(all_rewards) > 0:
                # Calculate the mean reward over the last 100 episodes
                mean_reward = np.mean(all_rewards[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # Save the new best model if the mean reward is higher
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True

# vectorizing env
vec_env = make_vec_env(env_id, n_envs=n_envs, monitor_dir=log_dir)

# initialize your enviroment

# env = BallBalanceEnv(render_mode="rgb_array")
# env = Monitor(env, log_dir)

# env = VecCheckNan(env, raise_exception=True)
# it will check your custom environment and output additional warnings if needed

# check_env(env)

# device check

# device = th.device('cuda' if th.cuda.is_available() else 'cpu')
device = th.device('cpu')




# learning with tensorboard logging and saving model
model = SAC("MlpPolicy", vec_env, verbose=1, tensorboard_log="./sac_ball_balance_tensorboard/", device=device)

# Uncomment for multiGPU
# if th.cuda.device_count() > 1:
#     model.policy = torch.nn.DataParallel(model.policy)

# # Move the model to GPU
# model.policy.to('cuda')



render_freq = 1
render_callback = RenderCallback(render_freq=render_freq)
best_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
callback = CallbackList([best_callback])
model.learn(total_timesteps=200000000, log_interval=4, callback=callback)
model.save("sac_ball_balance")

