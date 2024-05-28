import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback


env = gym.make("Ant-v4", max_episode_steps=3000, render_mode="rgb_array")
env.reset()

checkpoint_callback = CheckpointCallback(
  save_freq=5000,
  save_path="/content/PPO_TRPO_Ant_results",
  name_prefix="PPO_TRPO_Ant_results"
)

dir_path = '/content/PPO_TRPO_Ant_results'
new_logger = configure(dir_path, ["csv", "tensorboard"])

model = PPO("MlpPolicy", env, verbose=1)
model.set_logger(new_logger)

model.learn(total_timesteps=1_000_000, progress_bar=True, callback=checkpoint_callback)
# model.save("PPO(TRPO)_LunarLander")

# model = PPO.load("PPO(TRPO)_LunarLander", env=env)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)


print(mean_reward, std_reward)

