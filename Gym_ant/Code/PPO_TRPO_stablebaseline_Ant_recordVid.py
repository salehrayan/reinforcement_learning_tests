import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv


env = gym.make("Ant-v4", max_episode_steps=1000, render_mode="rgb_array")


video_folder = "./PPO_TRPO_stablebaseline_results/"
video_length = 1000

vec_env = DummyVecEnv([lambda: env])
vec_env.reset()

vec_env = VecVideoRecorder(vec_env, video_folder,
                       record_video_trigger=lambda x: True, video_length=video_length,
                       name_prefix=f"PPO_TRPO_stablebaseline_Ant_gym")
# model = A2C("MlpPolicy", env, verbose)
# model.set_logger(new_logger)
#
# model.learn(total_timesteps=10_000, progress_bar=True)
# model.save("A2C_LunarLander")

model = PPO.load("PPO_TRPO_Ant_results'_540000_steps.zip", env=vec_env)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=3)


print(mean_reward, std_reward)

