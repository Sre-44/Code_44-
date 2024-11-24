from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import cv2

# Parallel environments
env = gym.make("CartPole-v1")
vec_env = make_vec_env("CartPole-v1", n_envs=1)


model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=1000)


obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")