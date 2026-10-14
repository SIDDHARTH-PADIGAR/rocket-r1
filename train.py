import gymnasium as gym
from stable_baselines3 import PPO
from pathlib import Path

env = gym.make("LunarLander-v3")

model = PPO(
    "MlpPolicy",
    env, 
    verbose=1
)

model.learn(total_timesteps=500_000)
Path("models").mkdir(parents=True, exist_ok=True)
model.save("models/rocket_ppo")

env.close()