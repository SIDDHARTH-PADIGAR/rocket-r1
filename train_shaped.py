
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from reward import RotationalRewardWrapper

env = gym.make("LunarLander-v3")
env = RotationalRewardWrapper(env)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
)

model.learn(total_timesteps=500_000)

Path("models").mkdir(parents=True, exist_ok=True)
model.save("models/rocket_ppo_shaped")

env.close()
