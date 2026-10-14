
import gymnasium as gym
from reward import RotationalRewardWrapper

env = gym.make("LunarLander-v3")
env = RotationalRewardWrapper(env)

observation, info = env.reset()

print("Observation shape:", observation.shape)

for i in range(5):
    observation, reward, terminated, truncated, info = env.step(0)

    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)

env.close()
