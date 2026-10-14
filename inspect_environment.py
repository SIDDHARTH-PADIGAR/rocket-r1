import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset()

print(observation)
print(observation.shape)
print(env.action_space)

for i in range(5):
    observation, reward, terminated, truncated, info = env.step(0)
    print("observation:", observation)
    print("reward:", reward)
    print("terminated:", terminated)
    print("truncated:", truncated)
    

env.close()