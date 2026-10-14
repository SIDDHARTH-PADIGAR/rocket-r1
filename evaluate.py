import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3", render_mode="human")
model = PPO.load("models/rocket_ppo", env=env)

episodes = 20
successful_landings = 0
total_rewards = []

for episode in range(episodes):
    observation, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action, _ = model.predict(observation, deterministic=True)
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    
    total_rewards.append(total_reward)

    print(
        f"Episode {episode + 1}: "
        f"Reward = {total_reward:.2f}, "
        f"Terminated = {terminated}, "
        f"Truncated = {truncated}"
    )
     

env.close()

print(f"\nMean reward: {sum(total_rewards) / episodes:.2f}")
print(f"Episodes evaluated: {episodes}")
