import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np


EPISODES = 20
SEEDS = list(range(100, 100 + EPISODES))

MODELS = {
    "Baseline": "models/rocket_ppo",
    "Rotation-shaped": "models/rocket_ppo_shaped",
}


def evaluate_model(model_path, seeds):
    env = gym.make("LunarLander-v3")
    model = PPO.load(model_path)

    rewards = []
    lengths = []
    successes = 0

    for seed in seeds:
        observation, info = env.reset(seed=seed)
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        lengths.append(steps)

        # Lunar Lander gives +100 for a successful landing.
        # Use the original environment reward as a practical
        # signal, not as a perfect success classifier.
        if terminated and total_reward >= 100:
            successes += 1

    env.close()

    return {
        "mean_reward": np.mean(rewards),
        "mean_length": np.mean(lengths),
        "successes": successes,
        "success_rate": successes / len(seeds) * 100,
    }


for name, path in MODELS.items():
    results = evaluate_model(path, SEEDS)

    print(f"\n{name}")
    print(f"Mean reward: {results['mean_reward']:.2f}")
    print(f"Mean episode length: {results['mean_length']:.1f}")
    print(
        f"Successful episodes: {results['successes']}/{EPISODES}"
    )
    print(f"Success rate: {results['success_rate']:.1f}%")
