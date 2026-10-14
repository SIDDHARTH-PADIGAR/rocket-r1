import gymnasium as gym

class RotationalRewardWrapper(gym.Wrapper):
    def __init__(self, env, rotation_weight=1.0):
        super().__init__(env)
        self.rotation_weight = rotation_weight
        self.previous_observation = None
    
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_observation = observation
        return observation, info
        
    def step(self, action):
        previous_angular_velocity = self.previous_observation[5]
        
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )
        
        current_angular_velocity = observation[5]
        
        extra_reward = self.rotation_weight * (
            abs(previous_angular_velocity) - abs(current_angular_velocity)
        )
        
        reward += extra_reward
        self.previous_observation = observation
        
        return observation, reward, terminated, truncated, info