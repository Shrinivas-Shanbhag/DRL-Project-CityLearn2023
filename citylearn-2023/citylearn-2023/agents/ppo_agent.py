import os
import numpy as np
from stable_baselines3 import PPO
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper

class PPOAgent:
    def __init__(self, env, model_path='./generated_data/trained_models/ppo_model'):
        self.env = NormalizedObservationWrapper(env)
        self.env = StableBaselines3Wrapper(self.env)
        self.env.reward_function.env = self.env
        self.model_path = model_path
        self.model = PPO("MlpPolicy", self.env, verbose=1)

    def train(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)

    def register_reset(self):
        observations, _ = self.env.reset()
        return self.predict(observations)

    def predict(self, observations):
        action, _ = self.model.predict(observations, deterministic=True)
        return action

    def step(self, actions):
        return self.env.step(actions)

    def save_model(self):
        # Save the model to the specified path
        self.model.save(self.model_path)

    def load_model(self):
        # Load the model from the specified path
        self.model = PPO.load(self.model_path, env=self.env)
