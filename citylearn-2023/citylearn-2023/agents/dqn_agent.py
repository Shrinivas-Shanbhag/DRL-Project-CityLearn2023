import os
import numpy as np
from gymnasium.spaces import Discrete
from stable_baselines3 import DQN
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper,TabularQLearningActionWrapper

class DQNAgent:
    def __init__(self, env, model_path='./generated_data/trained_models/dqn_model'):
        self.env = TabularQLearningActionWrapper(env,[{'dhw_storage': 5, 'electrical_storage': 5, 'cooling_device': 5}, {'dhw_storage': 5, 'electrical_storage': 5, 'cooling_device': 5}, {'dhw_storage': 5, 'electrical_storage': 5, 'cooling_device': 5}])
        self.env = StableBaselines3Wrapper(self.env)
        self.env.reward_function.env = self.env
        self.model_path = model_path
        self.model = DQN("MlpPolicy", self.env, verbose=1)

    def train(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)

    def register_reset(self):
        observations,_ = self.env.reset()
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
        self.model = DQN.load(self.model_path, env=self.env)
