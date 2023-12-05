import os
import numpy as np
from stable_baselines3.ddpg.ddpg import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper

class DDPGAgent:
    def __init__(self, env, model_path='./generated_data/trained_models/ddpg_model'):
        self.env = NormalizedObservationWrapper(env)
        self.env = StableBaselines3Wrapper(self.env)
        self.env.reward_function.env = self.env
        self.model_path = model_path

        # Defining DDPG hyperparameters
        self.hyperparams = {
            "policy_kwargs": dict(net_arch=[64, 64]),  # Adjust the neural network architecture
            "learning_rate": 1e-3,
            "buffer_size": int(1e6),
            "batch_size": 64,
            "gamma": 0.99,
            "train_freq": 1,
            # Add more hyperparameters as needed
        }

        # Adding action noise
        action_noise = NormalActionNoise(mean=np.zeros(len(env.action_space)),
                                         sigma=0.1 * np.ones(len(env.action_space)))

        # Creating DDPG model
        self.model = DDPG("MlpPolicy", self.env, verbose=1, action_noise=action_noise, **self.hyperparams)

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
        self.model = DDPG.load(self.model_path, env=self.env)
