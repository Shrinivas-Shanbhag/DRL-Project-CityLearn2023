import numpy as np
import time
import os

from citylearn.citylearn import CityLearnEnv

from agents.ddpg_agent import DDPGAgent
from agents.envfactory.citylearn_env import create_citylearn_env
from agents.rewards.custom_reward2 import ComfortRewardFunction2

from metrics.metrics_file import write_metrics_to_file


def train_agent(config, agent):
    episode_metrics = []
    for i in range(config.num_episodes):
        print("Episode: "+str(i))
        agent.train(config.episode_size)
        metrics_df = agent.env.evaluate_citylearn_challenge()
        episode_metrics.append(metrics_df)
        print(metrics_df)

    write_metrics_to_file(episode_metrics, config.output_file)


def evaluate(agent):
    agent_time_elapsed = 0
    step_start = time.perf_counter()
    actions = agent.register_reset()
    actions = actions[0]
    agent_time_elapsed += time.perf_counter() - step_start

    num_steps = 0
    interrupted = False
    try:
        while True:
            observations, _, done, _ = agent.step(actions)
            if not done:
                step_start = time.perf_counter()
                actions = agent.predict(observations)
                agent_time_elapsed += time.perf_counter()- step_start
            else:
                metrics_df = agent.env.evaluate_citylearn_challenge()
                print(f"Episode is complete | Latest env metrics: {metrics_df}", )

                step_start = time.perf_counter()
                agent_time_elapsed += time.perf_counter()- step_start
                break
            
            num_steps += 1

    except KeyboardInterrupt:
        print("========================= Stopping Evaluation =========================")
        interrupted = True
    
    if not interrupted:
        print("=========================Completed=========================")

    print(f"Total time taken by agent: {agent_time_elapsed}s")


def create_agent(config):
    env, wrapper_env = create_citylearn_env(config, ComfortRewardFunction2)
    return DDPGAgent(env)


if __name__ == '__main__':
    class Config:
        data_dir = './data/'
        SCHEMA = os.path.join(data_dir, 'schemas/warm_up/schema.json')
        num_episodes = 100
        episode_size = 1000
        output_file = "generated_data/train_metrics/ddpg_agent_metrics.csv"
    
    config = Config()

    config = Config()
    agent = create_agent(config)
    train_agent(config, agent)
    evaluate(agent)
    agent.save_model()
