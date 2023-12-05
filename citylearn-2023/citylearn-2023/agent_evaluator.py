import numpy as np
import time
import os
import matplotlib.pyplot as plt

import numpy as np


from citylearn.citylearn import CityLearnEnv

from agents.rbc_agent import BasicRBCAgent
from agents.dqn_agent import DQNAgent
from agents.ddpg_agent import DDPGAgent
from agents.ppo_agent import PPOAgent
from agents.envfactory.citylearn_env import create_citylearn_env
from agents.rewards.comfort_reward import ComfortRewardFunction
from agents.rewards.custom_reward2 import ComfortRewardFunction2

from metrics.metrics_file import write_metrics_to_file

def evaluateRBC(agent, env):
    observations = env.reset()
    actions = agent.register_reset(observations[0])
    num_steps = 0
    total_reward = 0
    rewards = []
    episode_count = 0
    while True:
        observations, reward, done, _ = env.step(actions)
        reward = reward[0]
        total_reward +=reward
        rewards.append(total_reward)
        episode_count += 1
        if not done:
            actions = agent.predict(observations)
        else:
            metrics_df = agent.env.evaluate_citylearn_challenge()
            print(f"Episode is complete | Latest env metrics: {metrics_df}", )
            break

    return episode_count, rewards, total_reward

def evaluateDQN(agent):
    actions = agent.register_reset()
    num_steps = 0
    total_reward = 0
    rewards = []
    episode_count = 0
    while True:
        observations, reward, done, _ = agent.step(actions)
        total_reward +=reward
        rewards.append(total_reward)
        episode_count += 1
        if not done:
            actions = agent.predict(observations)
            actions = [actions]
        else:
            metrics_df = agent.env.evaluate_citylearn_challenge()
            print(f"Episode is complete | Latest env metrics: {metrics_df}", )
            break

    return episode_count, rewards, total_reward

def evaluateDDPGAndPPO(agent):
    actions = agent.register_reset()
    actions = actions[0]
    num_steps = 0
    total_reward = 0
    rewards = []
    episode_count = 0
    while True:
        observations, reward, done, _ = agent.step(actions)
        total_reward +=reward
        rewards.append(total_reward)
        episode_count += 1
        if not done:
            actions = agent.predict(observations)
        else:
            metrics_df = agent.env.evaluate_citylearn_challenge()
            print(f"Episode is complete | Latest env metrics: {metrics_df}", )
            break

    return episode_count, rewards, total_reward

def pad_with_none(metrics, episode_count):
    if episode_count > len(metrics):
        metrics += [None] * (episode_count - len(metrics))

    return metrics

def plot_graph(episode_count, rbc_metrics, dqn_metrics, ddpg_metrics, ppo_metrics):
    rbc_metrics = pad_with_none(rbc_metrics, episode_count)
    dqn_metrics = pad_with_none(dqn_metrics, episode_count)
    ddpg_metrics = pad_with_none(ddpg_metrics, episode_count)
    ppo_metrics = pad_with_none(ppo_metrics, episode_count)

    episode_count_list = list(range(1, episode_count + 1))

    # Plot both sets of values on the same graph
    plt.plot(episode_count_list, rbc_metrics, marker='.', linestyle='-', color='red', label='RBC reward')
    plt.plot(episode_count_list, dqn_metrics, marker='.', linestyle='-', color='blue', label='DQN reward')
    plt.plot(episode_count_list, ddpg_metrics, marker='.', linestyle='-', color='green', label='DDPG reward')
    plt.plot(episode_count_list, ppo_metrics, marker='.', linestyle='-', color='yellow', label='PPO reward')

    plt.xlabel('Episode Count')
    plt.ylabel("cumulative reward")
    # plt.title(f'Plot of {graph_name} for {len(episode_count)} Episodes')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.savefig('allreward.png')


def save_arrays_to_file(array1, array2, array3, array4):
    np.savez("allreward.npz", array1=array1, array2=array2, array3=array3, array4=array4)

def load_arrays_from_file():
    data = np.load("allreward.npz")
    array1 = data['array1']
    array2 = data['array2']
    array3 = data['array3']
    array4 = data['array4']
    return array1, array2, array3, array4

if __name__ == '__main__':
    class Config:
        data_dir = './data/'
        SCHEMA = os.path.join(data_dir, 'schemas/warm_up/schema.json')

    config = Config()

    # Evaluating RBC
    print("Evaluating RBC")
    env, wrapper_env = create_citylearn_env(config, ComfortRewardFunction2)
    rbc_agent = BasicRBCAgent(wrapper_env)
    rbc_ep_count, rbc_rewards, rbc_tr = evaluateRBC(rbc_agent, env)

    print("Evaluating DQN")
    env, wrapper_env = create_citylearn_env(config, ComfortRewardFunction2)
    dqn_agent = DQNAgent(env)
    dqn_agent.load_model()
    dqn_ep_count, dqn_rewards, dqn_tr = evaluateDQN(dqn_agent)

    print("Evaluating DDPG")
    env, wrapper_env = create_citylearn_env(config, ComfortRewardFunction2)
    ddpg_agent = DDPGAgent(env)
    ddpg_agent.load_model()
    print("loaded i guess")
    ddpg_ep_count, ddpg_rewards, ddpg_tr = evaluateDDPGAndPPO(ddpg_agent)

    print("Evaluating PPO")
    env, wrapper_env = create_citylearn_env(config, ComfortRewardFunction2)
    ppo_agent = PPOAgent(env)
    ppo_agent.load_model()
    ppo_ep_count, ppo_rewards, ppo_tr = evaluateDDPGAndPPO(ppo_agent)

    max_ep_count = max(rbc_ep_count, dqn_ep_count, ddpg_ep_count, ppo_ep_count)
    plot_graph(max_ep_count, rbc_rewards, dqn_rewards, ddpg_rewards, ppo_rewards)

    save_arrays_to_file(rbc_rewards, dqn_rewards, ddpg_rewards, ppo_rewards)
