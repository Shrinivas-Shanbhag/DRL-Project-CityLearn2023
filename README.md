This project is built on the starter kit(https://gitlab.aicrowd.com/aicrowd/challenges/citylearn-challenge/citylearn-2023-starter-kit) provided for CityLearn 2023 Challenge.

This project is developed as a part of CSCE 642 Deep Re-Inforcement Learning course offered at Texas A&M University, College Station during Fall-2023.

# Project Structure:

There are two folders:
1. citylearn-2023:
    * This folder consists of implementation of DQN, DDPG and PPO agents.
    * Runner methods are designed to train each of these agents and plot graphs based on the environment metrics.
    * More on this folder is explained in README.md file within citylearn-2023 folder.
2. dependencies:
   * This folder consists of modified dependent libraries, which are patched by us to resolve few dependency conflicts.
   * It has below dependencies:
     * CityLearn-master:2.1.0: 
       * This folder is downloaded from https://github.com/intelligent-environments-lab/CityLearn/releases/tag/v2.1.0
       * Some modifications are made to this folder, to solve the dependency conflicts
     * gym-master:0.26.2:
       * This folder is downloaded from https://github.com/openai/gym/releases/tag/0.26.2
       * Some modifications are made to this folder, to solve the dependency conflicts
     * stable-baselines3-master:2.2.1:
       * This folder is downloaded from https://github.com/DLR-RM/stable-baselines3/releases/tag/v2.2.1
       * Some modifications are made to this folder, to solve the dependency conflicts

# How to prepare the environment:
1. Create python environment with python=3.8
2. Run all the dependencied mentioned in ./citylearn-2023/requirements.txt using below command
   * pip install -r ./citylearn-2023/requirements.txt
3. Install dependencies from dependencies folder:
   * Go to ./dependencies/CityLearn-master/CityLearn-master and run below command to install it
     * pip install -e .
   * Go to ./dependencies/gym-master/gym-master and run below command to install it
     * pip install -e .
   * Go to ./dependencies/stable-baselines3-master/stable-baselines3-master and run below command to install it
     * pip install -e .
