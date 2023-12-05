This project is built on the starter kit(https://gitlab.aicrowd.com/aicrowd/challenges/citylearn-challenge/citylearn-2023-starter-kit) provided for CityLearn 2023 Challenge.

This project is developed as a part of CSCE 642 Deep Re-Inforcement Learning course offered at Texas A&M University, College Station during Fall-2023.

# Project structure:
1. agents:
    * This folder consists of implementation of DQN, DDPG, and PPO models.
    * Custom reward functions
    * envfactory to create Citylearn environment with wrapper.
2. data:
   * This folder consists of data on which all the models are trained.
3. generated_data:
   * This stores trained models, metrics while training models and graphs plotted against these data
4. metrics:
   * This folder consists of modules to calculate and store metrics and to plot graphs
5. *_runner.py: 
   * These runner code is used to train the model and save them. 
   * These also does evaluation of the trained model.
   * agent_evaluator.py: this code is to reload the trained model and run one episode for each model and plot the graph of cumulated reward over steps.
