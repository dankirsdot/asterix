# Reinforcement Learning Agents for Astérix Gameplay on Sega Master System

This GitHub project, developed as part of the CS394R/ECE381V "Reinforcement Learning: Theory and Practice" course, presents our endeavor to train reinforcement learning agents for playing Astérix, a classic game on the Sega Master System platform. Our primary objective was to complete the first level of the game, leveraging two prominent reinforcement learning algorithms: Deep Q-Network (DQN) and Proximal Policy Optimization (PPO). Through this project, we aimed to compare the performance of these algorithms and evaluate their effectiveness in navigating the diverse and challenging environments presented in the game. The GitHub repository contains the codebase, including implementations of DQN and PPO, as well as the augmented Astérix environment we have designed.

In order to run the code, install all the dependencies:

``` bash
python -m pip install gymnasium pyglet==1.5.28
python -m pip install git+https://github.com/Farama-Foundation/stable-retro.git
python -m pip install stable_baselines3 tensorboard opencv-python
```

After that, you can run the training process with the following command:

```bash
python dqn.py
```

or

```bash
python ppo.py
```

The command line arguments *n_envs* and *checkpoint* could be passed to the Python files above to change the number of parallel environments set up and start from a preexisting checkpoint.
