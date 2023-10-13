import time
import os
import importlib

import numpy as np

import torch
from torch.autograd import Variable

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import pygame
#import pettingzoo.mpe
#from pettingzoo.mpe import simple_tag_v3

USE_CUDA = torch.cuda.is_available() # False

def make_env(env_config):
    try:
        i = importlib.import_module(env_config.name)
        #env = i.parallel_env(render_mode=env_config.render_mode, 
        #                     num_good=env_config.num_good, num_adversaries=env_config.num_adversaries, 
        #                     num_obstacles=env_config.num_obstacles, max_cycles=env_config.episode_length, 
        #                     continuous_actions=env_config.continuous_actions)
        env = i.parallel_env(**env_config.hyperparams)
        return env
    except Exception as e:
        print(f'[ex] make_env:{e}')
    return None

def run_parallel_env(env_config, logger):

    # List of all the environments
    #print(gym.envs.registry.keys())

    # Training
    # render_mode human (slow) rgb (faster)
    env = make_env(env_config=env_config)
    if env is None:
        print(f'[e] name:{__name__}')
        return
    # Need (?)
    obs_dict, _ = env.reset()  # Get the initial observations and ignore the second return value

    for ep_i in range(0, env_config.n_episodes, env_config.n_rollout_threads):

        # Reset the environment
        obs_dict, _ = env.reset()

        # Episode length
        for et_i in range(env_config.episode_length):

            # Convert the observations to torch Tensor
            torch_obs = []
            for agent in env.possible_agents:
                #print(f'agent:{agent}')
                agent_obs = [obs_dict[agent]]
                torch_obs.append(Variable(torch.Tensor(agent_obs), requires_grad=False))

            # Get actions from the MARL policy and explore if needed
            #torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # clip the action between a minimum and maximum value to prevent noisy warning messages
            #agent_actions = {agent: np.clip(ac, 0, 1) for agent, ac in zip(env.possible_agents, torch_agent_actions)}

            # dummy action (if no MARL policy)
            agent_actions = dict()
            for agent in env.possible_agents:
                action = env.action_space(agent).sample()  # this is where you would insert your policy
                agent_actions[agent] = action

            # Take a step in the environment with the selected actions
            next_obs, rewards, dones, truncations, infos = env.step(agent_actions)
            obs_dict = next_obs  # Update obs_dict for the next iteration

            # render
            env.render()

            # Check if it should complete the episode because done or truncated is true in any agent
            if True:
                completed = False
                for agent in env.possible_agents:
                    if dones[agent] or truncations[agent]:
                        completed = True
                if completed: 
                    obs_dict, _ = env.reset()
                    break

            #sum_rews = sum(rewards)

            # ------------ Log episode score
            logger.log(rewards)
            #logger.log(dict(something here))

    env.close()

