import time
import os
import importlib

import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import pygame

def run_parallel_env(env, env_evaluate, model, logger, env_config) -> None:

    if env_evaluate is not None:
        evaluate_parallel_env(env_evaluate, model, logger, env_config)

    if env is None:
        print(f'[e] name:{__name__}')
        return
    # Need (?)
    obs_dict, _ = env.reset()  # Get the initial observations and ignore the second return value

    total_steps = 0
    for ep_i in range(0, env_config.n_episodes, env_config.n_rollout_threads):

        # Reset the environment
        obs_dict, _ = env.reset()

        # Inform a new episode begin
        model.begin_episode(ep_i)

        # Episode length
        for ep_cycle_i in range(env_config.episode_length):

            # Inform pre episode cycle
            model.pre_episode_cycle(ep_cycle_i)

            # Get the agent/agents action
            agent_actions = model.step(obs_dict)

            # Take a step in the environment with the selected actions
            next_obs, rewards, dones, truncations, infos = env.step(agent_actions)

            # Inform pre episode cycle
            model.post_episode_cycle(ep_cycle_i, obs_dict, agent_actions, rewards, next_obs, dones)

            # render
            env.render()

            # Update for the next iteration
            # i.e. obs_dict for the next iteration
            obs_dict = next_obs  

            # Evaluate
            if env_evaluate is not None:
                if total_steps % env_config.evaluate_freq == 0:
                    evaluate_parallel_env(env_evaluate, model, logger, env_config)

            # Check if it should complete the episode because done or truncated is true in any agent
            if True:
                completed = False
                for agent in env.possible_agents:
                    if dones[agent] or truncations[agent]:
                        completed = True
                if completed: 
                    obs_dict, _ = env.reset()
                    break

            # ------------ Log episode score
            logger.log(rewards)
            #logger.log(dict(something here))

        # Inform a new episode begin
        model.end_episode(ep_i)

    env.close()
    env_evaluate.close()


def evaluate_parallel_env(env, model, logger, env_config) -> None:

    if env is None:
        print(f'[e] name:{__name__}')
        return
    # Need (?)
    obs_dict, _ = env.reset()  # Get the initial observations and ignore the second return value

    evaluate_reward = 0
    for ep_i in range(0, env_config.evaluate_times, env_config.n_rollout_threads):

        # Reset the environment
        obs_dict, _ = env.reset()

        # Inform a new episode begin
        model.begin_episode(ep_i)

        # Episode length
        episode_reward = 0
        for ep_cycle_i in range(env_config.episode_length):

            # Get the agent/agents action
            agent_actions = model.step(obs_dict)

            # Take a step in the environment with the selected actions
            next_obs, rewards, dones, truncations, infos = env.step(agent_actions)

            # render
            env.render()

            # Update for the next iteration
            # i.e. obs_dict for the next iteration
            obs_dict = next_obs  

            for agent in env.possible_agents:
                episode_reward += rewards[agent]

            # Check if it should complete the episode because done or truncated is true in any agent
            if True:
                completed = False
                for agent in env.possible_agents:
                    if dones[agent] or truncations[agent]:
                        completed = True
                if completed: 
                    obs_dict, _ = env.reset()
                    break

            #logger.log(dict(something here))

        evaluate_reward += episode_reward
    # ------------ Log episode score
    logger.log(rewards)
    print(f'evaluate_reward:{evaluate_reward}')


