import time
import os
import importlib

import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box, Discrete

from register import register_trainer

def list2dictlist(model, container):
    out_dict = dict()
    for i, name in enumerate(model.config.names):
        out_dict[name] = container[i]
    return out_dict

def dictlist2list(container):
    out_list = []
    for name in container:
        out_list.append(container[name])
    return out_list


@register_trainer
def run_parallel_mappo(env, env_evaluate, model, logger, env_config, *args, **kwargs) -> None:

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

            agent_actions, agent_log_actions, state, v_n = model.step(obs_dict, evaluate=False)

            #print(f'agent_actions:{agent_actions}')
            agent_actions = list2dictlist(model, agent_actions)
            #print(f'agent_actions:{agent_actions}')

            # Take a step in the environment with the selected actions
            next_obs, rewards, dones, truncations, infos = env.step(agent_actions)

            #if ep_cycle_i/100000 == 0:
            #    print(f'------ reward at episode :{ep_i} is {rewards}')
            # Inform pre episode cycle
            obs_dict = dictlist2list(obs_dict)
            agent_actions = dictlist2list(agent_actions)
            rewards = dictlist2list(rewards)
            dones = dictlist2list(dones)
            truncations = dictlist2list(truncations)
            model.post_episode_cycle(ep_cycle_i, obs_dict, state, v_n, agent_actions, agent_log_actions, rewards, dones)

            #logger.log(model.loss_dict)
            
            # render
            if env_config.do_render:
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
                if all(dones) or all(truncations):
                    completed = True
                if completed: 
                    obs_dict, _ = env.reset()
                    break

            # ------------ Log episode score
            log_out = model.get(what='log_dict')
            if log_out is not None: 
                logger.log(log_out)

            total_steps += 1

        # Inform a new episode begin
        model.end_episode(ep_i)
    print('Fin')

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
            agent_actions = model.step(obs_dict, evaluate=True)

            #print(f'agent_actions:{agent_actions}')
            agent_actions = list2dictlist(model, agent_actions)
            #print(f'agent_actions:{agent_actions}')

            # Take a step in the environment with the selected actions
            next_obs, rewards, dones, truncations, infos = env.step(agent_actions)
            dones = dictlist2list(dones)
            truncations = dictlist2list(truncations)

            # render
            if env_config.do_render:
                env.render()

            # Update for the next iteration
            # i.e. obs_dict for the next iteration
            obs_dict = next_obs  

            for agent in env.possible_agents:
                episode_reward += rewards[agent]

            # Check if it should complete the episode because done or truncated is true in any agent
            if True:
                completed = False
                if all(dones) or all(truncations):
                    completed = True
                if completed: 
                    obs_dict, _ = env.reset()
                    break
            
        evaluate_reward += episode_reward
    # ------------ Log episode score
    evaluate_reward_dict = dict()
    evaluate_reward_dict['evaluate_reward'] = evaluate_reward
    logger.log(evaluate_reward_dict)
