import time
import os
import importlib

from omegaconf import OmegaConf
import numpy as np

from MARL.utils.dictionary import AttrDict
from MARL.algorithms.qmix.replay_buffer import ReplayBuffer
from MARL.algorithms.qmix.qmix_smac import QMIX_SMAC
from MARL.algorithms.qmix.normalization import Normalization

from register import register_trainer

@register_trainer
def run_parallel_smacv2(env, env_evaluate, model, logger, env_config, *args, **kwargs) -> None:

    if env is None:
        print(f'[e] name:{__name__}')
        return
    
    env_info = env.get_env_info()
    episode_limit = env_info["episode_limit"]  # Maximum number of steps per episode

    total_steps = 0
    evaluate_num = -1  # Record the number of evaluations
    max_train_steps = int(env_config.max_train_steps / episode_limit)
    for ep_i in range(max_train_steps):

        # Evaluate
        if env_evaluate is not None:
            if total_steps // env_config.evaluate_freq > evaluate_num:
                evaluate_parallel_env(env_evaluate, model, logger, env_config, total_steps)
                evaluate_num += 1

        # Reset the environment
        env.reset()

        # Inform a new episode begin
        model.begin_episode(ep_i)

        # Episode length
        for ep_cycle_i in range(episode_limit):

            # Inform pre episode cycle
            model.pre_episode_cycle(ep_cycle_i)

            obs_n = env.get_obs()  # obs_n.shape=(N,obs_dim)
            s = env.get_state()  # s.shape=(state_dim,)
            avail_a_n = env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)

            # Get the agent/agents action
            agent_actions = model.step(obs_n, avail_a_n, evaluate=False)

            # Take a step in the environment with the selected actions
            rewards, dones, infos = env.step(agent_actions)

            # Inform pre episode cycle
            model.post_episode_cycle(ep_cycle_i, obs_n, s, avail_a_n, agent_actions, rewards, dones, infos)
            
            # render
            if env_config.do_render:
                env.render()

            # Check if it should complete the episode because done or truncated is true in any agent
            if dones:
                break

            # ------------ Log episode score
            log_out = model.get(what='log_dict_episode_cycle')
            if log_out is not None: 
                logger.log(log_out)

            total_steps += 1

        # Inform a new episode begin
        # An episode is over, store obs_n, s and avail_a_n in the last step
        obs_n = env.get_obs()
        s = env.get_state()
        avail_a_n = env.get_avail_actions()
        model.end_episode(ep_cycle_i + 1, obs_n, s, avail_a_n)

        # ------------ Log episode score
        log_out = model.get(what='log_dict_episode')
        if log_out is not None: 
            logger.log(log_out)

def evaluate_parallel_env(env, model, logger, env_config, total_steps) -> None:

    if env is None:
        print(f'[e] name:{__name__}')
        return
    
    env_info = env.get_env_info()
    episode_limit = env_info["episode_limit"]  # Maximum number of steps per episode

    win_times = 0
    evaluate_reward = 0
    for ep_i in range(env_config.evaluate_times):

        # Reset the environment
        env.reset()

        # Inform a new episode begin
        model.begin_episode()

        # Episode length
        win_tag = False
        episode_reward = 0
        for ep_cycle_i in range(episode_limit):

            obs_n = env.get_obs()  # obs_n.shape=(N,obs_dim)
            avail_a_n = env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)

            # Get the agent/agents action
            agent_actions = model.step(obs_n, avail_a_n, evaluate=True)

            # Take a step in the environment with the selected actions
            rewards, dones, infos = env.step(agent_actions)

            win_tag = True if dones and 'battle_won' in infos and infos['battle_won'] else False

            # render
            if env_config.do_render:
                env.render()

            episode_reward += rewards

            # Check if it should complete the episode because done or truncated is true in any agent
            if dones:
                break

        if win_tag:
            win_times += 1
        evaluate_reward += episode_reward

    win_rate = win_times / env_config.evaluate_times
    evaluate_reward = evaluate_reward / env_config.evaluate_times
    print("total_steps:{} \t win_rate:{} \t win_times:{} \t et:{} \t evaluate_reward:{}".format(total_steps, win_rate, win_times, env_config.evaluate_times, evaluate_reward))

    # ------------ Log episode score
    evaluate_reward_dict = dict()
    evaluate_reward_dict['evaluate_reward'] = evaluate_reward
    evaluate_reward_dict['win_rate'] = win_rate
    logger.log(evaluate_reward_dict)

