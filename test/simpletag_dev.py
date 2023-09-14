import argparse
import torch
import torch.nn.functional as F
import time
import os
import numpy as np
from gymnasium.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import sys
sys.path.append('./')
sys.path.append('../MARL/')

from MARL.utils.buffer import ReplayBuffer
from MARL.algorithms.maddpg_dev import MADDPG

import dataclasses
import yaml
# keep this order to avoid circular import error
from MARL.configs.method_configs import method_configs
from MARL.utils_log import writer
from rich.console import Console
CONSOLE = Console(width=120)

import pygame
from pettingzoo.mpe import simple_tag_v3

USE_CUDA = torch.cuda.is_available() # False

def run(config):
    # Logging
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) 
                         for folder in model_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)

    # Seeding
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)

    # Training
    # render_mode human (slow) rgb (faster)
    env = simple_tag_v3.parallel_env(render_mode='human', num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=config.episode_length, continuous_actions=True)
    obs_dict, _ = env.reset()  # Get the initial observations and ignore the second return value

    #possible algs: MADDPG,DDPG
    #possible adversary algs: MADDPG,DDPG  
    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)

    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [env.observation_space(agent).shape[0] for agent in env.possible_agents], # comment this line for fix size
                                 [env.action_space(agent).shape[0] if isinstance(env.action_space(agent), Box) 
                                  else env.action_space(agent).n for agent in env.possible_agents])

    #print("replay buffer observation dimensions",replay_buffer.obs_dims)

    t = 0
    with writer.TimeWriter(writer, writer.EventName.TOTAL_TRAIN_TIME): # <--- non-mandatory 
        for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
            with writer.TimeWriter(writer, writer.EventName.ITER_TRAIN_TIME, step=ep_i) as train_t: # <--- non-mandatory 
                #print("Episodes %i-%i of %i" % (ep_i + 1,
                #                                ep_i + 1 + config.n_rollout_threads,
                #                                config.n_episodes))
                obs_dict, _ = env.reset()

                maddpg.prep_rollouts(device='cpu')
                #noise w.r.t. episode percent remaining
                explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
                maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
                maddpg.reset_noise()

                # Episode length
                for et_i in range(config.episode_length):

                    # Convert the observations to torch Tensor
                    torch_obs = []
                    for agent in env.possible_agents:
                        #print(f'agent:{agent}')
                        agent_obs = [obs_dict[agent]]
                        torch_obs.append(Variable(torch.Tensor(agent_obs), requires_grad=False))

                    # Get actions from the MADDPG policy and explore if needed
                    torch_agent_actions = maddpg.step(torch_obs, explore=True)
                    # clip the action between a minimum and maximum value to prevent noisy warning messages
                    agent_actions = {agent: np.clip(ac, 0, 1) for agent, ac in zip(env.possible_agents, torch_agent_actions)}
                    # Take a step in the environment with the selected actions
                    next_obs, rewards, dones, truncations, infos = env.step(agent_actions)

                    replay_buffer.push(obs_dict, agent_actions, rewards, next_obs, dones)  # Use obs_dict here instead of obs
                    obs_dict = next_obs  # Update obs_dict for the next iteration
                    t += config.n_rollout_threads
                    if (len(replay_buffer) >= config.batch_size and
                        (t % config.steps_per_update) < config.n_rollout_threads):
                        #train critic, actor, target networks
                        if USE_CUDA:
                            maddpg.prep_training(device='gpu')
                        else:
                            maddpg.prep_training(device='cpu')
                        for u_i in range(config.n_rollout_threads):
                            for a_i in range(maddpg.nagents):
                                sample = replay_buffer.sample(config.batch_size, to_gpu=USE_CUDA)
                                maddpg.update(sample, a_i, logger=writer)
                            maddpg.update_all_targets()
                        maddpg.prep_rollouts(device='cpu')

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

                ep_rews = replay_buffer.get_average_rewards(config.episode_length * config.n_rollout_threads)
                sum_rews = sum(ep_rews)

                # update_viewer_state
                # a batch of train rays
                writer.put_scalar(name="sum_rews", scalar=sum_rews, step=ep_i)
                # write out the log
                writer.write_out_storage()

                if ep_i % config.save_interval < config.n_rollout_threads:
                    os.makedirs(run_dir / 'incremental', exist_ok=True)
                    maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
                    maddpg.save(run_dir / 'model.pt')

            # Skip the first two steps to avoid skewed timings that break the viewer rendering speed estimate.
            if ep_i > 1:
                train_num_actions_per_batch = config.episode_length # debug number for visualization
                tmp_duration = train_t.duration
                if tmp_duration == 0: tmp_duration = 0.00001
                writer.put_time(
                    name=writer.EventName.TRAIN_ITEMS_PER_SEC,
                    duration=train_num_actions_per_batch / tmp_duration,#train_t.duration,
                    step=ep_i,
                    avg_over_steps=True,
                )
        

    maddpg.save(run_dir / 'model.pt')
    env.close()
    return env, maddpg

def print_observation_space_dimensions(env):
    for agent in env.possible_agents:
        obsp = env.observation_space(agent)
        #print(f"Agent {agent}: {obsp.shape}")
def print_policy_network_dimensions(maddpg):
    for i, agent in enumerate(maddpg.agents):
        #print(f"Agent {i + 1} - Policy Network:")
        #print(f"Input Dimension: {agent.policy.fc1.in_features}")
        print(f"Output Dimension: {agent.policy.fc3.out_features}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name", help="Name of directory to store model/training contents")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=2000, type=int)
    parser.add_argument("--episode_length", default=250, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size", default=1024, type=int, help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=2000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--agent_alg", default="MADDPG", type=str, choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg", default="MADDPG", type=str, choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action", action='store_false')

    config = parser.parse_args()

    # set up writers/profilers if enabled
    # Choose a base configuration and override values.
    # This part should be modified with the hydra configuration
    config_logger = method_configs['basic']
    config_logger.max_num_iterations = config.n_episodes
    config_logger.set_timestamp()
    if config_logger.load_config:
        CONSOLE.log(f"Loading pre-set config from: {config.load_config}")
        config_logger = yaml.load(config.load_config.read_text(), Loader=yaml.Loader)
    """Logging configuration"""    
    writer_log_path = config_logger.get_base_dir()# + '/log'# + config.logging.relative_log_dir
    writer.setup_event_writer(config_logger.is_wandb_enabled(), config_logger.is_tensorboard_enabled(), entity='utokyo-marl', projectname = 'testlog', log_dir=writer_log_path)
    writer.setup_local_writer(config_logger.logging, max_iter=config_logger.max_num_iterations, banner_messages=['banner_messages'])
    writer.put_config(name="config_logger", config_dict=dataclasses.asdict(config_logger), step=0)
    # print and save config
    config_logger.print_to_terminal()
    config_logger.save_config()

    env, maddpg = run(config)
    print_observation_space_dimensions(env)
    print_policy_network_dimensions(maddpg)
