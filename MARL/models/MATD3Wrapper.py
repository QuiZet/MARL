import copy
import numpy as np

import torch
from torch.autograd import Variable

from gymnasium.spaces import Box

from MARL.utils.dictionary import AttrDict

from MARL.algorithms.matd3.matd3_gymnasium import MATD3
from MARL.algorithms.maddpg.replay_buffer_gymnasium import ReplayBufferGymnasium as ReplayBuffer
from MARL.models.abstractwrapper import AbstractWrapper

class MATD3Wrapper(AbstractWrapper):
    def __init__(self, *args, **kwargs):
        print(f"args:{args}")
        print(f'kwargs:{kwargs}')

        self.config = AttrDict(kwargs)
        print(f'self.config:{self.config}')

        self.env = args[0]
        self.device = args[1]
        print(f'device:{self.device}')

        self.config.names = self.env.possible_agents
        self.config.N = self.env.max_num_agents  # The number of agents
        self.config.obs_dim_n = dict()
        self.config.action_dim_n = dict()
        for agent in self.env.possible_agents:
            self.config.obs_dim_n[agent] = self.env.observation_space(agent).shape[0]  # obs dimensions of N agents
            self.config.action_dim_n[agent] = self.env.action_space(agent).shape[0]  # actions dimensions of N agents
        print(f'self.args.N:{self.config.N}')
        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.config.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.config.action_dim_n))

        # Create N agents
        self.agent_n = dict() 
        for agent_id in self.env.possible_agents:
            print("Algorithm: MATD3")
            self.agent_n[agent_id] = MATD3(self.config, agent_id)

        print(f'self.agent_n:{self.agent_n}')

        self.replay_buffer = ReplayBuffer(self.config)

        self.noise_std = self.config.noise_std_init  # Initialize noise_std
        self.noise_std_decay = (self.config.noise_std_init - self.config.noise_std_min) / self.config.noise_decay_steps

        # output log dictionary
        self.log_dict_out = None

    def step(self, obs_dict, *args, **kwargs):
        # Convert the observations to torch Tensor
        torch_obs = []
        for agent in self.env.possible_agents:
            agent_obs = [obs_dict[agent]]
            torch_obs.append(Variable(torch.Tensor(agent_obs), requires_grad=False))

        # Each agent selects actions based on its own local observations(add noise for exploration)
        agent_actions = dict()
        for agent, obs in zip(self.agent_n, obs_dict):
            agent_actions[agent] = self.agent_n[agent].choose_action(obs_dict[obs], noise_std=self.noise_std)
        return agent_actions
    
    # Add env_evaluation in the main trainable environemnt and the option for the evaluation

    def save(self, fname):
        pass

    def begin_episode(self, ep_i, *args, **kwargs):
        pass

    def end_episode(self, *args, **kwargs):
        pass

    def pre_episode_cycle(self, *args, **kwargs):
        pass

    def post_episode_cycle(self, *args, **kwargs):      #equivalent to def run
        ep_cycle_i, obs_dict, agent_actions, rewards, next_obs, dones = args
        self.log_dict_out = dict(rewards)

        # Store the transition
        self.replay_buffer.store_transition(obs_dict, agent_actions, rewards, next_obs, dones)
        # Decay noise_std
        if self.config.use_noise_decay:
           self.noise_std = self.noise_std - self.noise_std_decay if self.noise_std - self.noise_std_decay > self.config.noise_std_min else self.config.noise_std_min

        if self.replay_buffer.current_size > self.config.batch_size:
            # Train each agent individually
            for agent_id in self.env.possible_agents:
                # Train the agent
                loss_out = self.agent_n[agent_id].train(self.replay_buffer, self.agent_n)
                if self.config.log_loss:
                    for l in loss_out:
                        self.log_dict_out[l + '_' + agent_id] = loss_out[l]

    def get(self, *args, **kwargs):
        what = kwargs['what']
        if what == 'log_dict':
            return self.log_dict_out
        return None
