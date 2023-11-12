import copy
import numpy as np

import torch
from torch.autograd import Variable

from gymnasium.spaces import Box, Discrete

from MARL.utils.dictionary import AttrDict

from MARL.algorithms.mappo.mappo_gymnasium import MAPPO
from MARL.algorithms.mappo.replay_buffer_gymnasium import  ReplayBuffer
from MARL.models.abstractwrapper import AbstractWrapper

class MAPPOWrapper(AbstractWrapper):
    def __init__(self, *args, **kwargs):
        try:
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
            self.config.max_obs_dim = int()
            for agent in self.env.possible_agents:
                self.config.obs_dim_n[agent] = self.env.observation_space(agent).shape[0]  # obs dimensions of N agents
                if type(self.env.action_space(agent)) == Discrete:
                    self.config.action_dim_n[agent] = self.env.action_space(agent).n  # actions dimensions of N agents
                else:
                    self.config.action_dim_n[agent] = self.env.action_space(agent).shape[0]  # actions dimensions of N agents
            for agent in self.env.possible_agents:
                self.config.state_dim = np.sum(self.config.obs_dim_n[names] for names in self.config.names)
            for agent in self.env.possible_agents:
                self.config.max_obs_dim = self.config.obs_dim_n[agent] if self.config.obs_dim_n[agent] > self.config.max_obs_dim else self.config.max_obs_dim
            print(f'self.args.N:{self.config.N}')
            print("obs_dim_n={}".format(self.config.obs_dim_n))
            print("action_dim_n={}".format(self.config.action_dim_n))
            print("state_dim={}".format(self.config.state_dim))
            print(f'max_obs_dim:{self.config.max_obs_dim}')
            
            # Create N agents
            self.agent_n = dict() 
            self.agent_replaybuffer_n = dict()
            obs_n = []
            for agent_id in self.env.possible_agents:
                print("Algorithm: MAPPO")
                self.agent_n[agent_id] = MAPPO(self.config, agent_id)
            self.agent_replaybuffer_n = ReplayBuffer(self.config, agent_id)
            
            # output log dictionary
            self.log_dict_out = None
            
            #self.lr_decay = self.config.lr_decay
            
        except Exception as e:
            print(f'MAPPOWrapper [ex]:{e}')
            raise e

    def step(self, obs_dict, *args, **kwargs):
        torch_obs = []
        for agent in self.env.possible_agents:
            agent_obs = [obs_dict[agent]]
            torch_obs.append(torch.tensor(agent_obs, requires_grad=False))
        #print(f'torch_obs:{torch_obs}')
        
        cat_obs = []
        dim_max = max(tensor.size(1) for tensor in torch_obs)
        padded_tensors = [torch.cat([tensor, torch.zeros(1, dim_max - tensor.size(1))], dim =1) for tensor in torch_obs]
        cat_obs = torch.cat(padded_tensors, dim=0)
        print(f'cat_obs:{cat_obs}')
        
        agent_actions = dict()
        for agent_num in range(self.config.N-1):
            for agent_id in self.env.possible_agents:
                agent_actions[agent_id] = int(self.agent_n[agent_id].choose_action(cat_obs)[agent_num])
                print(f'{agent_id}_action: {agent_actions[agent_id]}')
        return agent_actions
        
    def save(self, fname):
        pass

    def begin_episode(self, ep_i, *args, **kwargs):
        pass

    def end_episode(self, *args, **kwargs):
        pass

    def pre_episode_cycle(self, *args, **kwargs):
        pass

    def post_episode_cycle(self, *args, **kwargs):
        ep_cycle_i, obs_dict, agent_actions, rewards, next_obs, dones = args
        self.log_dict_out = dict(rewards)

        #Sotre the transition
        self.replay_buffer.store_transition(obs_dict, agent_actions, rewards, next_obs, dones)

        if self.replay_buffer.episode_num == self.args.batch_size:
            #train agents and log mean actor loss/mean critic loss (mean over number of episodes/batches)
            for agent_id in self.env.possible_agents:
                loss_out = self.agent_n[agent_id].train(self.replay_buffer, self.agent_n)
                if self.config.log_loss:
                    for l in loss_out:
                        self.log_dict_out[l+'_'+agent_id] = loss_out[l]

    def get(self, *args, **kwargs):
        what = kwargs['what']
        if what == 'log_dict':
            return self.log_dict_out
        return None