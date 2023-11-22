import copy
import numpy as np

import torch
from torch.autograd import Variable

from gymnasium.spaces import Box, Discrete

from MARL.utils.dictionary import AttrDict

from MARL.algorithms.mappo.mappo_gymnasium import MAPPO
from MARL.algorithms.mappo.replay_buffer_gymnasium import ReplayBuffer
from MARL.models.abstractwrapper import AbstractWrapper

class MAPPOWrapper(AbstractWrapper):

    def dict2list(self, dic):
        r = []
        for d in dic:
            r.append(dic[d])
        return r
    
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
            # Only Homogeneous environment
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
                self.config.obs_dim = self.config.max_obs_dim

            self.config.action_dim_n = self.dict2list(self.config.action_dim_n)
            self.config.obs_dim_n = self.dict2list(self.config.obs_dim_n)
            print(f'self.config:{self.config}')

            # Create env
            env_config = dict()
            env_config['N'] = self.env.max_num_agents  # The number of agents
            env_config['obs_dim'] = self.config.obs_dim_n[0]  # The dimensions of an agent's observation space
            env_config['state_dim'] = self.config.state_dim  # The dimensions of global state space
            env_config['action_dim'] = self.config.action_dim_n[0]  # The dimensions of an agent's action space
            #env_config['episode_limit'] = env_info["episode_limit"]  # Maximum number of steps per episode
            #env_config['epsilon_decay'] = (self.config['epsilon'] - self.config['epsilon_min']) / self.config['epsilon_decay_steps']

            self.all_cfgs = self.config | env_config
            self.all_cfgs = AttrDict(self.all_cfgs)
            print(f'self.all_cfgs:{self.all_cfgs}')

            print(f'self.args.N:{self.config.N}')
            print("obs_dim_n={}".format(self.config.obs_dim_n))
            print("action_dim_n={}".format(self.config.action_dim_n))
            print("state_dim={}".format(self.config.state_dim))
            print(f'max_obs_dim:{self.config.max_obs_dim}')
            # Create N agents
            self.agent_n = dict() 
            self.agent_replaybuffer_n = dict()
            obs_n = []
            print("Algorithm: MAPPO")
            self.agent_n = MAPPO(self.all_cfgs)
            
            self.agent_replaybuffer_n = ReplayBuffer(self.all_cfgs)

            # output log dictionary
            self.log_dict_out = None
            
            self.total_steps = 0
            
            #self.lr_decay = self.config.lr_decay
            
        except Exception as e:
            print(f'MAPPOWrapper [ex]:{e}')
            raise e

    def step(self, obs_dict, *args, **kwargs):
        evaluate = kwargs['evaluate']
        torch_obs = []
        for agent in self.env.possible_agents:
            agent_obs = [obs_dict[agent]]
            torch_obs.append(torch.tensor(agent_obs, requires_grad=False))
        #print(f'torch_obs:{torch_obs}')
        
        state = []
        dim_max = max(tensor.size(1) for tensor in torch_obs)
        padded_tensors = [torch.cat([tensor, torch.zeros(1, dim_max - tensor.size(1))], dim =1) for tensor in torch_obs]
        state = torch.cat(padded_tensors, dim=0)
        print(f'state(concatinated obs):{state.shape}')
        
        agent_actions = dict()
        agent_log_actions = dict()
        agent_actions, agent_log_actions = self.agent_n.choose_action(state, evaluate)

        #print(f'agent_actions, agent_log_actions:{agent_actions.shape} {agent_log_actions} {state.shape}')
        #exit(0)
        v_n = self.agent_n.get_value(state)
        self.total_steps += 1
        if evaluate:
            return agent_actions
        else:
            return agent_actions, agent_log_actions, state, v_n
        
        if evaluate:
            agent_actions[agent_id], agent_log_actions[agent_id] = int(self.agent_n[agent_id].choose_action(cat_obs, evaluate)[agent_num])
            return agent_actions
        else:
            agent_actions, agent_log_actions = self.agent_n.choose_action(cat_obs, evaluate)
            v_n = self.agent_n.get_v(cat_obs)
            return agent_actions, agent_log_actions, v_n
        
    def save(self, fname):
        pass

    def begin_episode(self, ep_i, *args, **kwargs):
        pass

    def end_episode(self, *args, **kwargs):
        pass

    def pre_episode_cycle(self, *args, **kwargs):
        pass

    def post_episode_cycle(self, *args, **kwargs):
        ep_cycle_i, obs_dict, state, v_n, agent_actions, agent_log_actions, rewards, dones = args
        #self.log_dict_out = dict(rewards)

        #print(f'state:{state.shape}')
        state = torch.flatten(state)

        #Sotre the transition
        #for self.agent_id in self.env.possible_agents:
        #    reward += rewards[self.agent_id]
        #    reward = reward / len(self.env.possible_agents)
        #print(f'obs_dict:{obs_dict}, agent_actions:{agent_actions}, rewards:{rewards}, next_obs:{next_obs}, dones:{dones}')
        self.agent_replaybuffer_n.store_transition(ep_cycle_i, obs_dict, state, v_n, agent_actions, agent_log_actions, rewards, dones)
        #episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n
        if self.agent_replaybuffer_n.episode_num == self.all_cfgs.batch_size:
            #train agents and log mean actor loss/mean critic loss (mean over number of episodes/batches)
            loss_out = self.agent_n.train(self.agent_replaybuffer_n, self.total_steps)
            if self.config.log_loss:
                self.log_dict_out=loss_out

    def get(self, *args, **kwargs):
        what = kwargs['what']
        if what == 'log_dict':
            return self.log_dict_out
        return None