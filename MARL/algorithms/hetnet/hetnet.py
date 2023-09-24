import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from gymnasium.spaces import Box, Discrete
import numpy as np

from hetgat_real import HeteroGATLayerReal, MultiHeteroGATLayerReal

#Refer for env argumentshttps://pettingzoo.farama.org/environments/mpe/simple_tag/
from pettingzoo.mpe import simple_tag_v3


#Combine class UAVNetA2CEasy from HetNet/hetgat/uavnet.py with class A2CPolicy from HetNet/hetgat/policy.py
#uavnet.py https://github.com/CORE-Robotics-Lab/HetNet/blob/main/hetgat/uavnet.py#L83
#A2CPolicy https://github.com/CORE-Robotics-Lab/HetNet/blob/main/hetgat/policy.py#L1046
#pos_len = 

#Original implementation input dimensions:
""" pos_len = args.dim ** 2
    SSN_state_len = 4

    in_dim_raw = {'vision': args.vision,
                  'P': pos_len + SSN_state_len,
                  'A': pos_len,
                  'state': SSN_state_len
                  }
    in_dim = {'P': pos_len + SSN_state_len,
              'A': pos_len,
              'state': SSN_state_len}
    hid_dim = {'P': 16,
               'A': 16,
               'state': 16}
    out_dim = {'P': 5,
               'A': 6,
               'state': 8}
    obs = None if not hasattr(args, 'vision') else (2 * args.vision + 1) ** 2
 """

#My implementation input dimensions:
"""
    in_dim_raw = {'vision': args.vision,
                    'C1': grid_size,
                    'C2': grid_size,
                    'C3': grid_size,
                    'state': state_len
                    }  
    in_dim = {'C1': grid_size,
              'C2': grid_size,
              'C3': grid_size,
              'state': state_len}
"""
class A2CHetGat(object):
    def __init__(self, in_dim_raw, in_dim, hid_dim, out_dim, num_agents_class1, num_agents_class2, num_agents_class3,
                 num_heads, msg_dim=16, use_CNN=True, device='gpu', per_class_critic=False, per_agent_critic=False,
                 tensor_obs=False, with_two_state=False, obs=1):
        super().__init__()
        self.device = device
        
        self.num_C1 = num_agents_class1
        self.num_C2 = num_agents_class2
        self.num_C3 = num_agents_class3
        
        self.tensor_obs = tensor_obs
        self.vision = in_dim_raw['vision']
        self.world_dim = Box(-np.inf, np.inf, (62,), np.float32)
        
        self.in_dim = in_dim
        self.C1_s = in_dim['C1']
        self.C2_s = in_dim['C2']
        self.C3_s = in_dim['C3']
        self.world_dim = 62 #World grid size of Pettingzoo Simple Tag
        self.hid_size = 32
        self.obs_squares = 1 if obs is None else obs #from original code, in original main.py obs = None (line 247 in main.py)
        
        hid_dim_input = {}
        for key in hid_dim:
            hid_dim_input[key] = hid_dim[key] * self.hid_size
        
        #gnn layers
        #N layers = N round of communication during one time step
        self.layer1 = MultiHeteroGATLayerReal(in_dim, hid_dim, num_heads)
        self.layer2 = MultiHeteroGATLayerReal(hid_dim, hid_dim, num_heads, merge='avg')
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.per_class_critic = per_class_critic
        self.par_agent_critic = per_agent_critic
        self.with_two_state = with_two_state
        if self.per_class_critic:
            self.C1_critic_head = nn.Linear(out_dim['state'], 1)
            self.C2_critic_head = nn.Linear(out_dim['state'], 1)
            self.C3_critic_head = nn.Linear(out_dim['state'], 1)
        elif self.per_agent_critic:
            self.C1_critic_head = nn.Linear(out_dim['C1'] + out_dim['state'], 1)
            self.C2_critic_head = nn.Linear(out_dim['C2'] + out_dim['state'], 1)
            self.C3_critic_head = nn.Linear(out_dim['C3'] + out_dim['state'], 1)
        else:
            self.critic_head = nn.Linear(out_dim['state'], 1)
            
    def remove_excess_action_features_from_all(self, x):
        C1 = torch.zeros(1, self.C1_s)
        C2 = torch.zeros(1, self.C2_s)
        C3 = torch.zeros(1, self.C3_s)
        
        for i in range(self.num_C1+self.num_C2+self.num_C3):
            x_pos, f_pos, t_pos = 0, 0
            if i < self.num_C1:
                dx, df, dt = self.in_dim['C1'], self.in_dim['C1'], self.in_dim['C1']
            elif i < self.num_C1 + self.num_C2:
                dx, df, dt = self.in_dim['C1'], self.in_dim['C2'], self.in_dim['C2']
            else:
                dx, df, dt = self.in_dim['C1'], self.in_dim['C2'], self.in_dim['C3']
                
            f_i = torch.zeros(1, df)
            t_i = torch.zeros(1, dt)
            
            for _ in range(self.obs_squares):
                f_i[0, f_pos:f_pos + df] = x[0][i][x_pos:x_pos + df]
                t_i[0, f_pos:f_pos + dt] = x[0][i][x_pos:x_pos + dt]
                x_pos += dx
                f_pos += df
                t_pos += dt
                
            if i < self.num_C1:
                if i == 0:
                    C1 = f_i
                else:
                    C1 = torch.cat((C1, f_i), dim=0)
            elif i < self.num_C2:
                if i == self.num_C1:
                    C2 = f_i
                else:
                    C2 = torch.cat((C2, f_i), dim=0)
            else:
                if i == self.num_C1 + self.num_C2:
                    C3 = f_i
                else:
                    C3 = torch.cat((C3, f_i), dim=0)
            return C1, C2, C3
        
    def get_obs_features(self, x):
        C1 = torch.zeros(1, self.in_dim['state'])
        for i in range(self.num_C1):


        
    def forward(self, x, g):
        extras = x
        hidden_state_c1_stat = extras['C1_s'][0].to(self.device)
        hidden_state_c2_stat = extras['C2_s'][0].to(self.device)
        hidden_state_c3_stat = extras['C3_s'][0].to(self.device)
        hidden_state_c1_obs = extras['C1_o'][0].to(self.device)
        hidden_state_c2_obs = extras['C2_o'][0].to(self.device)
        hidden_state_c3_obs = extras['C3_o'][0].to(self.device)
        cell_state_c1_stat = extras['C1_s'][1].to(self.device)
        cell_state_c2_stat = extras['C2_s'][1].to(self.device)
        cell_state_c3_stat = extras['C3_s'][1].to(self.device)
        cell_state_c1_obs = extras['C1_o'][1].to(self.device)
        cell_state_c2_obs = extras['C2_o'][1].to(self.device)
        cell_state_c3_obs = extras['C3_o'][1].to(self.device)
        
        x_c1_stat, x_c2_stat, x_c3_stat = self.remove_excess_action_features_from_all(x)
        x_c1_obs, x_c2_obs, x_c3_obs = self.get_obs_features(x).to(self.device)
