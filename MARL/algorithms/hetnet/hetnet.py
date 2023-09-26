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
              
    What is actually passed to the network is the following(original code):
    in_dim_raw: {'vision': 2, 'P': 29, 'A': 25, 'state': 4} 
    in_dim: {'P': 29, 'A': 25, 'state': 4, 'obs_squares': 25} 
    hid_dim: {'P': 16, 'A': 16, 'state': 16} 
    out_dim: {'P': 5, 'A': 6, 'state': 8}
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
        self.per_agent_critic = per_agent_critic
        self.with_two_state = with_two_state
        
        self.prepro_stat = nn.Linear(in_dim['state'] * self.obs_squares, hid_dim['state'] * self.obs_squares) #reason for *self.obs_squares? : from original code
        self.prepro_obs_C1 = nn.Linear(self.C1_s * self.obs_squares, self.C1_s * self.obs_squares)
        self.prepro_obs_C2 = nn.Linear(self.C2_s * self.obs_squares, self.C2_s * self.obs_squares)
        self.prepro_obs_C3 = nn.Linear(self.C3_s * self.obs_squares, self.C3_s * self.obs_squares)
        
        self.f_module_obs = nn.LSTMCell(in_dim_raw['state'] * self.obs_squares, in_dim_raw['state'])
        self.f_module_stat_C1 = nn.LSTMCell(self.C1_s * self.obs_squares, self.C1_s)
        self.f_module_stat_C2 = nn.LSTMCell(self.C2_s * self.obs_squares, self.C2_s)
        self.f_module_stat_C3 = nn.LSTMCell(self.C3_s * self.obs_squares, self.C3_s)
        
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
        
        for i in range(self.num_C1 + self.num_C2 + self.num_C3):
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
                """
                x:tensor([[[0., 0., 0.,  ..., 1., 0., 0.],
                [0., 1., 0.,  ..., 1., 0., 0.],
                [0., 0., 0.,  ..., 0., 0., 1.]]])
                x:torch.Size([1, 3, 725])
                """
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
            if self.num_C3==0:
                return C1,C2
            else: 
                return C1, C2, C3
        
    def get_obs_features(self, x):
        C1 = torch.zeros(1, self.in_dim['state'])
        for i in range(self.num_C1):
            pass

    
    def init_hidden(self, batch_size):
        h = {}
        h['C1_s'] = tuple((torch.zeros(self.num_C1, self.C1_s, requires_grad=True).to(self.device),
                           torch.zeros(self.num_C1, self.C1_s, requires_grad=True).to(self.device)))
        h['C1_o'] = tuple((torch.zeros(self.num_C1, self.in_dim['state'], requires_grad=True).to(self.device),
                           torch.zeros(self.num_C1, self.in_dim['state'], requires_grad=True).to(self.device)))
        h['C2_s'] = tuple((torch.zeros(self.num_C2, self.C2_s, requires_grad=True).to(self.device),
                           torch.zeros(self.num_C2, self.C2_s, requires_grad=True).to(self.device)))
        h['C2_o'] = tuple((torch.zeros(self.num_C2, self.in_dim['state'], requires_grad=True).to(self.device),
                           torch.zeros(self.num_C2, self.in_dim['state'], requires_grad=True).to(self.device)))
        if self.num_C3 != 0:
            h['C3_s'] = tuple((torch.zeros(self.num_C3, self.C3_s, requires_grad=True).to(self.device),
                            torch.zeros(self.num_C3, self.C3_s, requires_grad=True).to(self.device)))
            h['C3_o'] = tuple((torch.zeros(self.num_C3, self.in_dim['state'], requires_grad=True).to(self.device),
                            torch.zeros(self.num_C3, self.in_dim['state'], requires_grad=True).to(self.device)))
        return h
    
    def forward(self, x, g):
        #x:state, g:graph
        """
        np.shape(x):(2,)
        x is a list of 2 elements, a tensor and dictionary, dictionary contains P_s, P_o, A_s:
        x: [tensor([[[0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 1., 0., 0.]]]), 
         {'P_s': (tensor([[-0.1282,  0.0042, ...,  0.0707,  0.0662,-0.0984],
        [-0.1757,  0.0658,  ...,  0.0793, 0.0318],
        [-0.0868,  0.1846,  ...,  0.2028, -0.1345]], grad_fn=<MulBackward0>), 
        tensor([[-0.2388,  0.0068,  0.2022,  0.0076,  0.3910,  0.2726,  0.2112, -0.0750,
         -0.2984,  0.3829,  0.3367, -0.3802, -0.2464, -0.5738, -0.4936,  0.1726,
         -0.0479,  0.1855,  0.5187,  0.3184, -0.0739, -0.2645,  0.1341,  0.1421,
         -0.1626],
        [-0.3249,  0.1208,  0.2724, -0.0987,  0.1104,  0.2554,  0.1089, -0.1495,
         -0.1165,  0.2105, -0.3298, -0.1133,  0.3043, -0.4150, -0.0432,  0.1063,
         -0.2455,  0.3169,  0.2816, -0.1538, -0.0297, -0.3529, -0.0634,  0.1913,
          0.0524],
        [-0.1617,  0.3387,  0.1661,  0.1329, -0.0184,  0.0365, -0.0397, -0.0046,
         -0.2285,  0.2399, -0.0069, -0.1593,  0.2113, -0.3549, -0.4711,  0.0991,
         -0.0760,  0.2475,  0.2174,  0.0954,  0.0115, -0.6180,  0.0799,  0.4516,
         -0.2421]], grad_fn=<AddBackward0>)), 
         'P_o': (tensor([[-0.3884,  0.3898, -0.0393,  0.1624],
        [-0.3563,  0.4386, -0.0127,  0.2142],
        [-0.3543,  0.4891, -0.0339,  0.2517]], grad_fn=<MulBackward0>), tensor([[-0.5762,  0.6721, -0.0970,  0.2006],
        [-0.5273,  0.8065, -0.0282,  0.2745],
        [-0.5670,  0.9303, -0.0695,  0.3067]], grad_fn=<AddBackward0>)), 
        'A_s': (tensor([], size=(0, 25)), tensor([], size=(0, 25)))}]
        """
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
            
        if self.num_C3 == 0:
            x_c1_stat, x_c2_stat = self.remove_excess_action_features_from_all(x)
            x_c1_obs, x_c2_obs = self.get_obs_features(x).to(self.device)
        else:
            x_c1_stat, x_c2_stat, x_c3_stat = self.remove_excess_action_features_from_all(x)
            x_c1_obs, x_c2_obs, x_c3_obs = self.get_obs_features(x).to(self.device)
            
        feat_dict = {}
        
        state_c1_stat = x_c1_stat.clone().detach()
        state_c2_stat = x_c2_stat.clone().detach()
        state_c3_stat = x_c3_stat.clone().detach()
        
        state_c1_stat = self.relu(self.prepro_stat(state_c1_stat))
        state_c2_stat = self.relu(self.prepro_stat(state_c2_stat))
        state_c3_stat = self.relu(self.prepro_stat(state_c3_stat))
        
        x_c1_obs = x_c1_obs.to(self.device)
        x_c2_obs = x_c2_obs.to(self.device)
        x_c3_obs = x_c3_obs.to(self.device)
                
        x_c1_obs = self.relu(self.prepro_obs_C1(state_c1_stat))
        x_c2_obs = self.relu(self.prepro_obs_C2(state_c2_stat))
        x_c3_obs = self.relu(self.prepro_obs_C3(state_c3_stat))
        
        #self.f_module_obs probably needs modification, individual LSTM for each agent. As f_module_obs(LSTM) input dim is in_dim_raw['state'] * self.obs_squares
        hidden_state_c1_obs, cell_state_c1_obs = self.f_module_obs(x_c1_obs.squeeze(), (hidden_state_c1_obs.double(), cell_state_c1_obs.double()))
        hidden_state_c2_obs, cell_state_c2_obs = self.f_module_obs(x_c2_obs.squeeze(), (hidden_state_c2_obs.double(), cell_state_c2_obs.double()))
        hiddel_state_c3_obs, cell_state_c3_obs = self.f_module_obs(x_c3_obs.squeeze(), (hidden_state_c3_obs.double(), cell_state_c3_obs.double()))
        
        hidden_state_c1_stat, cell_state_c1_stat = self.f_module_stat_C1(state_c1_stat.squeeze(), (hidden_state_c1_stat.double(), cell_state_c1_stat.double()))
        hidden_state_c2_stat, cell_state_c2_stat = self.f_module_stat_C2(state_c2_stat.squeeze(), (hidden_state_c2_stat.double(), cell_state_c2_stat.double()))
        hidden_state_c3_stat, cell_state_c3_stat = self.f_module_stat_C3(state_c3_stat.squeeze(), (hidden_state_c3_stat.double(), cell_state_c3_stat.double()))