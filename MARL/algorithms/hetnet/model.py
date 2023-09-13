import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import wandb

#Refer for env argumentshttps://pettingzoo.farama.org/environments/mpe/simple_tag/
from pettingzoo.mpe import simple_tag_v3


#Combine class UAVNetA2CEasy from HetNet/hetgat/uavnet.py with class A2CPolicy from HetNet/hetgat/policy.py
#uavnet.py https://github.com/CORE-Robotics-Lab/HetNet/blob/main/hetgat/uavnet.py#L83
#A2CPolicy https://github.com/CORE-Robotics-Lab/HetNet/blob/main/hetgat/policy.py#L1046
#pos_len = 


class A2CHetGat(object):
    def __init__(self, in_dim, hid_dim, out_dim, num_agents_class1, num_agents_class2, num_agents_class3,
                 num_heads, msg_dim=16, use_CNN=True, use_real=True, use_tanh=False, device='gpu'):
        super().__init__()
        self.device = device
        