import numpy as np
import torch
from torch.distributions import Categorical
from torch.autograd import Variable

def select_action(args, action_out):
    num_agents = args.num_C1 =+ args.num_C2 + args.num_C3
    x = torch.zeros((1,num_agents,1))
    m_c1 = Categorical(logits=action_out['C1'])
    m_c2 = Categorical(logits=action_out['C2'])
    m_c3 = Categorical(logits=action_out['C3'])
    
    c1_idx = m_c1.sample()
    c2_idx = m_c2.sample()
    c3_idx = m_c3.sample()
    
    for i in range(args.num_C1):
        x[0,i] = c1_idx[i]