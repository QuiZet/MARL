import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from pettingzoo.mpe import simple_tag_v3

def sample_random_action(env):
    action = [env.action_space(agent).sample() for agent in env.possible_agents]
    return action