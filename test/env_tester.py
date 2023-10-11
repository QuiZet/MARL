import argparse
import torch
import time
import os
import sys
import numpy as np
import importlib
import wandb
import random

from gymnasium.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable


sys.path.append('./')
sys.path.append('../MARL/')

from pettingzoo.mpe import simple_tag_v3

#env
parser = argparse.ArgumentParser(description='RL testbed for multi-agent reinforcement learning algorithms')
parser.add_argument('--framework', type=str, required=True, help='name of the framework')
parser.add_argument('--env', type=str, required=True, help='name of the environment')
parser.add_argument('--parallel_env', type=bool, default=False, help='whether to use parallel environment (default False)')

#model
parser.add_argument('--num_class', type=int, required=True, help='number of classes')
parser.add_argument('--seed', type=int, default=8, help='random seed (default 8)')
parser.add_argument('--model', type=str, default='random', help='name of the model (default random)')
#logging
parser.add_argument('--log_type', type=str, default='terminal', help='logging type (default terminal)')

args = parser.parse_args()


#init env (naive approach)
env = simple_tag_v3.parallel_env()

##init env
#env_name = args.env
#env = importlib.import_module(f'{args.framework}.{env_name}')
#print(f'env:{env}')
#env = env()

##init env v2
#_tmep = importlib.import_module(str(args.framework))
#var = _tmep.env_name
#env = var()

##init env v3
#def call_env(framework, env):
#    env_module = importlib.import_module(framework)
#    env_class = getattr(env_module, env)
#    env_instance = env_class()
#    return env_instance

#init env through function
def call_env(framework, env):
    env = importlib.import_module(f'{framework}.{env}')
    env = env()
    return env


#logger
if args.log_type == 'wandb':
    run = wandb.init(project = f'RL test {args.framework}', entity = 'utokyo-marl', name=f'{args.env}_{args.model}',config = args)
    wandb.watch(arg.model)
    #what to log
    run.log()

#seeding
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

#random action
def sample_random_action(env):
    action = [env.action_space(agent).sample() for agent in env.possible_agents]
    return action

#init model
if args.model != 'random':
    model_name = args.model
else:
    pass

def main():
    env = simple_tag_v3.parallel_env()
    obs, info = env.reset(args.seed)

    while env.agents:
        #Insert policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
    env.close()

if __name__ == '__main__':
    main()