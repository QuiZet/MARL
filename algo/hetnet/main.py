import argparse
from model import A2CHetGat

parser =argparse.ArgumentParser(description='Hetnet')
#environment arguments
parser.add_argument('--env_name', default='simple_tag_v3', type=str)
parser.add_argument('hetgat', action='store_true', default=False)
parser.add_argument('--num_agents_class1', default=2, type=int)
parser.add_argument('--num_agents_class2', default=1, type=int)
parser.add_argument('--num_agents_class3', default=2, type=int)
parser.add_argument('--max_cycles', default=25, type=int)
parser.add_argument('--continuous_actions', action='store_true', default=False)
parser.add_argument('hid_dim', default=64, type=int)


args = parser.parse_args()

from pettingzoo.mpe import simple_tag_v3

env = simple_tag_v3.env(num_good=args.num_agents_class1, num_adversaries=args.num_agents_class2, num_obstacles=args.num_agents_class3,
                        max_cycles=args.max_cycles, continuous_actions=args.continuous_actions)

observations = env.reset()

observation_space = env.observation_space
#print(f'observation_space: {env.observation_spaces}')
action_space = env.action_space
#print(f'action_space: {env.action_spaces}')

grid_size = env.state_space.shape[0]
#print(f'grid_size: {grid_size}')

if args.hetgat:
    #get grid size
    grid_size = int(env.state_space) ** 2
    state_len = (args.num_agents_class2 + args.num_agents_class3 +1) #change if class1 is included in training
    
    in_dim = {'class2': grid_size, #why is in_dim[class2,class3] == grid_size?
              'class3': grid_size, 
              'state': state_len} #why is state = num_agents_class2 + num_agents_class3 +1 ?
    
    hid_dim1 = {'class2': args.hid_dim,
                'class3': args.hid_dim,
                'state': args.hid_dim}
    
    out_dim = {'class2': 5,
               'class3': 6,
                'state': 8} #following original hetnet code
    
    