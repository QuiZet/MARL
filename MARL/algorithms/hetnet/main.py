import argparse
from hetnet import A2CHetGat
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

parser =argparse.ArgumentParser(description='Hetnet')
#environment arguments
parser.add_argument('--env_name', default='simple_tag_v3', type=str)
parser.add_argument('hetgat', action='store_true', default=False)
parser.add_argument('--num_agents_class1', default=2, type=int)
parser.add_argument('--num_agents_class2', default=1, type=int)
parser.add_argument('--num_agents_class3', default=2, type=int)
parser.add_argument('--max_cycles', default=25, type=int)
parser.add_argument('--continuous_actions', action='store_true', default=False)
parser.add_argument('--hid_dim', default=64, type=int)
#AC2 params
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--tau', default=0.01, type=float)

args = parser.parse_args()

#env
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

#params for wandb logging
param_config = {
  "activation": "tanh",
  'num_agents': [args.num_agents_class1, args.num_agents_class2, args.num_agents_class3],
  'num_max_cycles': args.max_cycles,
  'num_hid_dim': args.hid_dim,
  "lr": args.lr,
  "tau": args.tau,
  "gamma": args.gamma,
  "epochs": 20
}

wandb.init(project='HetNet', name='A2CHetGat', config=param_config)
wandb.config.update(param_config)

#hydra
#@hydra.main(version_base=None, config_path="conf", config_name="config")
def run():
    if args.hetgat:
        #log hyperparameters to wandb
        lr  =  wandb.config['lr']
        bs = wandb.config['num_agents'][0]
        epochs = wandb.config['epochs']

        #get grid size
        print(f'env.state_space:{env.state_space}')
        grid_size = 44**2 #int(env.state_space) ** 2
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
        
if __name__ == "__main__":
    run()