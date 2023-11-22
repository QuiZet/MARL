import torch
import torch.nn as nn
import torch.nn.functional as F

# Orthogonal initialization for RNN Layers to avoid exploding/vanishing gradients
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)
            
class Actor_RNN(nn.Module):
    def __init__(self, args, action_dim, actor_input_dim):
        super().__init__()
        self.rnn_hidden = None
        
        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        
        if args.use_orthogonal_init:
            print("------use_orthogonal_init for Actor RNN------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2, gain=0.01)
            
    def forward(self, actor_input):
        #When 'choose_action':actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        #When 'train'        :actor_input.shape=(mini_batch_size*N, actor_input_dim), prob.shape=(mini_batch_size*N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        prob = torch.softmax(self.fc2(self.rnn_hidden), dim=1)
        return prob
    
class Critic_RNN(nn.Module):
    def __init__(self, args, critic_input_dim):
        super().__init__()
        self.rnn_hidden = None
        
        self.fc1 = nn.Linear(critic_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu] 
        if args.use_orthogonal_init:
            print("------use_orthogonal_init for Critic RNN------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2, gain=0.01)
    
    def forward(self, actor_input):
        #When 'get_value':actor_input.shape=(N, actor_input_dim, prob.shape=(N,action_dim)
        #When 'train'    :actor_input.shape=(mini_batch_size*N, actor_input_dim), prob.shape=(mini_batch_size*N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value

class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super().__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        
        if args.use_orthogonal_init:
            print("------use_orthogonal_init for Actor MLP------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)
            
    def forward(self, actor_input):
        #When 'choose_action':actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        #When 'train'        :actor_input.shape=(mini_batch_size, episode_limit, N, actor_input_dim), prob.shape=(mini_batch_size, episode_limit, N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        x - self.activate_func(self.fc2(x))
        prob = torch.softmax(self.fc3(x), dim=-1)
        return prob

class Critic_MLP(nn.Module):
    def __init__(self, args, critic_input_dim):
        super().__init__()
        self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        
        if args.use_orthogonal_init:
            print("------use_orthogonal_init for Critic MLP------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)
            
    def forward(self, critic_input):
        #When 'get_value':critic_input.shape=(N, critic_input_dim), value.shape=(N,1)
        #When 'train'    :critic_input.shape=(mini_batch_size, episode_limit, N, critic_input_dim), value.shape=(mini_batch_size, episode_limit, N, 1)
        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value
    