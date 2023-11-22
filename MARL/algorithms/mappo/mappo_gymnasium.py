import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import *
import numpy as np
import copy
from .networks_gymnasium import Actor_RNN, Actor_MLP, Critic_RNN, Critic_MLP


class MAPPO:
    def __init__(self, args):
        #self.agent_id = agent_id
        self.N = args.N
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_limit = args.episode_limit
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_rnn = args.use_rnn
        self.add_agent_id = args.add_agent_id
        self.use_value_clip = args.use_value_clip
        
        #get input dimension of actor and critic
        self.actor_input_dim = args.max_obs_dim
        self.critic_input_dim = args.state_dim
        if self.add_agent_id:
            print("------add_agent_id------")
            self.actor_input_dim += args.N
            self.critic_input_dim += args.N
        
        if self.use_rnn:
            print("------use_rnn------")
            self.actor = Actor_RNN(args, self.action_dim, self.actor_input_dim)
            self.critic = Critic_RNN(args, self.critic_input_dim)
        else:
            print("------no use_rnn------")
            self.actor = Actor_MLP(args, self.actor_input_dim)
            print("------no use_rnn------")
            self.critic = Critic_MLP(args, self.critic_input_dim)
            print("------no use_rnn------")
        
        print("------AA------")
        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        print("------BB------")
        if self.set_adam_eps:
            print("------set_adam_eps------")
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)
        else:
            print("------no set_adam_eps------")
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr)
        
    def choose_action(self, obs_n, evaluate=False):
        with torch.no_grad():
            actor_inputs = []
            actor_inputs.append(obs_n) # obs_n.shape=(N, obs_dim)
            if self.add_agent_id:
                """
                    Add an one-hot vector to represent the agent_id
                    For example, if N=3
                    [obs of agent_1] + [1,0,0]
                    [obs of agent_2] + [0,1,0]
                    [obs of agent_3] + [0,0,1]
                """
                # torch.eye returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
                actor_inputs.append(torch.eye(self.N)) 
                
            actor_inputs = torch.cat([x for x in actor_inputs], dim=-1) #actor_input.shape=(N, actor_input_dim)
            print(f'actor_inputs:{actor_inputs}')
            prob = self.actor(actor_inputs) #prob.shape=(N, action_dim)
            print(f'prob:{prob}')

            if evaluate:  # When evaluating the policy, we select the action with the highest probability
                a_n = prob.argmax(dim=-1)
                return a_n.numpy(), None
            else:
                dist = Categorical(probs=prob)
                a_n = dist.sample()
                a_logprob_n = dist.log_prob(a_n)
                return a_n.numpy(), a_logprob_n.numpy()
            
    #ToDo: check where this function is called and what is passed as global_state
    def get_value(self, global_state):
        with torch.no_grad():
            critic_inputs = []
            global_state = torch.flatten(global_state)
            #Becasue each agent has the same global state, we need to repea the global state N times
            global_state = torch.tensor(global_state, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1) #(global_state_dim, ) -> (N, global_state_dim)
            #global_state = torch.tensor(global_state, dtype=torch.float32) #(global_state_dim, ) -> (N, global_state_dim)
            print(f'global_state:{global_state}, global_state_shape:{global_state.shape}')
            critic_inputs.append(global_state)
            if self.add_agent_id: #Add an one-hot vector to represent the agent_id
                critic_inputs.append(torch.eye(self.N))
            critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_input.shape=(N, critic_input_dim)
            v_n = self.critic(critic_inputs)  # v_n.shape(N,1)
            return v_n.numpy().flatten()
            
    def train(self, replay_buffer, total_steps):
        loss_out = dict()
        batch = replay_buffer.get_training_data() #get training data
        
        #calculate the advantages using GAE
        adv = []
        gae = 0
        with torch.no_grad(): #adv and td_target have no gradient
            deltas = batch['r_n'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['done_n']) - batch['v_n'][:, :-1]  # deltas.shape=(batch_size,episode_limit,N)
            for t in reversed(range(self.episode_limit)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert
            adv = torch.stack(adv, dim=1) #adv.shape=(batch_size,episode_limit,N)
            v_target = adv + batch['v_n'][:, :-1] #v_target.shape=(batch_size,episode_limit,N)
            if self.use_adv_norm: #Normalize the advantages
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)
                
        """
            Get actor_inputs and critic_inputs
            actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
            critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        """
        actor_inputs, critic_inputs = self.get_inputs(batch)
        
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                """
                    get probs_now and values_now
                    probs_now.shape = (mini_batch_size, episode_limit, N, action_dim)
                    values_now.shape = (mini_batch_size, episode_limit, N)
                """
                if self.use_rnn:
                    #If use RNN, we neet to reset rnn_hidden of actor and critic
                    self.actor.rnn_hidden = None
                    self.critic.rnn_hidden = None
                    probs_now, values_now = [], []
                    for t in range(self.episode_limit):
                        prob = self.actor(actor_inputs[index, t].reshape(self.mini_batch_size * self.N, -1)) #prob.shape=(mini_batch_size*N, action_dim)
                        probs_now.appen(prob.reshape(self.mini_batch_size, self.N, -1)) #probs_now.shape=(mini_batch_size, N, action_dim)
                        value = self.critic(critic_inputs[index, t].reshape(self.mini_batch_size * self.N, -1)) #value.shape = (mini_batch_size*N, 1)
                        values_now.append(value.reshape(self.mini_batch_size, self.N))
                    #stack probs and values according to the time (dim=1)
                    probs_now = torch.stack(probs_now, dim=1)
                    values_now = torch.stack(values_now, dim=1)
                else:
                    probs_now = self.actor(actor_inputs[index])
                    values_now = self.critic(critic_inputs[index]).squeeze(-1) #values_now.shape=(mini_batch_size, episode_limit, N)
                    
                dist_now = Categorical(probs_now)
                dist_entropy = dist_now.entropy() # dist_entropy.shape=(mini_batch_size, episode_limit, N)
                a_logprob_n_now = dist_now.log_prob(batch['a_n'][index]) #a_logprob_n_now.shape=(mini_batch_size, episode_limit, N)
                ratios = torch.exp(a_logprob_n_now - batch['a_logprob_n'][index].detach())#ratios.shape=(mini_batch_size, episode_limit, N)
                surr1 = ratios * adv[index] #surr1.shape=(mini_batch_size, episode_limit, N)
                surr2 = torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon) * adv[index] #surr2.shape=(mini_batch_size, episode_limit, N)
                actor_loss = -torch.min(surr1, surr2) -self.entropy_coef * dist_entropy #actor_loss.shape=(mini_batch_size, episode_limit, N)
                loss_out['mean_actor_loss'] = actor_loss.clone().detach().mean().item() 

                if self.use_value_clip:
                    values_old = batch["v_n"][index, :-1].detach()
                    values_error_clipped = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old - v_target[index]
                    values_error_original = values_now - v_target[index]
                    critic_loss = torch.max(values_error_clipped **2, values_error_original **2)
                else:
                    critic_loss = (values_now - v_target[index]) **2
                loss_out['mean_critic_loss'] = critic_loss.clone().detach().mean().item()
                
                self.ac_optimizer.zero_grad()
                ac_loss = actor_loss.mean() + critic_loss.mean()
                ac_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                self.ac_optimizer.step()
                
        if self.use_lr_decay:
            self.lr_decay(total_steps)

        return loss_out
            
    def get_inputs(self, batch):
        actor_inputs, critic_inputs = [], []
        actor_inputs.append(batch['obs_n'])
        critic_inputs.append(batch['s'].unsqueeze(2).repeat(1, 1, self.N, 1))
        if self.add_agent_id:
            #agent_id_one_hot.shape = (mini_batch_size, max_episode_len, N, N)
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).repeat(self.batch_size, self.episode_limit, 1, 1)
            actor_inputs.append(agent_id_one_hot)
            critic_inputs.append(agent_id_one_hot)
            
        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1) #actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1) #critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        return actor_inputs, critic_inputs

    def lr_decay(self, total_steps):
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p['lr'] = lr_now
            
    def save_model(self, env_name, algorithm, number, total_steps, agent_id):
        torch.save(self.actor.state_dict(), "./model/{}/{}_actor_number_{}_step_{}k_agent_{}.pth".format(env_name, algorithm, number, int(total_steps / 1000), agent_id))
        