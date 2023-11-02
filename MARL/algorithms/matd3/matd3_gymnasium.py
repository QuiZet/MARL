import torch
import torch.nn.functional as F
import numpy as np
import copy
from .networks_gymnasium import Actor,Critic_MATD3


class MATD3(object):
    def __init__(self, args, agent_id):
        self.N = args.N
        self.agent_id = agent_id
        self.min_action = args.min_action
        self.max_action = args.max_action
        self.action_dim = args.action_dim_n[agent_id]
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.tau = args.tau
        self.use_grad_clip = args.use_grad_clip
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_update_freq = args.policy_update_freq
        self.actor_pointer = 0
        # Create an individual actor and critic for each agent according to the 'agent_id'
        self.actor = Actor(args, agent_id)
        self.critic = Critic_MATD3(args)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    # Each agent selects actions based on its own local observations(add noise for exploration)
    def choose_action(self, obs, noise_std):
        obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float), 0)
        a = self.actor(obs).data.numpy().flatten()
        a = (a + np.random.normal(0, noise_std, size=self.action_dim)).clip(self.min_action, self.max_action)
        # Cast also the dtype
        #return a
        return a.astype('float32') # Env supports only Float32

    def train(self, replay_buffer, agent_n):
        self.actor_pointer += 1
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()

        batch_obs_wo_dict_n = []
        batch_obs_next_wo_dict_n = []
        batch_a_wo_dict_n = []
        agent_id_as_num = -1
        for i, agent_id in enumerate(batch_obs_next_n):
            batch_obs_wo_dict_n.append(batch_obs_n[agent_id])
            batch_obs_next_wo_dict_n.append(batch_obs_next_n[agent_id])
            batch_a_wo_dict_n.append(batch_a_n[agent_id])
            if agent_id == self.agent_id:
                agent_id_as_num = i
        
        loss_out = dict()
        
        # Compute target_Q
        with torch.no_grad():  # target_Q has no gradient
            # Trick 1:target policy smoothing
            batch_a_next_n = []
            #print(f'agent_n:{agent_n}')
            #print(f'batch_obs_next_n:{batch_obs_next_n}')
            #for i in range(self.N):
            for agent_id in batch_obs_next_n:
                batch_a_next = agent_n[agent_id].actor_target(batch_obs_next_n[agent_id])
                noise = (torch.randn_like(batch_a_next) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                batch_a_next = (batch_a_next + noise).clamp(self.min_action, self.max_action)
                batch_a_next_n.append(batch_a_next)

            # Trick 2:clipped double Q-learning
            Q1_next, Q2_next = self.critic_target(batch_obs_next_wo_dict_n, batch_a_next_n)
            target_Q = batch_r_n[self.agent_id] + self.gamma * (1 - batch_done_n[self.agent_id]) * torch.min(Q1_next, Q2_next)  # shape:(batch_size,1)

        # Compute current_Q
        current_Q1, current_Q2 = self.critic(batch_obs_wo_dict_n, batch_a_wo_dict_n)  # shape:(batch_size,1)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()
        # NOTE: Prepare the output
        loss_out['critic_loss'] = critic_loss.clone().detach().item()

        # Trick 3:delayed policy updates
        if self.actor_pointer % self.policy_update_freq == 0:
            # Reselect the actions of the agent corresponding to 'agent_id', the actions of other agents remain unchanged
            batch_a_n[self.agent_id] = self.actor(batch_obs_n[self.agent_id])
            # >>> IMPORTANT IMPORTANT IMPORTANT <<<
            # Update the batch in a list form after updating from dictionary form
            batch_a_wo_dict_n[agent_id_as_num] = self.actor(batch_obs_n[self.agent_id])
            actor_loss = -self.critic.Q1(batch_obs_wo_dict_n, batch_a_wo_dict_n).mean()  # Only use Q1
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optimizer.step()

            # Softly update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # NOTE: Prepare the output
            loss_out['actor_loss'] = actor_loss.clone().detach().item()
        
        #print(f'actor_loss, critic_loss:{actor_loss, critic_loss}')
        return loss_out

    def save_model(self, env_name, algorithm, number, total_steps, agent_id):
        torch.save(self.actor.state_dict(), "./model/{}/{}_actor_number_{}_step_{}k_agent_{}.pth".format(env_name, algorithm, number, int(total_steps / 1000), agent_id))
