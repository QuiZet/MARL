from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from .networks import MLPNetwork
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise

class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True, norm_in=False):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        print('self.policy')
        self.policy = MLPNetwork(num_in_pol, num_out_pol, hidden_dim=hidden_dim)
        print('self.target_policy')
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol, hidden_dim=hidden_dim)
        print('self.critic')
        self.critic = MLPNetwork(num_in_critic, 1, hidden_dim=hidden_dim)
        print('self.target_critic')
        self.target_critic = MLPNetwork(num_in_critic, 1, hidden_dim=hidden_dim)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.check_model_shapes()
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = None  # epsilon for eps-greedy -> edit: None
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action and self.exploration is not None:
            self.exploration.reset()

    def scale_noise(self, scale):
        if not self.discrete_action and self.exploration is not None:
            self.exploration.scale = scale
        #if self.discrete_action:
        #    self.exploration = scale
        #else:
        #    self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in the environment for a minibatch of observations.
        Inputs:
            obs (PyTorch Tensor): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Tensor): Actions for this agent
        """
        action = self.policy(obs)

        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore and self.exploration is not None:
                action += Tensor(self.exploration.noise()).to(obs.device)
            action = action.clamp(-1, 1)

        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

    def check_model_shapes(self):
        for target_param, param in zip(self.target_policy.parameters(), self.policy.parameters()):
            print(f"target_param(self.target_policy) shape: {target_param.shape}, param(self.policy) shape: {param.shape}")
