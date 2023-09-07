from MARL.algorithms.maddpg_dev import MADDPG

class ModelWrapper():
    def __init__(self, *args, **kwargs):
        print(f"args:{args}")
        print(f'kwargs:{kwargs}')
        if kwargs['model_name']=='MADDPG':
            #possible algs: MADDPG,DDPG
            #possible adversary algs: MADDPG,DDPG  
            self.model = MADDPG.init_from_env(kwargs['env'], agent_alg=kwargs['config'].agent_alg,
                                  adversary_alg=kwargs['config'].adversary_alg,
                                  tau=kwargs['config'].tau,
                                  lr=kwargs['config'].lr,
                                  hidden_dim=kwargs['config'].hidden_dim)
            
    def nagents(self):
        return self.model.nagents
    
    def prep_rollouts(self, device='cpu'):
        self.model.prep_rollouts(device)

    def scale_noise(self, noise):
        self.model.scale_noise(noise)

    def reset_noise(self):
        self.model.reset_noise()

    def step(self, torch_obs, explore=True):
        return self.model.step(torch_obs, explore=explore)

    def prep_training(self, device='cpu'):
        self.model.prep_training(device)

    def update(self, sample, a_i, logger):
        self.model.update(sample, a_i, logger=logger)

    def update_all_targets(self):
        self.model.update_all_targets()

    def save(self, fname):
        self.model.save(fname)