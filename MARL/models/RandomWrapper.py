from MARL.utils.dictionary import AttrDict

from MARL.models.modelbaseclass import ModelBaseClass

class RandomWrapper(ModelBaseClass):
    def __init__(self, *args, **kwargs):
        print(f"args:{args}")
        print(f'kwargs:{kwargs}')

        self.config = AttrDict(kwargs)
        print(f'self.config:{self.config}')

        self.env = args[0]
        self.device = args[1]
        print(f'device:{self.device}')
        self.t = 0

    def step(self, torch_obs, explore=True):
        # dummy action (if no MARL policy)
        agent_actions = dict()
        for agent in self.env.possible_agents:
            action = self.env.action_space(agent).sample()  # this is where you would insert your policy
            agent_actions[agent] = action
        return agent_actions

    def save(self, fname):
        pass

    def begin_episode(self, ep_i, *args, **kwargs):
        pass

    def end_episode(self, *args, **kwargs):
        pass

    def pre_episode_cycle(self, *args, **kwargs):
        pass

    def post_episode_cycle(self, *args, **kwargs):
        pass