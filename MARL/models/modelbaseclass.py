# Python program showing
# abstract base class work
from abc import ABC, abstractmethod

class ModelBaseClass(ABC):
    def __init__(self, *args, **kwargs):
        print(f"args:{args}")
        print(f'kwargs:{kwargs}')
            

    @abstractmethod
    def begin_episode(self, *args, **kwargs):
        pass

    @abstractmethod
    def end_episode(self, *args, **kwargs):
        pass

    @abstractmethod
    def pre_episode_cycle(self, *args, **kwargs):
        pass

    @abstractmethod
    def post_episode_cycle(self, *args, **kwargs):
        pass

    @abstractmethod
    def step(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, fname):
        pass