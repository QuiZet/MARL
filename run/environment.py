import time
import os
import importlib

import numpy as np

def make_env(env_config):
    try:
        i = importlib.import_module(env_config.name)
        env = i.parallel_env(**env_config.hyperparams)
        return env
    except Exception as e:
        print(f'[ex] make_env:{e}')
    return None