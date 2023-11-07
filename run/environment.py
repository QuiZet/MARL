import time
import os
import importlib

def make_env(env_config):
    try:
        i = importlib.import_module(env_config.module)
        env = getattr(i, env_config.function)(**env_config.hyperparams)
        return env
    except Exception as e:
        print(f'[ex] make_env:{e}')
    return None