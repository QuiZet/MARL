#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Main entry point for the project.

Example:

    For more detailed examples, check the code.
    
Todo:

"""

import sys
sys.path.append('.')
from threading import Thread

# Configs
# https://towardsdatascience.com/complete-tutorial-on-how-to-use-hydra-in-machine-learning-projects-1c00efcc5b9b
import hydra
from omegaconf import OmegaConf

from MARL.utils_log.printing import print_config

# Environment
from run import environment
# Logger
from MARL.utils_log import loggers

# Trainer
import register
import trainer
import trainer_smacv2
import trainer_smacv3

# Model
import MARL.models

@hydra.main(config_path="cfg", config_name="config.yaml", version_base="1.2")
def main(
    cfg: OmegaConf,
):
    # We possibly want to add fields to the config file. Thus, we set struct to False.
    OmegaConf.set_struct(cfg, False)

    verify_config(cfg)

    # display config
    #print(OmegaConf.to_yaml(cfg))
    print_config(cfg, resolve=True) # <-- Pretty

    # logger
    # Create logger
    logger = getattr(loggers, cfg.logger.class_name)()
    # Initialize the logger
    logger.init(
        #dict(OmegaConf.to_container(cfg)),
        **cfg.logger.kwargs
    )
    # pop log information and add to wandb
    if(cfg.logger.class_name == "WandbDistributedLogger"):
        logger_thread_obj = Thread(target = logger.log_loop)
        logger_thread_obj.start()

    # device
    device = cfg.device

    # environment
    try:
        env = environment.make_env(cfg.environment)
        if cfg.evaluate.do is None:
            env_evaluate = None
        elif cfg.evaluate.do.lower() == "make":
            env_evaluate = environment.make_env(cfg.environment)
        elif cfg.evaluate.do.lower() == "copy":
            env_evaluate = env
        else:
            env_evaluate = None
    except Exception as e:
        env = None
        env_evaluate = None
        print('Environment Exception:'.format(e))

    # Combine multiple configurations as input for the model
    try:
        container = dict()
        for val in cfg.model.model_configs:
            # convert in a dictionary
            container[val] = OmegaConf.to_container(getattr(cfg, val))
    except Exception as e:
        print('Container Exception:'.format(e))

    # model
    try:
        print('NOTE[Future arch change]: remove **cfg.model')
        model = getattr(MARL.models, cfg.model.name)(env, device, container, **cfg.model)
    except Exception as e:
        model = None
        print('Model Exception:'.format(e))
    
    # start trainer
    register.get_trainer(cfg.run_env)(env, env_evaluate, model, logger, cfg.environment, cfg.model)
    #run_parallel_env(env, env_evaluate, model, logger, cfg.environment)

    # Close the logger
    logger.finish()

    # Close the environment
    try:
        env.close()
        if cfg.evaluate.do:
            env_evaluate.close()    
    except Exception as e:
        print(f'[ex] environment_trainer.py:{e}')

    # Stop the logger thread
    try:
        if(cfg.logger.class_name == "WandbDistributedLogger"):
            while logger_thread_obj.is_alive():
                print(f'logger_thread_obj:{logger_thread_obj.is_alive()}')
                logger_thread_obj._stop()
            logger_thread_obj.join()
    except Exception as e:
        print(f'[ex] environment_trainer.py:{e}')

def verify_config(cfg: OmegaConf):
    if cfg.train.distributed and cfg.train.avail_gpus < 2:
        raise ValueError(
            f"Distributed only available with more than 1 GPU. Avail={cfg.train.avail_gpus}"
        )
    if cfg.train.batch_size % cfg.train.accumulate_grad_steps:
        raise ValueError(
            f"Batch size must be divisible by the number of grad accumulation steps.\n"
            f"Values: batch_size:{cfg.train.batch_size}, "
            f"accumulate_grad_steps:{cfg.train.accumulate_grad_steps}",
        )

if __name__ == "__main__":
    main()