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

# Environment
from run import environment
# Logger
from MARL.utils_log import loggers

# Trainer
from run.trainer import run_parallel_env

# Model
from MARL.models.MADDPGWrapper import MADDPGWrapper

@hydra.main(config_path="cfg", config_name="config.yaml", version_base="1.2")
def main(
    cfg: OmegaConf,
):
    # We possibly want to add fields to the config file. Thus, we set struct to False.
    OmegaConf.set_struct(cfg, False)

    verify_config(cfg)

    # display config
    print(OmegaConf.to_yaml(cfg))

    # logger
    # Create logger
    logger = getattr(loggers, cfg.logger.class_name)()
    # Initialize the logger
    logger.init(
        config=dict(OmegaConf.to_container(cfg)),
        **cfg.logger.kwargs
    )
    # pop log information and add to wandb
    if(cfg.logger.class_name == "WandbDistributedLogger"):
        Thread(target = logger.log_loop).start()

    # device
    device = cfg.device

    # environment
    env = environment.make_env(cfg.environment)

    # model
    model = MADDPGWrapper(env, device, **cfg.model)
    
    # start trainer
    run_parallel_env(env, model, logger, cfg.environment)

    # Close the logger
    logger.finish()

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