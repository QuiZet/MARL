How to Install

Install pytorch separately
pip install -r requirements.txt

Demo

Collection of working environments

Test

Collection of trainable environments (partially corrected or completed)

algorithms

Collection of MARL algorithms.
  maddpg_naive : Lee-san's imlementation.
  maddpg : Some issues to fix. 

Code to run script
  from MARL directory run the following code
  python test/simpletag_dev.py cooperativepong maddpg


MARL/utils_log
  Collection of files to create log data. Decorators and Console visualization are also collectect here.
  The log class (writer), supports multi-threading and push the result in the main thread only.

MARL/configs
  Configuration files for the application. It should not be mandatory. 
  It uses dataclasses.

MARL/build_hetgraph
  Initialization of hetgraph using dgl, given observation dictionary of multiple agents
  python MARL/algorithms/hetnet/test_utils.py


# Run a trainable environments

No Logger
python run/environment_trainer.py ++logger.class_name='NoLogger'
python run/environment_trainer.py ++logger.class_name='NoLogger' model=ddpg_maddpg/default
python run/environment_trainer.py ++logger.class_name='NoLogger' environment=pettingzoo_mpe_simple_v3/default 
python run/environment_trainer.py ++logger.class_name='NoLogger' environment=pettingzoo_mpe_simple_v3/default model=minimal/default
python run/environment_trainer.py ++logger.class_name='NoLogger' model=maddpg/default ++model.min_action=0
python run/environment_trainer.py ++logger.class_name='NoLogger' model=matd3/default ++model.min_action=0
python run/environment_trainer.py ++logger.class_name='NoLogger' model=maddpg ++model.min_action=0 environment=pettingzoo_mpe_simple_adversary_v3

WanDB
python run/environment_trainer.py ++logger.kwargs.name='baseline' ++logger.kwargs.group='MATD3' model=matd3 ++model.min_action=0
python run/environment_trainer.py ++logger.kwargs.name='baseline' ++logger.kwargs.group='MADDPG' model=maddpg ++model.min_action=0

#Train MAPPO for PettingZoo envds

Before running code, in MARL/MARL/algorithms/mappo/MAPPO_MPE_main.py assign env arguments in line 23 and env name in line 197
Create a folder named 'data_train' and 'model' in github_projects directory
python ~/MARL/MARL/algorithms/mappo/MAPPO_MPE_main.py