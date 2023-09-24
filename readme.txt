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

