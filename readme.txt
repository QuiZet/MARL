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


```
terminal1 : python test/simpletag_dev.py simpletag maddpg
terminal2 : tensorboard --logdir=./models/simpletag/maddpg/run1           <= change the path for other data
open tensorboard (i.e. http://localhost:6006/)
```
