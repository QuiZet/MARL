from utils import build_hetgraph
import numpy as np
from pettingzoo.mpe import simple_tag_v3

#pos = [p[:self.in_dim['A']] for p in x[0][0]]
"""
in_dim = {'P': 4,
          'A': 6,
          'state': 9}
"""
#x=[state, prev_hid]
#state = env.reset()
#x[0][0] == observation return from env
"""
    pos_len = args.dim(=5) ** 2 
    SSN_state_len = 4

    in_dim_raw = {'vision': args.vision,
                  'P': pos_len + SSN_state_len,
                  'A': pos_len,
                  'state': SSN_state_len
                  }
    in_dim = {'P': pos_len + SSN_state_len,
              'A': pos_len,
              'state': SSN_state_len}
"""

def fake_node_helper(num_C1, num_C2, h_dim_C1, h_dim_C2):
    feat_dict = {}

    feat_dict['C1'] = np.random.rand(num_C1, h_dim_C1)

    feat_dict['C2'] = np.random.rand(num_C2, h_dim_C2)

    return feat_dict

def fake_raw_input(num_P, num_A, in_dim_raw):
    raw_f_d = {}

    raw_f_d['C1_s'] = np.random.rand(num_C1, in_dim_raw['Sensor'][0],
                                    in_dim_raw['Sensor'][1])

    raw_f_d['C1'] = np.random.rand(num_C1, in_dim_raw['Status'])

    raw_f_d['C2'] = np.random.rand(num_C2, in_dim_raw['Status'])

    return raw_f_d

#Sanity Check
if __name__ == '__main__':
    env = simple_tag_v3.parallel_env()
    obs_dict, _ = env.reset()
    in_dim = {'C1':25,'C2':25,'state':4}
    num_C1 = 1
    num_C2 = 1
    num_C3 = 0
    C1nC1 = []
    #PnP = [[0,1]]
    C1nC2 = [[0,2],[1,3],[1,4]] # first is C1, second is C2
    C2nC2 = [[3,4],[4,3]]
    pos = []
   # pos = [p[:self.in_dim['A']] for p in x[0][0]]

    hetg = build_hetgraph(obs_dict, num_C1, num_C2, num_C3, C1nC1, C1nC2, C2nC2, with_state = True)
    print(hetg)
    print(hetg['c1c1'].number_of_edges())

    f_d = fake_node_helper(num_C1, num_C2, h_dim_C1 = 6, h_dim_C2 = 3)
    print(f_d)

    in_dim_raw = {'Status': 5,
                  'Sensor': (16,16)}

    r_f_d = fake_raw_input(num_C1, num_C2, in_dim_raw)
    print(r_f_d)