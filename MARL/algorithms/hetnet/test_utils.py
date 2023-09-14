from utils import build_hetgraph
import numpy as np

def fake_node_helper(num_P, num_A, h_dim_P, h_dim_A):
    feat_dict = {}

    feat_dict['P'] = np.random.rand(num_P, h_dim_P)

    feat_dict['A'] = np.random.rand(num_A, h_dim_A)

    return feat_dict

def fake_raw_input(num_P, num_A, in_dim_raw):
    raw_f_d = {}

    raw_f_d['P_s'] = np.random.rand(num_P, in_dim_raw['Sensor'][0],
                                    in_dim_raw['Sensor'][1])

    raw_f_d['P'] = np.random.rand(num_P, in_dim_raw['Status'])

    raw_f_d['A'] = np.random.rand(num_A, in_dim_raw['Status'])

    return raw_f_d

#Sanity Check
if __name__ == '__main__':
    num_P = 2
    num_A = 3
    PnP = []
    #PnP = [[0,1]]
    PnA = [[0,2],[1,3],[1,4]] # first is P, second is A
    AnA = [[3,4],[4,3]]

    hetg = build_hetgraph(num_P, num_A, PnP, PnA, AnA, with_state = True)
    print(hetg)
    print(hetg['p2p'].number_of_edges())

    f_d = fake_node_helper(num_P, num_A, h_dim_P = 6, h_dim_A = 3)
    print(f_d)

    in_dim_raw = {'Status': 5,
                  'Sensor': (16,16)}

    r_f_d = fake_raw_input(num_P, num_A, in_dim_raw)
    print(r_f_d)