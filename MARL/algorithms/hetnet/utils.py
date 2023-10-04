import numpy as np
import torch
import dgl

'''
Helper function for building heterograph (supports up to 3 classes, default is 2calsses)
    pos: ordered list of one hot of each agents position
    num_C1: number of agents in class 1
    num_C2: number of agents in class 2
    num_C3: number of agents in class 3
    C1nC1, C1nC2, C2nC2: lists of communication pairs of different types
        [[0,2],[1,3],...]
        C1 id starts with 0
        C1 and C2 share the same idx sequence
        C1nC1 and C2nC2 have both edges of [0, 1] & [1, 0]
        C1nC2 only have one (no C2nC1)
    with_state: w/or w/o including state summary node
    comm_range_*: range of communication from agents of * type
'''

#Q&A
#1. what is pos? -> ordered list of one hot of each agents position
#2. what is with_state? -> with_state: w/or w/o including state summary node
#3. whtat is with_self_loop? -> probably self loop for each node
#4. what is with_two_state? -> probably allowing two state for each node
#5. why is comm_range_P=-1? -> probably no range limit for P

def cartesian_from_one_hot(one_hot):
    dim = np.sqrt(len(one_hot))
    hot_one = np.argmax(one_hot)

    x = int(hot_one % dim)
    y = int(np.floor(hot_one / dim))

    return np.array([x, y])


def build_hetgraph(agent_names, obs, num_C1, num_C2, num_C3=0, C1nC1=None, C1nC2=None, C2nC2=None,
        with_state=False, with_self_loop=False, with_two_state=False,
        comm_range_C1=-1, comm_range_C2=-1, comm_range_C3=-1):
    
    #pos: dictionary of agents obs {'adversary0':array[],'adversary1':array[],...}
    #pos_coords = [cartesian_from_one_hot(pos[x]) for x in pos]
    pos_dist = {}
    #agent_names : list of agent names ['adversary0','adversary1',...]
    agent_names = agent_names

    #list of communication pairs of agents, ex) [[0,2],[1,3],...]
    C1nC1, C1nC2, C2nC1, C2nC2 = [], [], [], []
    #list of communication pairs of agents' distance
    C1nC1_dist, C1nC2_dist, C2nC1_dist, C2nC2_dist = [], [], [], []

    C1_i = range(num_C1) #class_1 agent index starts from 0, ends at (num_C1)-1
    C2_i = range(num_C1, num_C1 + num_C2)
    
    #TODO:modify for 3 classes,and improve to use for different envs
    def get_rel_pos(name, x):
        if "adversary" in name:
            if x < num_C1:  # target is also an adversary, given there are 2 landmarks
                return obs[name][8+2*x:10+2*x] if x != int(name.split("_")[-1]) else obs[name][2:4] #else, return self_pos
            else:  # target is an agent
                return obs[name][8+2*(x+num_C1):10+2*(x+num_C1)]
        else:
            if x < num_C1:  # target is an adversary, given there are 2 landmarks
                return obs[name][8+2*x:10+2*x]
            else:  # target is also an agent
                return obs[name][6+2*(x+num_C1):8+2*(x+num_C1)] if x != int(name.split("_")[-1]) else obs[name][2:4]

    if num_C3 == 0:
        for c1 in C1_i:
            for x in range(num_C1 + num_C2):
                if c1 != x:
                    key = (min(c1, x), max(c1, x))
                    #comm_dist = pos_dist.get(key, np.linalg.norm(pos_coords[c1] - pos_coords[x], ord=2))
                    #Agent and adversary observations: [self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, other_agent_velocities]
                    rel_pos = get_rel_pos(agent_names[c1], x)
                    comm_dist = np.linalg.norm(rel_pos)
                    pos_dist[key] = comm_dist
                    print(f'pos_dist[key]: {pos_dist[key]}')
                    
                    if comm_range_C1 == -1 or comm_dist <= comm_range_C1:
                        if x < num_C1:
                            C1nC1.append([c1, x])
                            C1nC1_dist.append(comm_dist)
                        else:
                            C1nC2.append([c1, x])
                            C1nC2_dist.append(comm_dist)

        for c2 in C2_i:
            for x in range(num_C1 + num_C2):
                if c2 != x:
                    key = (min(c2, x), max(c2, x))
                    rel_pos = get_rel_pos(agent_names[c2], x)
                    comm_dist = np.linalg.norm(rel_pos)
                    pos_dist[key] = comm_dist

                    if comm_range_C2 == -1 or comm_dist <= comm_range_C2:
                        if x < num_C1:
                            C2nC1.append([c2, x])
                            C2nC1_dist.append(comm_dist)
                        else:
                            C2nC2.append([c2, x])
                            C2nC2_dist.append(comm_dist)

#Graph Data preperation for 3 Class scenario
    else: # num_C3 !=0:
        C1nC3, C2nC3, C3nC1, C3nC2, C3nC3 = [], [], [], [], []
        C1nC3_dist, C2nC3_dist, C3nC1_dist, C3nC2_dist, C3nC3_dist = [], [], [], [], []
        C3_i = range(num_C1 + num_C2, num_C1 + num_C2 + num_C3)

        for c1 in C1_i:
            for x in range(num_C1 + num_C2 + num_C3):
                if c1 != x:
                    key = (min(c1, x), max(c1, x))
                    rel_pos = get_rel_pos(agent_names[c1], x)
                    comm_dist = np.linalg.norm(rel_pos)
                    pos_dist[key] = comm_dist

                    if comm_range_C1 == -1 or comm_dist <= comm_range_C1:
                        if x < num_C1:
                            C1nC1.append([c1, x])
                            C1nC1_dist.append(comm_dist)
                        elif x < num_C1 + num_C2:
                            C1nC2.append([c1, x])
                            C1nC2_dist.append(comm_dist)
                        else:
                            C1nC3.append([c1,x])
                            C1nC3_dist.append(comm_dist)

        for c2 in C2_i:
            for x in range(num_C1 + num_C2 + num_C3):
                if c2 != x:
                    key = (min(c2, x), max(c2, x))
                    rel_pos = get_rel_pos(agent_names[c2], x)
                    comm_dist = np.linalg.norm(rel_pos)
                    pos_dist[key] = comm_dist

                    if comm_range_C2 == -1 or comm_dist <= comm_range_C2:
                        if x < num_C1:
                            C2nC1.append([c2, x])
                            C2nC1_dist.append(comm_dist)
                        elif x < num_C1 + num_C2:
                            C2nC2.append([c2, x])
                            C2nC2_dist.append(comm_dist)
                        else:
                            C2nC3.append([c2,x])
                            C2nC2_dist.append(comm_dist)

        for c3 in C3_i:
            for x in range(num_C1 + num_C2 + num_C3):
                if c3 != x:
                    key = (min(c3, x), max(c3, x))
                    rel_pos = get_rel_pos(agent_names[c3], x)
                    comm_dist = np.linalg.norm(rel_pos)
                    pos_dist[key] = comm_dist

                    if comm_range_C3 == -1 or comm_dist <= comm_range_C3:
                        if x < num_C1:
                            C3nC1.append([c3, x])
                            C3nC1_dist.append(comm_dist)
                        elif x < num_C1 + num_C2:
                            C3nC2.append([c3, x])
                            C3nC2_dist.append(comm_dist)
                        else:
                            C3nC3.append([c3,x])
                            C3nC3_dist.append(comm_dist)

    if with_state:
        if with_two_state:
            if num_C3 == 0: #if there is no Class 3
                num_nodes_dict = {'C1': num_C1, 'C2': num_C2, 'state': 2}
            else:
                num_nodes_dict = {'C1': num_C1, 'C2': num_C2, 'C3':num_C3, 'state': 2}
        else:
            if num_C3 == 0:
                num_nodes_dict = {'C1': num_C1, 'C2': num_C2, 'state': 1}
            else:
                num_nodes_dict = {'C1': num_C1, 'C2': num_C2, 'C3':num_C3, 'state': 1}
    else:
        if num_C3 == 0:
            num_nodes_dict = {'C1': num_C1, 'C2': num_C2}
        else:
            num_nodes_dict = {'C1': num_C1, 'C2': num_C2, 'C3':num_C3}

    data_dict = {}

    c1c1_u, c1c2_u, c1c3_u, c2c1_u, c2c2_u, c2c3_u, c3c1_u, c3c2_u, c3c3_u = [], [], [], [], [], [], [], [], []
    c1c1_v, c1c2_v, c1c3_v, c2c1_v, c2c2_v, c2c3_v, c3c1_v, c3c2_v, c3c3_v = [], [], [], [], [], [], [], [], []

    for i in range(len(C1nC1)):
        c1c1_u.append(C1nC1[i][0]) #C1nC1[i][0] is the id of the first agent in the communication pair (int)
        c1c1_v.append(C1nC1[i][1]) #C1nC1[i][1] is the id of the second agent in the communication pair (int)

    for i in range(len(C1nC2)):
        c1c2_u.append(C1nC2[i][0])
        c1c2_v.append(C1nC2[i][1] - num_C1) #reason for - num_C1: C1 and C2 share the same idx sequence

    for i in range(len(C2nC1)):
        c2c1_u.append(C2nC1[i][0] - num_C1)
        c2c1_v.append(C2nC1[i][1])

    for i in range(len(C2nC2)):
        c2c2_u.append(C2nC2[i][0] - num_C1)
        c2c2_v.append(C2nC2[i][1] - num_C1)

    if num_C3 != 0:
        for i in range(len(C1nC3)):
            c1c3_u.append(C1nC3[i][0])
            c1c3_v.append(C1nC3[i][1] - num_C1 - num_C2)
        for i in range(len(C2nC3)):
            c2c3_u.append(C2nC3[i][0] - num_C1)
            c2c3_v.append(C2nC3[i][1] - num_C1 - num_C2)
        
        for i in range(len(C3nC1)):
            c3c1_u.append(C3nC1[i][0] - num_C1 - num_C2)
            c3c1_v.append(C3nC1[i][1])
        
        for i in range(len(C3nC2)):
            c3c2_u.append(C3nC2[i][0] - num_C1 - num_C2)
            c3c2_v.append(C3nC2[i][1] - num_C1)
        
        for i in range(len(C3nC3)):
            c3c3_u.append(C3nC3[i][0] - num_C1 - num_C2)
            c3c3_v.append(C3nC3[i][1] - num_C1 - num_C2)

    if with_self_loop:
        for i in range(num_C1):
            c1c1_u.append(i)
            c1c1_v.append(i)
        for i in range(num_C2):
            c2c2_u.append(i)
            c2c2_v.append(i)
        if num_C3 != 0:
            for i in range(num_C3):
                c3c3_u.append(i)
                c3c3_v.append(i)

    data_dict[('C1', 'c1c1', 'C1')] = (c1c1_u, c1c1_v)
    data_dict[('C1', 'c1c2', 'C2')] = (c1c2_u, c1c2_v)
    data_dict[('C2', 'c2c1', 'C1')] = (c2c1_u, c2c1_v)
    data_dict[('C2', 'c2c2', 'C2')] = (c2c2_u, c2c2_v)

    if num_C3 != 0:
        data_dict[('C3', 'c3c1', 'C1')] = (c3c1_u, c3c1_v)
        data_dict[('C3', 'c3c2', 'C2')] = (c3c2_u, c3c2_v)
        data_dict[('C3', 'c3c3', 'C3')] = (c3c3_u, c3c3_v)
        data_dict[('C1', 'c1c3', 'C3')] = (c1c3_u, c1c3_v)
        data_dict[('C2', 'c2c3', 'C3')] = (c2c3_u, c2c3_v)

    if with_state:
        if with_two_state:
            # state node #0 is C1 state node
            data_dict[('C1','c1_to_s','state')] = (list(range(num_C1)),
                                              [0 for i in range(num_C1)])
            # state node #1 is C2 state node
            data_dict[('C2','c2_to_s','state')] = (list(range(num_C2)),
                                              [1 for i in range(num_C2)])
            data_dict[('state', 'in', 'state')] = ([0, 1], [0, 1])
            # state node #2 is C3 state node
            if num_C3 != 0:
                data_dict[('C3','c3_to_s','state')] = (list(range(num_C3)),
                                                  [2 for i in range(num_C3)])
            data_dict[('state', 'in', 'state')] = ([0, 1, 2], [0, 1, 2])
        else: # with one state
            data_dict[('C1','c1_to_s','state')] = (list(range(num_C1)),
                                              np.zeros(num_C1, dtype=np.int64))
            data_dict[('C2','c2_to_s','state')] = (list(range(num_C2)),
                                              np.zeros(num_C2, dtype=np.int64))
            data_dict[('state', 'in', 'state')] = ([0], [0])
            if num_C3 != 0:
                data_dict[('C3','c3_to_s','state')] = (list(range(num_C3)),
                                                  np.zeros(num_C3, dtype=np.int64))
                data_dict[('state', 'in', 'state')] = ([0], [0], [0])
    
    #dgl.heterograph reference:https://docs.dgl.ai/en/0.8.x/generated/dgl.heterograph.html#dgl.heterograph
    g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    
    #dgl edata update reference:https://docs.dgl.ai/en/0.8.x/generated/dgl.DGLGraph.edata.html
    #Return an edge data view for setting/getting edge features.
    #update edge data for each edge type with distance between each pair of agents
    g['c1c1'].edata.update({'dist': torch.Tensor(C1nC1_dist)})
    g['c1c2'].edata.update({'dist': torch.Tensor(C1nC2_dist)})
    g['c2c1'].edata.update({'dist': torch.Tensor(C2nC1_dist)})
    g['c2c2'].edata.update({'dist': torch.Tensor(C2nC2_dist)})

    if num_C3 != 0:
        g['c1c3'].edata.update({'dist': torch.Tensor(C1nC3_dist)})
        g['c2c3'].edata.update({'dist': torch.Tensor(C2nC3_dist)})
        g['c3c1'].edata.update({'dist': torch.Tensor(C3nC1_dist)})
        g['c3c2'].edata.update({'dist': torch.Tensor(C3nC2_dist)})
        g['c3c3'].edata.update({'dist': torch.Tensor(C3nC3_dist)})

    # g['c1c1'].srcdata.update({'point': P_pos_coords})
    # g['c1c1'].dstdata.update({'point': P_pos_coords})

    # g['c1c2'].srcdata.update({'point': P_pos_coords})
    # g['c1c2'].dstdata.update({'point': A_pos_coords})

    # g['c2c1'].srcdata.update({'point': A_pos_coords})
    # g['c2c1'].dstdata.update({'point': P_pos_coords})

    # g['c2c2'].srcdata.update({'point': A_pos_coords})
    # g['c2c2'].dstdata.update({'point': A_pos_coords})

    return g