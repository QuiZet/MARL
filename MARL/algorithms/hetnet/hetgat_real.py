import math
import torch
import torch.nn as nn
import dgl.function as fn
from dgl.ops import edge_softmax
from bitstring import BitArray

#binarize latent vector h with length bin_size
def real_to_bin(h, bin_size=32):
    bin_h = torch.zeros((h.shape[0], h.shape[1], bin_size), dtype=int)
    for i, agent in enumerate(h):
        for j, value in enumerate(agent):
            binary = BitArray(float=value, length=bin_size)
            bin_list = torch.tensor(list(map(int, list(binary.bin))))
            bin_h[i,j] = bin_list
    return bin_h

#convert binarized latent vector h to real vector
def bin_to_real(bin_h):
    h = torch.zeros((bin_h.shape[0], bin_h.shape[1]), dtype=torch.double)
    for i, agent in enumerate(bin_h):
        for j, value in enumerate(agent):
            binary = ''.join(list(map(str, value.tolist())))
            real = BitArray(bin=binary)
            real = real.float
            if math.isnan(real) or math.isinf(real):
                real = 0
            h[i,j] = real
    return h

#compute communication loss due to binaraization
def real_comm_loss(dist, h, bitless_k, bin_size=32):
    bin_h = real_to_bin(h, bin_size)
    noise = torch.rand(bin_h.shape)
    for i in range(len(bin_h)):
        bits = h.shape[-1]
        k = bits * bitless_k
        pr = 0.5 * torch.erfc(torch.sqrt(k / (bits * dist[i]**2)))
        for j in range(noise.shape[1]):
            noise_temp = torch.clone(noise[i,j])
            if math.isnan(pr):
                noise[i,:] = 0
                continue
            noise[i,j][noise_temp <= pr] = 1
            noise[i,j][noise_temp > pr] = 0
            noise[i,j][0] = 0
    bin_h = torch.remainder(bin_h + noise, 2).to(int)
    h = bin_to_real(bin_h)
    h[h != h] = 0.0

    return torch.abs(h)

class HeteroGATLayerReal(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, l_alpha=0.2, use_relu=True):
        super().__init__()
        self._num_heads = num_heads
        self._in_dim = in_dim
        self._out_dim = out_dim
        self.fc = nn.ModuleDict({
            'C1': nn.Linear(in_dim['C1'], out_dim['C1'] * num_heads),
            'C2': nn.Linear(in_dim['C2'], out_dim['C2'] * num_heads),
            'C3': nn.Linear(in_dim['C3'], out_dim['C3'] * num_heads),
            'c1c1': nn.Linear(in_dim['C1'], out_dim['C1'] * num_heads),
            'c1c2': nn.Linear(in_dim['C1'], out_dim['C2'] * num_heads),
            'c1c3': nn.Linear(in_dim['C1'], out_dim['C3'] * num_heads),
            'c2c1': nn.Linear(in_dim['C2'], out_dim['C1'] * num_heads),
            'c2c2': nn.Linear(in_dim['C2'], out_dim['C2'] * num_heads),
            'c2c3': nn.Linear(in_dim['C2'], out_dim['C3'] * num_heads),
            'c3c1': nn.Linear(in_dim['C3'], out_dim['C1'] * num_heads),
            'c3c2': nn.Linear(in_dim['C3'], out_dim['C2'] * num_heads),
            'c3c3': nn.Linear(in_dim['C3'], out_dim['C3'] * num_heads),
            'c1s': nn.Linear(in_dim['C1'], out_dim['state'] * num_heads),
            'c2s': nn.Linear(in_dim['C2'], out_dim['state'] * num_heads),
            'c3s': nn.Linear(in_dim['C3'], out_dim['state'] * num_heads),
            'in': nn.Linear(in_dim['state'], out_dim['state'] * num_heads)
            })
        self.leaky_relu = nn.LeakyReLU(negative_slope=l_alpha)
        self.use_relu = use_relu
        if self.use_relu:
            self.relu = nn.ReLU()
        # attention coefficients
        self.c1c1_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C1'])))
        self.c1c1_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C1'])))
        self.c1c2_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C2'])))
        self.c1c2_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C2'])))
        self.c1c3_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C3'])))
        self.c1c3_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C3'])))
        self.c2c1_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C1'])))
        self.c2c1_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C1'])))
        self.c2c2_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C2'])))
        self.c2c2_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C2'])))
        self.c2c3_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C3'])))
        self.c2c3_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C3'])))
        self.c3c1_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C1'])))
        self.c3c1_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C1'])))
        self.c3c2_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C2'])))
        self.c3c2_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C2'])))
        self.c3c3_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C3'])))
        self.c3c3_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C3'])))
        # state node
        # self.attn_fc_c1s = nn.Linear(2 * out_dim['state'], 1, bias=False)
        # self.attn_fc_c2s = nn.Linear(2 * out_dim['state'], 1, bias=False)
        self.c1s_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
        self.c1s_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
        self.c2s_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
        self.c2s_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
        self.c3s_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
        self.c3s_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters
        References
            1 https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html
            2 https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
        """
        gain = nn.init.calculate_gain('relu')

        # attention
        nn.init.xavier_normal_(self.c1c1_src, gain=gain)
        nn.init.xavier_normal_(self.c1c1_dst, gain=gain)
        nn.init.xavier_normal_(self.c1c2_src, gain=gain)
        nn.init.xavier_normal_(self.c1c2_dst, gain=gain)
        nn.init.xavier_normal_(self.c1c3_src, gain=gain)
        nn.init.xavier_normal_(self.c1c3_dst, gain=gain)
        nn.init.xavier_normal_(self.c2c1_src, gain=gain)
        nn.init.xavier_normal_(self.c2c1_dst, gain=gain)
        nn.init.xavier_normal_(self.c2c2_src, gain=gain)
        nn.init.xavier_normal_(self.c2c2_dst, gain=gain)
        nn.init.xavier_normal_(self.c2c3_src, gain=gain)
        nn.init.xavier_normal_(self.c2c3_dst, gain=gain)
        nn.init.xavier_normal_(self.c3c1_src, gain=gain)
        nn.init.xavier_normal_(self.c3c1_dst, gain=gain)
        nn.init.xavier_normal_(self.c3c2_src, gain=gain)
        nn.init.xavier_normal_(self.c3c2_dst, gain=gain)
        nn.init.xavier_normal_(self.c3c3_src, gain=gain)
        nn.init.xavier_normal_(self.c3c3_dst, gain=gain)

        nn.init.xavier_normal_(self.c1s_src, gain=gain)
        nn.init.xavier_normal_(self.c1s_dst, gain=gain)
        nn.init.xavier_normal_(self.c2s_src, gain=gain)
        nn.init.xavier_normal_(self.c2s_dst, gain=gain)
        nn.init.xavier_normal_(self.c3s_src, gain=gain)
        nn.init.xavier_normal_(self.c3s_dst, gain=gain)

    def forward(self, g, feat_dict):
        '''
        From hi to Whi
        '''
        # feature of C1
        if 'C1' in feat_dict:
            Whc1 = self.fc['C1'](feat_dict['C1']).view(-1, self._num_heads, self._out_dim['C1'])
            g.nodes['C1'].data['Wh_C1'] = Whc1

        # feature of C2
        if 'C2' in feat_dict:
            Whc2 = self.fc['C2'](feat_dict['C2']).view(-1, self._num_heads, self._out_dim['C2'])
            g.nodes['C2'].data['Wh_C2'] = Whc2

        # feature of C3
        if 'C3' in feat_dict:
            Whc3 = self.fc['C3'](feat_dict['C3']).view(-1, self._num_heads, self._out_dim['C3'])
            g.nodes['C3'].data['Wh_C3'] = Whc3
        
        '''
        Feature transform for each edge type (communication channel)
        '''
        if 'C1' in feat_dict:
            # c1c1
            Whc1c1 = self.fc['c1c1'](feat_dict['C1']).view(-1, self._num_heads, self._out_dim['C1'])
            g.nodes['C1'].data['Wh_c1c1'] = Whc1c1

            # c1c2
            Whc1c2 = self.fc['c1c2'](feat_dict['C1']).view(-1, self._num_heads, self._out_dim['C2'])
            g.nodes['C1'].data['Wh_c1c2'] = Whc1c2

            if 'C3' in feat_dict:
                #c1c3
                Whc1c3 = self.fc['c1c3'](feat_dict['C1']).view(-1, self._num_heads, self._out_dim['C3'])
                g.nodes['C1'].data['Wh_c1c3'] = Whc1c3

        if 'C2' in feat_dict:
            # c2c1
            Whc2c1 = self.fc['c2c1'](feat_dict['C2']).view(-1, self._num_heads, self._out_dim['C1'])
            g.nodes['C2'].data['Wh_c2c1'] = Whc2c1

            # c2c2
            Whc2c2 = self.fc['c2c2'](feat_dict['C2']).view(-1, self._num_heads, self._out_dim['C2'])
            g.nodes['C2'].data['Wh_c2c2'] = Whc2c2

            if 'C3' in feat_dict:
                #c2c3
                Whc2c3 = self.fc['c2c3'](feat_dict['C2']).view(-1, self._num_heads, self._out_dim['C3'])
                g.nodes['C2'].data['Wh_c2c3'] = Whc2c3

        if 'C3' in feat_dict:
            #c3c1
            Whc3c1 = self.fc['c3c1'](feat_dict['C3']).view(-1, self._num_heads, self._out_dim['C1'])
            g.nodes['C3'].data['Wh_c3c1'] = Whc3c1

            #c3c2
            Whc3c2 = self.fc['c3c2'](feat_dict['C3']).view(-1, self._num_heads, self._out_dim['C2'])
            g.nodes['C3'].data['Wh_c3c2'] = Whc3c2

            #c3c3
            Whc3c3 = self.fc['c3c3'](feat_dict['C3']).view(-1, self._num_heads, self._out_dim['C3'])
            g.nodes['C3'].data['Wh_c3c3'] = Whc3c3

        # for state-related edges
        if 'C1' in feat_dict:
            Whc1s = self.fc['c1s'](feat_dict['C1']).view(-1, self._num_heads, self._out_dim['state'])
            g.nodes['C1'].data['Wh_c1s'] = Whc1s

        if 'C2' in feat_dict:
            Whc2s = self.fc['c2s'](feat_dict['C2']).view(-1, self._num_heads, self._out_dim['state'])
            g.nodes['C2'].data['Wh_c2s'] = Whc2s

        if 'C3' in feat_dict:
            Whc3s = self.fc['c3s'](feat_dict['C3']).view(-1, self._num_heads, self._out_dim['state'])
            g.nodes['C3'].data['Wh_c3s'] = Whc3s

        Whin = self.fc['in'](feat_dict['state'].double()).view(-1, self._num_heads, self._out_dim['state'])
        g.nodes['state'].data['Wh_in'] = Whin

        '''
        Attention computation on subgraphs
        '''
        # c1c1
        if g['c1c1'].number_of_edges() > 0:
            Attn_src_c1c1 = (Whc1c1 * self.c1c1_src).sum(dim=-1).unsqueeze(-1)
            Attn_dst_c1c1 = (Whc1 * self.c1c1_dst).sum(dim=-1).unsqueeze(-1)
            g['c1c1'].srcdata.update({'Attn_src_c1c1': Attn_src_c1c1})
            g['c1c1'].dstdata.update({'Attn_dst_c1c1': Attn_dst_c1c1})

            g['c1c1'].apply_edges(fn.u_add_v('Attn_src_c1c1', 'Attn_dst_c1c1', 'e_c1c1'))
            e_c1c1 = self.leaky_relu(g['c1c1'].edata.pop('e_c1c1'))

            # compute softmax
            g['c1c1'].edata['a_c1c1'] = edge_softmax(g['c1c1'], e_c1c1)
            # message passing
            g['c1c1'].update_all(fn.u_mul_e('Wh_c1c1', 'a_c1c1', 'm_c1c1'),
                                fn.sum('m_c1c1', 'ft_c1c1'))

        # c1c2
        if g['c1c2'].number_of_edges() > 0:
            Attn_src_c1c2 = (Whc1c2 * self.c1c2_src).sum(dim=-1).unsqueeze(-1)
            Attn_dst_c1c2 = (Whc2 * self.c1c2_dst).sum(dim=-1).unsqueeze(-1)
            # both works
            # g.nodes['P'].data['Attn_src_c1c2'] = Attn_src_c1c2
            g['c1c2'].srcdata.update({'Attn_src_c1c2': Attn_src_c1c2})
            # g.nodes['A'].data['Attn_dst_c1c2'] = Attn_dst_c1c2
            g['c1c2'].dstdata.update({'Attn_dst_c1c2': Attn_dst_c1c2})
            '''
            Note:
                g.dstdata['Attn_dst_c1c2']['C2'] gives the data tensor
                but g.dstdata['C2'] gives {}
                so the first key right after dstdata is the feature key
                    not the node type key
            '''
            g['c1c2'].apply_edges(fn.u_add_v('Attn_src_c1c2', 'Attn_dst_c1c2', 'e_c1c2'))
            # g['c1c2'].edata['e_c1c2'] gives the tensor
            e_c1c2 = self.leaky_relu(g['c1c2'].edata.pop('e_c1c2'))

            # compute softmax
            g['c1c2'].edata['a_c1c2'] = edge_softmax(g['c1c2'], e_c1c2)
            # message passing
            g['c1c2'].update_all(fn.u_mul_e('Wh_c1c2', 'a_c1c2', 'm_c1c2'),
                                fn.sum('m_c1c2', 'ft_c1c2'))
            # results =  g.nodes['A'].data['ft_c1c2']

        #c1c3
        if g['c1c3'].number_of_edges() > 0:
            Attn_src_c1c3 = (Whc1c3 * self.c1c3_src).sum(dim=-1).unsqueeze(-1)
            Attn_dst_c1c3 = (Whc3 * self.c1c3_dst).sum(dim=-1).unsqueeze(-1)
            g['c1c3'].srcdata.update({'Attn_src_c1c3': Attn_src_c1c3})
            g['c1c3'].dstdata.update({'Attn_dst_c1c3': Attn_dst_c1c3})
            g.apply_edges(fn.u_add_v('Attn_src_c1c3', 'Attn_dst_c1c3', 'e_c1c3'))
            e_c1c3 = self.leaky_relu(g['c1c3'].edata.pop('e_c1c3'))
            g['c1c3'].edata['a_c1c3'] = edge_softmax(g['c1c3'], e_c1c3)
            g['c1c3'].update_all(fn.u_mul_e('Wh_c1c3', 'a_c1c3', 'm_c1c3'),
                                fn.sum('m_c1c3', 'ft_c1c3'))

        # c2c1
        if g['c2c1'].number_of_edges() > 0:
            Attn_src_c2c1 = (Whc2c1 * self.c2c1_src).sum(dim=-1).unsqueeze(-1)
            Attn_dst_c2c1 = (Whc1 * self.c2c1_dst).sum(dim=-1).unsqueeze(-1)
            g['c2c1'].srcdata.update({'Attn_src_c2c1': Attn_src_c2c1})
            g['c2c1'].dstdata.update({'Attn_dst_c2c1': Attn_dst_c2c1})

            g['c2c1'].apply_edges(fn.u_add_v('Attn_src_c2c1', 'Attn_dst_c2c1', 'e_c2c1'))
            e_c2c1 = self.leaky_relu(g['c2c1'].edata.pop('e_c2c1'))

            # compute softmax
            g['c2c1'].edata['a_c2c1'] = edge_softmax(g['c2c1'], e_c2c1)
            # message passing
            g['c2c1'].update_all(fn.u_mul_e('Wh_c2c1', 'a_c2c1', 'm_c2c1'),
                                fn.sum('m_c2c1', 'ft_c2c1'))

        # c2c2
        if g['c2c2'].number_of_edges() > 0:
            Attn_src_c2c2 = (Whc2c2 * self.c2c2_src).sum(dim=-1).unsqueeze(-1)
            Attn_dst_c2c2 = (Whc2 * self.c2c2_dst).sum(dim=-1).unsqueeze(-1)
            g['c2c2'].srcdata.update({'Attn_src_c2c2': Attn_src_c2c2})
            g['c2c2'].dstdata.update({'Attn_dst_c2c2': Attn_dst_c2c2})

            g['c2c2'].apply_edges(fn.u_add_v('Attn_src_c2c2', 'Attn_dst_c2c2', 'e_c2c2'))
            e_c2c2 = self.leaky_relu(g['c2c2'].edata.pop('e_c2c2'))

            # compute softmax
            g['c2c2'].edata['a_c2c2'] = edge_softmax(g['c2c2'], e_c2c2)
            # message passing
            g['c2c2'].update_all(fn.u_mul_e('Wh_c2c2', 'a_c2c2', 'm_c2c2'),
                                fn.sum('m_c2c2', 'ft_c2c2'))

        #c2c3
        if g['c2c3'].number_of_edges() > 0:
            Attn_src_c2c3 = (Whc2c3 * self.c2c3_src).sum(dim=-1).unsqueeze(-1)
            Attn_dst_c2c3 = (Whc3 * self.c2c3_dst).sum(dim=-1).unsqueeze(-1)
            g['c2c3'].srcdata.update({'Attn_src_c2c3': Attn_src_c2c3})
            g['c2c3'].dstdata.update({'Attn_dst_c2c3': Attn_dst_c2c3})
            g.apply_edges(fn.u_add_v('Attn_src_c2c3', 'Attn_dst_c2c3', 'e_c2c3'))
            e_c2c3 = self.leaky_relu(g['c2c3'].edata.pop('e_c2c3'))
            g['c2c3'].edata['a_c2c3'] = edge_softmax(g['c2c3'], e_c2c3)
            g['c2c3'].update_all(fn.u_mul_e('Wh_c2c3', 'a_c2c3', 'm_c2c3'),
                                fn.sum('m_c2c3', 'ft_c2c3'))
            
        #c3c1
        if g['c3c1'].number_of_edges() > 0:
            Attn_src_c3c1 = (Whc3c1 * self.c3c1_src).sum(dim=-1).unsqueeze(-1)
            Attn_dst_c3c1 = (Whc1 * self.c3c1_dst).sum(dim=-1).unsqueeze(-1)
            g['c3c1'].srcdata.update({'Attn_src_c3c1': Attn_src_c3c1})
            g['c3c1'].dstdata.update({'Attn_dst_c3c1': Attn_dst_c3c1})
            g.apply_edges(fn.u_add_v('Attn_src_c3c1', 'Attn_dst_c3c1', 'e_c3c1'))
            e_c3c1 = self.leaky_relu(g['c3c1'].edata.pop('e_c3c1'))
            g['c3c1'].edata['a_c3c1'] = edge_softmax(g['c3c1'], e_c3c1)
            g['c3c1'].update_all(fn.u_mul_e('Wh_c3c1', 'a_c3c1', 'm_c3c1'),
                                fn.sum('m_c3c1', 'ft_c3c1'))
            
        #c3c2
        if g['c3c2'].number_of_edges() > 0:
            Attn_src_c3c2 = (Whc3c2 * self.c3c2_src).sum(dim=-1).unsqueeze(-1)
            Attn_dst_c3c2 = (Whc2 * self.c3c2_dst).sum(dim=-1).unsqueeze(-1)
            g['c3c2'].srcdata.update({'Attn_src_c3c2': Attn_src_c3c2})
            g['c3c2'].dstdata.update({'Attn_dst_c3c2': Attn_dst_c3c2})
            g.apply_edges(fn.u_add_v('Attn_src_c3c2', 'Attn_dst_c3c2', 'e_c3c2'))
            e_c3c2 = self.leaky_relu(g['c3c2'].edata.pop('e_c3c2'))
            g['c3c2'].edata['a_c3c2'] = edge_softmax(g['c3c2'], e_c3c2)
            g['c3c2'].update_all(fn.u_mul_e('Wh_c3c2', 'a_c3c2', 'm_c3c2'),
                                fn.sum('m_c3c2', 'ft_c3c2'))
            
        #c3c3
        if g['c3c3'].number_of_edges() > 0:
            Attn_src_c3c3 = (Whc3c3 * self.c3c3_src).sum(dim=-1).unsqueeze(-1)
            Attn_dst_c3c3 = (Whc3 * self.c3c3_dst).sum(dim=-1).unsqueeze(-1)
            g['c3c3'].srcdata.update({'Attn_src_c3c3': Attn_src_c3c3})
            g['c3c3'].dstdata.update({'Attn_dst_c3c3': Attn_dst_c3c3})
            g.apply_edges(fn.u_add_v('Attn_src_c3c3', 'Attn_dst_c3c3', 'e_c3c3'))
            e_c3c3 = self.leaky_relu(g['c3c3'].edata.pop('e_c3c3'))
            g['c3c3'].edata['a_c3c3'] = edge_softmax(g['c3c3'], e_c3c3)
            g['c3c3'].update_all(fn.u_mul_e('Wh_c3c3', 'a_c3c3', 'm_c3c3'),
                                fn.sum('m_c3c3', 'ft_c3c3'))

        # c1s
        if g['c1s'].number_of_edges() > 0:
            Attn_src_c1s = (Whc1s * self.c1s_src).sum(dim=-1).unsqueeze(-1)
            Attn_dst_c1s = (Whin * self.c1s_dst).sum(dim=-1).unsqueeze(-1)
            g['c1s'].srcdata.update({'Attn_src_c1s': Attn_src_c1s})
            g['c1s'].dstdata.update({'Attn_dst_c1s': Attn_dst_c1s})

            g['c1s'].apply_edges(fn.u_add_v('Attn_src_c1s', 'Attn_dst_c1s', 'e_c1s'))
            e_c1s = self.leaky_relu(g['c1s'].edata.pop('e_c1s'))

            # compute softmax
            g['c1s'].edata['a_c1s'] = edge_softmax(g['c1s'], e_c1s)
            # message passing
            g['c1s'].update_all(fn.u_mul_e('Wh_c1s', 'a_c1s', 'm_c1s'),
                                fn.sum('m_c1s', 'ft_c1s'))

        # c2s
        if g['c2s'].number_of_edges() > 0:
            Attn_src_c2s = (Whc2s * self.c2s_src).sum(dim=-1).unsqueeze(-1)
            Attn_dst_c2s = (Whin * self.c2s_dst).sum(dim=-1).unsqueeze(-1)
            g['c2s'].srcdata.update({'Attn_src_c2s': Attn_src_c2s})
            g['c2s'].dstdata.update({'Attn_dst_c2s': Attn_dst_c2s})

            g['c2s'].apply_edges(fn.u_add_v('Attn_src_c2s', 'Attn_dst_c2s', 'e_c2s'))
            e_c2s = self.leaky_relu(g['c2s'].edata.pop('e_c2s'))

            # compute softmax
            g['c2s'].edata['a_c2s'] = edge_softmax(g['c2s'], e_c2s)
            # message passing
            g['c2s'].update_all(fn.u_mul_e('Wh_c2s', 'a_c2s', 'm_c2s'),
                                fn.sum('m_c2s', 'ft_c2s'))

        #c3s
        if g['c3s'].number_of_edges() > 0:
            Attn_src_c3s = (Whc3s * self.c3s_src).sum(dim=-1).unsqueeze(-1)
            Attn_dst_c3s = (Whin * self.c3s_dst).sum(dim=-1).unsqueeze(-1)
            g['c3s'].srcdata.update({'Attn_src_c3s': Attn_src_c3s})
            g['c3s'].dstdata.update({'Attn_dst_c3s': Attn_dst_c3s})

            g['c3s'].apply_edges(fn.u_add_v('Attn_src_c3s', 'Attn_dst_c3s', 'e_c3s'))
            e_c3s = self.leaky_relu(g['c3s'].edata.pop('e_c3s'))

            # compute softmax
            g['c3s'].edata['a_c3s'] = edge_softmax(g['c3s'], e_c3s)
            # message passing
            g['c3s'].update_all(fn.u_mul_e('Wh_c3s', 'a_c3s', 'm_c3s'),
                                fn.sum('m_c3s', 'ft_c3s'))

        '''
        Combine features from subgraphs
            Sum up to hi'
            Need to check if subgraph is valid
        '''
        valid_ntypes = []

        if 'C1' in feat_dict:
            valid_ntypes.append('C1')
            # new feature of C1
            Whc1_new = g.nodes['C1'].data['Wh_C1'].clone()

            if g['c1c1'].number_of_edges() > 0:
                Whc1_new += g.nodes['C1'].data['ft_c1c1']
            if g['c2c1'].number_of_edges() > 0:
                Whc1_new += g.nodes['C1'].data['ft_c2c1']
            if g['c3c1'].number_of_edges() > 0:
                Whc1_new += g.nodes['C1'].data['ft_c3c1']

            g.nodes['C1'].data['h'] = Whc1_new

        if 'C2' in feat_dict:
            valid_ntypes.append('C2')
            # new feature of C2
            Whc2_new = g.nodes['C2'].data['Wh_C2'].clone()

            if g['c1c2'].number_of_edges() > 0:
                Whc2_new += g.nodes['C2'].data['ft_c1c2']
            if g['c2c2'].number_of_edges() > 0:
                Whc2_new += g.nodes['C2'].data['ft_c2c2']
            if g['c3c2'].number_of_edges() > 0:
                Whc2_new += g.nodes['C2'].data['ft_c3c2']

            g.nodes['C2'].data['h'] = Whc2_new

        if 'C3' in feat_dict:
            valid_ntypes.append('C3')
            # new feature of C3
            Whc3_new = g.nodes['C3'].data['Wh_C3'].clone()

            if g['c1c3'].number_of_edges() > 0:
                Whc3_new += g.nodes['C3'].data['ft_c1c3']
            if g['c2c3'].number_of_edges() > 0:
                Whc3_new += g.nodes['C3'].data['ft_c2c3']
            if g['c3c3'].number_of_edges() > 0:
                Whc3_new += g.nodes['C3'].data['ft_c3c3']

            g.nodes['C3'].data['h'] = Whc3_new

        valid_ntypes.append('state')
        # new feature of state
        Whstate_new = g.nodes['state'].data['Wh_in'].clone()
        # + \
        #     g.nodes['state'].data['ft_c1s'] + \
        #         g.nodes['state'].data['ft_c2s']

        if g['c1s'].number_of_edges() > 0:
            Whstate_new += g.nodes['state'].data['ft_c1s']
        if g['c2s'].number_of_edges() > 0:
            Whstate_new += g.nodes['state'].data['ft_c2s']
        if g['c3s'].number_of_edges() > 0:
            Whstate_new += g.nodes['state'].data['ft_c3s']

        g.nodes['state'].data['h'] = Whstate_new

        # deal with relu activation
        if self.use_relu:
            return {ntype: self.relu(g.nodes[ntype].data['h']) for ntype in valid_ntypes}
        else:
            return {ntype: g.nodes[ntype].data['h'] for ntype in valid_ntypes}

# merge = 'cat' or 'avg'
class MultiHeteroGATLayerReal(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, merge='cat'):
        super().__init__()

        self._num_heads = num_heads
        self._merge = merge

        if self._merge == 'cat':
            self.gat_conv = HeteroGATLayerReal(in_dim, out_dim, num_heads)
        else:
            self.gat_conv = HeteroGATLayerReal(in_dim, out_dim, num_heads, use_relu=False)

    def forward(self, g, feat_dict):
        tmp = self.gat_conv(g, feat_dict)
        results = {}

        valid_ntypes = []

        if 'C1' in feat_dict:
            valid_ntypes.append('C1')

        if 'C2' in feat_dict:
            valid_ntypes.append('C2')
        
        if 'C3' in feat_dict:
            valid_ntypes.append('C3')

        valid_ntypes.append('state')

        if self._merge == 'cat':
            # concat on the output feature dimension (dim=1)
            for ntype in valid_ntypes:
                results[ntype] = torch.flatten(tmp[ntype], 1)
        else:
            # merge using average
            for ntype in valid_ntypes:
                results[ntype] = torch.mean(tmp[ntype], 1)

        return results

class HeteroGATLayerLossyReal(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, l_alpha=0.2, use_relu=True,
            comm_range_C1=-1, comm_range_C2=-1, comm_range_C3=-1, min_comm_loss=0, max_comm_loss=0.3):
        super().__init__()

        self._num_heads = num_heads
        self._in_dim = in_dim
        self._out_dim = out_dim

        self.comm_range_C1 = comm_range_C1
        self.comm_range_C2 = comm_range_C2
        self.comm_range_C3 = comm_range_C3
        self.min_comm_loss = min_comm_loss
        self.max_comm_loss = max_comm_loss

        self.fc = nn.ModuleDict({
            'C1': nn.Linear(in_dim['C1'], out_dim['C1'] * num_heads),
            'C2': nn.Linear(in_dim['C2'], out_dim['C2'] * num_heads),
            'C3': nn.Linear(in_dim['C3'], out_dim['C3'] * num_heads),
            'c1c1': nn.Linear(in_dim['C1'], out_dim['C1'] * num_heads),
            'c1c2': nn.Linear(in_dim['C1'], out_dim['C2'] * num_heads),
            'c1c3': nn.Linear(in_dim['C1'], out_dim['C3'] * num_heads),
            'c2c1': nn.Linear(in_dim['C2'], out_dim['C1'] * num_heads),
            'c2c2': nn.Linear(in_dim['C2'], out_dim['C2'] * num_heads),
            'c2c3': nn.Linear(in_dim['C2'], out_dim['C3'] * num_heads),
            'c3c1': nn.Linear(in_dim['C3'], out_dim['C1'] * num_heads),
            'c3c2': nn.Linear(in_dim['C3'], out_dim['C2'] * num_heads),
            'c3c3': nn.Linear(in_dim['C3'], out_dim['C3'] * num_heads),
            'c1s': nn.Linear(in_dim['C1'], out_dim['state'] * num_heads),
            'c2s': nn.Linear(in_dim['C2'], out_dim['state'] * num_heads),
            'c3s': nn.Linear(in_dim['C3'], out_dim['state'] * num_heads),
            'in': nn.Linear(in_dim['state'], out_dim['state'] * num_heads)
            })

        self.softmax = nn.Softmax(dim=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=l_alpha)

        self.use_relu = use_relu
        if self.use_relu:
            self.relu = nn.ReLU()

        # attention coefficients
        self.c1c1_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C1'])))
        self.c1c1_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C1'])))
        self.c1c2_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C2'])))
        self.c1c2_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C2'])))
        self.c1c3_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C3'])))
        self.c1c3_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C3'])))
        self.c2c1_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C1'])))
        self.c2c1_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C1'])))
        self.c2c2_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C2'])))
        self.c2c2_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C2'])))
        self.c2c3_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C3'])))
        self.c2c3_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C3'])))
        self.c3c1_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C1'])))
        self.c3c1_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C1'])))
        self.c3c2_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C2'])))
        self.c3c2_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C2'])))
        self.c3c3_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C3'])))
        self.c3c3_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['C3'])))

        # state node
        self.c1s_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
        self.c1s_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
        self.c2s_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
        self.c2s_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
        self.c3s_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
        self.c3s_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters
        References
            1 https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html
            2 https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
        """
        gain = nn.init.calculate_gain('relu')

        # attention
        nn.init.xavier_normal_(self.c1c1_src, gain=gain)
        nn.init.xavier_normal_(self.c1c1_dst, gain=gain)
        nn.init.xavier_normal_(self.c1c2_src, gain=gain)
        nn.init.xavier_normal_(self.c1c2_dst, gain=gain)
        nn.init.xavier_normal_(self.c1c3_src, gain=gain)
        nn.init.xavier_normal_(self.c1c3_dst, gain=gain)
        nn.init.xavier_normal_(self.c2c1_src, gain=gain)
        nn.init.xavier_normal_(self.c2c1_dst, gain=gain)
        nn.init.xavier_normal_(self.c2c2_src, gain=gain)
        nn.init.xavier_normal_(self.c2c2_dst, gain=gain)
        nn.init.xavier_normal_(self.c2c3_src, gain=gain)
        nn.init.xavier_normal_(self.c2c3_dst, gain=gain)
        nn.init.xavier_normal_(self.c3c1_src, gain=gain)
        nn.init.xavier_normal_(self.c3c1_dst, gain=gain)
        nn.init.xavier_normal_(self.c3c2_src, gain=gain)
        nn.init.xavier_normal_(self.c3c2_dst, gain=gain)
        nn.init.xavier_normal_(self.c3c3_src, gain=gain)
        nn.init.xavier_normal_(self.c3c3_dst, gain=gain)

        nn.init.xavier_normal_(self.c1s_src, gain=gain)
        nn.init.xavier_normal_(self.c1s_dst, gain=gain)
        nn.init.xavier_normal_(self.c2s_src, gain=gain)
        nn.init.xavier_normal_(self.c2s_dst, gain=gain)
        nn.init.xavier_normal_(self.c3s_src, gain=gain)
        nn.init.xavier_normal_(self.c3s_dst, gain=gain)

    '''Get amount of loss in communication.'''

    def get_comm_loss(self, dist, comm_range):
        comm_loss = torch.clamp(
            dist / comm_range * (self.max_comm_loss - self.min_comm_loss) + self.min_comm_loss,
            max=self.max_comm_loss)

        comm_loss[comm_loss != comm_loss] = self.min_comm_loss

        return comm_loss

    def lossy_u_mul_e(self, msg_type):
        m_type = 'm_' + msg_type
        h_type = 'h_' + msg_type
        a_type = 'a_' + msg_type

        sender_type = msg_type[0].upper()
        receiver_type = msg_type[-1].upper()

        Wh_self = 'Wh_' + receiver_type

        if msg_type == 'c1c1':
            src, dst = self.c1c1_src, self.c1c1_dst
        elif msg_type == 'c1c2':
            src, dst = self.c1c2_src, self.c1c2_dst
        elif msg_type == 'c1c3':
            src, dst = self.c1c3_src, self.c1c3_dst
        elif msg_type == 'c2c1':
            src, dst = self.c2c1_src, self.c2c1_dst
        elif msg_type == 'c2c2':
            src, dst = self.c2c2_src, self.c2c2_dst
        elif msg_type == 'c2c3':
            src, dst = self.c2c3_src, self.c2c3_dst
        elif msg_type == 'c3c1':
            src, dst = self.c3c1_src, self.c3c1_dst
        elif msg_type == 'c3c2':
            src, dst = self.c3c2_src, self.c3c2_dst
        elif msg_type == 'c3c3':
            src, dst = self.c3c3_src, self.c3c3_dst

        if sender_type == 'C1':
            commm_range = self.comm_range_C1
        elif sender_type == 'C2':
            comm_range = self.comm_range_C2
        elif sender_type == 'C3':
            comm_range = self.comm_range_C3

        def inner_lossy_u_mul_e(edges):
            dist = edges.data['dist']
            comm_loss = self.get_comm_loss(dist, comm_range)
            lossy_h = edges.src[h_type].clone()

            loss_count = torch.ceil(comm_loss * lossy_h.shape[1]).to(dtype=torch.int)

            if torch.sum(loss_count) > 0:
                loss_idx = np.range(lossy_h.shape)

                rand_mat = torch.rand(lossy_h.shape).cuda()
                top_loss_count = torch.topk(
                    rand_mat, torch.max(loss_count), largest=False)[0].cuda()
                stacked_loss_count = comm_loss.unsqueeze(dim=1).expand(
                    top_loss_count.shape).to(dtype=torch.int64)

                no_loss_mask = torch.where(loss_count == 0, 1, 0).unsqueeze(dim=1)
                stacked_loss_count[:, -1] = torch.clamp(loss_count - 1, min=0)

                loss_count_quant = torch.gather(
                    top_loss_count, dim=1, index=stacked_loss_count
                )[:, -1].unsqueeze(dim=1) - no_loss_mask

                noise_mask = torch.where(rand_mat - loss_count_quant <= 0, 1, 0).cuda()

                mean = comm_loss.unsqueeze(dim=1).expand(lossy_h.shape)
                noise = torch.normal(mean=mean, std=0.3) * noise_mask

                lossy_h += noise

            # Ensure correct amount of msg elements changed
            # assert torch.all(torch.eq(
            #     torch.sum(torch.where(lossy_h - edges.src[h_type] != 0, 1, 0), dim=1),
            #     loss_count
            # ))

            lossy_Wh = self.fc[msg_type](lossy_h).view(-1, self._num_heads, self._out_dim[receiver_type])

            Attn_src = (lossy_Wh * src).sum(dim=-1).unsqueeze(-1)
            Attn_dst = (edges.dst[Wh_self] * dst).sum(dim=-1).unsqueeze(-1)

            lossy_attn = self.softmax(
                self.leaky_relu(Attn_src + Attn_dst))

            return {m_type: lossy_Wh * lossy_attn}

        return inner_lossy_u_mul_e

    def forward(self, g, feat_dict):
        '''
        From hi to Whi
        '''
        # feature of C1
        Whc1 = self.fc['C1'](feat_dict['C1']).view(-1, self._num_heads, self._out_dim['C1'])
        g.nodes['C1'].data['Wh_C1'] = Whc1

        # feature of C2
        Whc2 = self.fc['A'](feat_dict['C2']).view(-1, self._num_heads, self._out_dim['C2'])
        g.nodes['C2'].data['Wh_C2'] = Whc2

        # feature of C3
        Whc3 = self.fc['C3'](feat_dict['C3']).view(-1, self._num_heads, self._out_dim['C3'])
        g.nodes['C3'].data['Wh_C3'] = Whc3

        '''
        Feature transform for each edge type (communication channel)
        '''
        g.nodes['C1'].data['h_c1c1'] = feat_dict['C1']
        g.nodes['C1'].data['h_c1c2'] = feat_dict['C1']
        g.nodes['C2'].data['h_c2c1'] = feat_dict['C2']
        g.nodes['C2'].data['h_c2c2'] = feat_dict['C2']
        g.nodes['C3'].data['h_c3c1'] = feat_dict['C3']
        g.nodes['C3'].data['h_c3c3'] = feat_dict['C3']

        # for state-related edges
        Whc1s = self.fc['c1s'](feat_dict['C1']).view(-1, self._num_heads, self._out_dim['state'])
        g.nodes['C1'].data['Wh_c1s'] = Whc1s

        Whc2s = self.fc['c2s'](feat_dict['C2']).view(-1, self._num_heads, self._out_dim['state'])
        g.nodes['C2'].data['Wh_c2s'] = Whc2s

        Whc3s = self.fc['c2s'](feat_dict['C3']).view(-1, self._num_heads, self._out_dim['state'])
        g.nodes['C3'].data['Wh_c3s'] = Whc3s

        Whin = self.fc['in'](feat_dict['state'].double()).view(-1, self._num_heads, self._out_dim['state'])
        g.nodes['state'].data['Wh_in'] = Whin

        '''
        Attention computation on subgraphs
        '''
        # c1c1 message passing
        if g['c1c1'].number_of_edges() > 0:
            g['c1c1'].update_all(self.lossy_u_mul_e('c1c1'),
                                fn.sum('m_c1c1', 'ft_c1c1'))

        # c1c2 message passing
        if g['c1c2'].number_of_edges() > 0:
            g['c1c2'].update_all(self.lossy_u_mul_e('c1c2'),
                                fn.sum('m_c1c2', 'ft_c1c2'))
        
        # c1c3 message passing
        if g['c1c3'].number_of_edges() > 0:
            g['c1c3'].update_all(self.lossy_u_mul_e('c1c3'),
                                fn.sum('m_c1c3', 'ft_c1c3'))

        # c2c1 message passing
        if g['c2c1'].number_of_edges() > 0:
            g['c2c1'].update_all(self.lossy_u_mul_e('c2c1'),
                                fn.sum('m_c2c1', 'ft_c2c1'))

        # c2c2 message passing
        if g['c2c2'].number_of_edges() > 0:
            g['c2c2'].update_all(self.lossy_u_mul_e('c2c2'),
                                fn.sum('m_c2c2', 'ft_c2c2'))
            
        #c2c3 message passing
        if g['c2c3'].number_of_edges() > 0:
            g['c2c3'].update_all(self.lossy_u_mul_e('c2c3'),
                                fn.sum('m_c2c3', 'ft_c2c3'))
        
        #c3c1 message passing
        if g['c3c1'].number_of_edges() > 0:
            g['c3c1'].update_all(self.lossy_u_mul_e('c3c1'),
                                fn.sum('m_c3c1', 'ft_c3c1'))
        
        #c3c2 message passing
        if g['c3c2'].number_of_edges() > 0:
            g['c3c2'].update_all(self.lossy_u_mul_e('c3c2'),
                                fn.sum('m_c3c2', 'ft_c3c2'))
            
        #c3c3 message passing
        if g['c3c3'].number_of_edges() > 0:
            g['c3c3'].update_all(self.lossy_u_mul_e('c3c3'),
                                fn.sum('m_c3c3', 'ft_c3c3'))

        # c1s
        Attn_src_c1s = (Whc1s * self.c1s_src).sum(dim=-1).unsqueeze(-1)
        Attn_dst_c1s = (Whin * self.c1s_dst).sum(dim=-1).unsqueeze(-1)

        g['c1s'].srcdata.update({'Attn_src_c1s': Attn_src_c1s})
        g['c1s'].dstdata.update({'Attn_dst_c1s': Attn_dst_c1s})

        g['c1s'].apply_edges(fn.u_add_v('Attn_src_c1s', 'Attn_dst_c1s', 'e_c1s'))
        e_c1s = self.leaky_relu(g['c1s'].edata.pop('e_c1s'))

        # compute softmax
        g['c1s'].edata['a_c1s'] = edge_softmax(g['c1s'], e_c1s)

        # message passing
        g['c1s'].update_all(fn.u_mul_e('Wh_c1s', 'a_c1s', 'm_c1s'),
                            fn.sum('m_c1s', 'ft_c1s'))

        # c2s
        Attn_src_c2s = (Whc2s * self.c2s_src).sum(dim=-1).unsqueeze(-1)
        Attn_dst_c2s = (Whin * self.c2s_dst).sum(dim=-1).unsqueeze(-1)

        g['c2s'].srcdata.update({'Attn_src_c2s': Attn_src_c2s})
        g['c2s'].dstdata.update({'Attn_dst_c2s': Attn_dst_c2s})

        g['c2s'].apply_edges(fn.u_add_v('Attn_src_c2s', 'Attn_dst_c2s', 'e_c2s'))
        e_c2s = self.leaky_relu(g['c2s'].edata.pop('e_c2s'))

        # compute softmax
        g['c2s'].edata['a_c2s'] = edge_softmax(g['c2s'], e_c2s)

        # message passing
        g['c2s'].update_all(fn.u_mul_e('Wh_c2s', 'a_c2s', 'm_c2s'),
                            fn.sum('m_c2s', 'ft_c2s'))
        
        # c3s
        Attn_src_c3s = (Whc3s * self.c3s_src).sum(dim=-1).unsqueeze(-1)
        Attn_dst_c3s = (Whin * self.c3s_dst).sum(dim=-1).unsqueeze(-1)

        g['c3s'].srcdata.update({'Attn_src_c3s': Attn_src_c3s})
        g['c3s'].dstdata.update({'Attn_dst_c3s': Attn_dst_c3s})

        g['c3s'].apply_edges(fn.u_add_v('Attn_src_c3s', 'Attn_dst_c3s', 'e_c3s'))
        e_c3s = self.leaky_relu(g['c3s'].edata.pop('e_c3s'))

        # compute softmax
        g['c3s'].edata['a_c3s'] = edge_softmax(g['c3s'], e_c3s)

        # message passing
        g['c3s'].update_all(fn.u_mul_e('Wh_c3s', 'a_c3s', 'm_c3s'),
                            fn.sum('m_c3s', 'ft_c3s'))

        '''
        Combine features from subgraphs
            Sum up to hi'
            Need to check if subgraph is valid
        '''
        # new feature of C1
        Whc1_new = g.nodes['C1'].data['Wh_C1'].clone()

        if g['c1c1'].number_of_edges() > 0:
            Whc1_new += g.nodes['C1'].data['ft_c1c1']
        if g['c2c1'].number_of_edges() > 0:
            Whc1_new += g.nodes['C1'].data['ft_c2c1']
        if g['c3c1'].number_of_edges() > 0:
            Whc1_new += g.nodes['C1'].data['ft_c3c1']

        g.nodes['C1'].data['h'] = Whc1_new

        # new feature of C2
        Whc2_new = g.nodes['C2'].data['Wh_C2'].clone()
        if g['c1c2'].number_of_edges() > 0:
            Whc2_new += g.nodes['C2'].data['ft_c1c2']
        if g['c2c2'].number_of_edges() > 0:
            Whc2_new += g.nodes['C2'].data['ft_c2c2']
        if g['c3c2'].number_of_edges() > 0:
            Whc2_new += g.nodes['C2'].data['ft_c3c2']

        g.nodes['C2'].data['h'] = Whc2_new

        # new features of C3
        Whc3_new = g.nodes['C3'].data['Wh_C3'].clone()
        if g['c1c3'].number_of_edges() > 0:
            Whc3_new += g.nodes['C3'].data['ft_c1c3']
        if g['c2c3'].number_of_edges() > 0:
            Whc3_new += g.nodes['C3'].data['ft_c2c3']
        if g['c3c3'].number_of_edges() > 0:
            Whc3_new += g.nodes['C3'].data['ft_c3c3']

        g.nodes['C3'].data['h'] = Whc3_new

        # new feature of state depending on if class 3 is valid
        if g['c3s'].number_of_edges() > 0:
            Whstate_new = g.nodes['state'].data['Wh_in'] + g.nodes['state'].data['ft_c1s'] +g.nodes['state'].data['ft_c2s']
        else:   
            Whstate_new = g.nodes['state'].data['Wh_in'] + g.nodes['state'].data['ft_c1s'] +g.nodes['state'].data['ft_c2s'] + g.nodes['state'].data['ft_c3s']

        g.nodes['state'].data['h'] = Whstate_new

        # deal with relu activation
        if self.use_relu:
            return {ntype: self.relu(g.nodes[ntype].data['h']) for ntype in g.ntypes}
        else:
            return {ntype: g.nodes[ntype].data['h'] for ntype in g.ntypes}

# merge = 'cat' or 'avg'
class MultiHeteroGATLayerLossyReal(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat',
            comm_range_C1=-1, comm_range_C2=-1, comm_range_C3=-1, min_comm_loss=0, max_comm_loss=0.3):
        super(MultiHeteroGATLayerLossyReal, self).__init__()

        self._num_heads = num_heads
        self._merge = merge

        if self._merge == 'cat':
            self.gat_conv = HeteroGATLayerLossyReal(in_dim, out_dim, num_heads,
                comm_range_C1=comm_range_C1, comm_range_C2=comm_range_C2, comm_range_C3=comm_range_C3,
                min_comm_loss=min_comm_loss, max_comm_loss=max_comm_loss)
        else:
            self.gat_conv = HeteroGATLayerLossyReal(in_dim, out_dim, num_heads,
                                                    use_relu=False, comm_range_C1=comm_range_C1,
                                                    comm_range_C2=comm_range_C2, comm_range_C3=comm_range_C3,
                                                    min_comm_loss=min_comm_loss, max_comm_loss=max_comm_loss)

    def forward(self, g, feat_dict):
        tmp = self.gat_conv(g, feat_dict)
        results = {}

        if self._merge == 'cat':
            # concat on the output feature dimension (dim=1)
            for ntype in g.ntypes:
                results[ntype] = torch.flatten(tmp[ntype], 1)
        else:
            # merge using average
            for ntype in g.ntypes:
                results[ntype] = torch.mean(tmp[ntype], 1)

        return results