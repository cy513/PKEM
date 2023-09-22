import dgl.function as fn
import torch
import torch.nn as nn


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, activation=None, self_loop=False, dropout=0.0, layer_norm=False):
        super(RGCNLayer, self).__init__()
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g):
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        self.propagate(g)
        node_repr = g.ndata['h']

        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.layer_norm:
            node_repr = self.normalization_layer(node_repr)
        if self.activation:
            node_repr = self.activation(node_repr)

        g.ndata['h'] = node_repr
        return node_repr


class RGCNBlockLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, activation=None, self_loop=False, dropout=0):
        super(RGCNBlockLayer, self).__init__(in_feat, out_feat, activation, self_loop=self_loop, dropout=dropout)
        self.num_rels = num_rels
        self.num_bases = num_bases

        assert self.num_bases > 0

        self.out_feat = out_feat
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges):
        weight = self.weight.index_select(0, edges.data['type']).view(-1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        return {'msg': msg}

    def propagate(self, g):
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'), self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


