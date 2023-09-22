
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import RGCNBlockLayer
import math

class PKEM_Model(nn.Module):
    def __init__(self, num_ent, num_rel, num_attr_values, num_attr_types, hidden_dim, time_interval, num_timestamps, event_related=False, period=7, tranx=0.1, ampy=0.005):
        super(PKEM_Model, self).__init__()

        self.period = period
        self.ampy = ampy
        self.tranx = tranx
        self.num_ent = num_ent
        self.num_timestamps = num_timestamps
        self.time_interval = time_interval
        self.event_related = event_related

        self.rel_emb = nn.Parameter(torch.Tensor(num_rel, hidden_dim), requires_grad=True).float()
        self.ent_emb = nn.Parameter(torch.Tensor(num_ent, hidden_dim), requires_grad=True).float()
        self.attr_emb = nn.Parameter(torch.Tensor(num_attr_values, hidden_dim), requires_grad=True).float()
        self.time_emb = nn.Parameter(torch.Tensor(num_timestamps, 1), requires_grad=False).float()

        self.static_rgcn_layer = RGCNBlockLayer(hidden_dim, hidden_dim, num_attr_types, num_bases=200, activation=F.rrelu, dropout=0.2, self_loop=False)
        self.decoder = Decoder(hidden_dim)
        self.linear = nn.Linear(hidden_dim * 2, num_ent)

        self.amps_linear = nn.Linear(hidden_dim * 2, 1)
        self.freq_linear = nn.Linear(hidden_dim * 2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.rel_emb, gain=nn.init.calculate_gain('relu'))
        nn.init.normal_(self.ent_emb)
        nn.init.xavier_normal_(self.attr_emb)

        if self.event_related is False:
            for t in range(self.num_timestamps):
                self.time_emb[t] = self.ampy * (math.sin(2 * math.pi * t / self.period) + 1) + self.tranx


    def get_embeddings(self, batch_data, static_emb, device):
        ent_idx = batch_data[:, 0]
        rel_idx = batch_data[:, 1]
        tim_idx = batch_data[:, 3] / self.time_interval

        ent = static_emb[ent_idx]
        rel = self.rel_emb[rel_idx]

        if self.event_related is True:
            amps = self.amps_linear(torch.cat((ent, rel), dim=1))
            amps = F.relu(amps)
            freq = self.freq_linear(torch.cat((ent, rel), dim=1))
            freq = F.relu(freq)
            timestamp = torch.FloatTensor(tim_idx).to(device)
            tim = amps * (torch.sin(freq * timestamp.unsqueeze(1)) + 1) + self.tranx
        else:
            tim = self.time_emb[tim_idx]

        return ent, rel, tim

    def forward(self, static_graph, batch_data, device):
        static_graph = static_graph.to(device)
        static_graph.ndata['h'] = torch.cat((self.ent_emb, self.attr_emb), dim=0)
        self.static_rgcn_layer(static_graph)
        static_emb = static_graph.ndata.pop('h')[:self.num_ent, :]

        ent_emb, rel_emb, tim_emb = self.get_embeddings(batch_data, static_emb, device)
        out = self.decoder.forward(ent_emb, rel_emb, tim_emb, static_emb).view(-1, self.num_ent)

        return out


class Decoder(torch.nn.Module):

    def __init__(self, embedding_dim, dropout=0.2):
        super(Decoder, self).__init__()

        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(embedding_dim * 2 + 1, embedding_dim)

    def forward(self, ent_emb, rel_emb, tim_emb, all_ent_emb):
        ent_emb = F.tanh(ent_emb)
        x = torch.cat([ent_emb, rel_emb, tim_emb], 1)
        x = self.linear(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = torch.mm(x, all_ent_emb.transpose(1, 0))
        return x

class Entity_Linear(nn.Module):
    def __init__(self, num_ent, hidden_dim):
        super(Entity_Linear, self).__init__()
        self.ent_emb = nn.Parameter(torch.Tensor(num_ent, hidden_dim))
        self.linear = nn.Linear(hidden_dim, num_ent)
        nn.init.xavier_uniform_(self.ent_emb, gain=nn.init.calculate_gain('relu'))

    def forward(self, batch_data):
        ent_emb = self.ent_emb[batch_data[:, 0]]
        out = self.linear(ent_emb)
        return out

class Relation_Linear(nn.Module):
    def __init__(self, num_ent, num_rel, hidden_dim):
        super(Relation_Linear, self).__init__()
        self.rel_emb = nn.Parameter(torch.Tensor(num_rel, hidden_dim))
        self.linear = nn.Linear(hidden_dim, num_ent)
        nn.init.xavier_uniform_(self.rel_emb, gain=nn.init.calculate_gain('relu'))

    def forward(self, batch_data):
        rel_emb = self.rel_emb[batch_data[:, 1]]
        out = self.linear(rel_emb)
        return out

