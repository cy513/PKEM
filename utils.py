import torch
import dgl

def load_quadruples(inpath):
    with open(inpath, 'r') as f:
        quadrupleList = []
        for line in f:
            try:
                line_split = line.split()
                head = int(line_split[0])
                rel = int(line_split[1])
                tail = int(line_split[2])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
            except:
                print(line)
    return quadrupleList


def build_static_graph(num_nodes, triples, device):

    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm

    src, rel, dst = triples.transpose()
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(dst, src)
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.edata['type'] = torch.LongTensor(rel)
    g = g.to(device)
    return g


def get_entity_attribute_triplets(filename):
    l = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            triplet = line.strip().split('\t')
            s = int(triplet[0])
            r = int(triplet[1])
            o = int(triplet[2])
            l.append([s, r, o])
    return l

def get_stat_data(input_path='stat.txt'):
    stat = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            stat = line.split()
    return int(stat[0]), int(stat[1])