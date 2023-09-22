import torch
from config import args
import utils
import models
from models import PKEM_Model
import numpy as np
import evoluation

device = torch.device('cuda' if args.gpu == 1 else 'cpu')
batch_size = args.batch_size

test_data = utils.load_quadruples('./data/{}/test.txt'.format(args.dataset))
num_ent, num_rel = utils.get_stat_data('./data/{}/stat.txt'.format(args.dataset))
time_interval = 24
if args.dataset == 'GDELT':
    time_interval = 15
num_timestamps = int(test_data[-1][3]/time_interval) + 1

static_triples = np.array(utils.get_entity_attribute_triplets('./data/{}/entity2attributes.txt'.format(args.dataset)))
num_attribute_types = len(np.unique(static_triples[:, 1]))
num_attribute_values = len(np.unique(static_triples[:, 2]))
static_triples[:, 2] = static_triples[:, 2] + num_ent
num_static_nodes = num_ent + num_attribute_values
static_graph = utils.build_static_graph(num_static_nodes, static_triples, device)

model = PKEM_Model(num_ent, num_rel, num_attribute_values, num_attribute_types, hidden_dim=args.hidden_dim, time_interval=time_interval, num_timestamps=num_timestamps)
model.to(device)
model.load_state_dict(torch.load('./models/PKEM_Model_{}.pt'.format(args.dataset)))
model.eval()
if args.joint_model == 1:
    model_ent = models.Entity_Linear(num_ent, args.hidden_dim)
    model_ent.load_state_dict(torch.load('./models/Entity_Linear_{}.pt'.format(args.dataset)))
    model_ent.to(device)
    model_ent.eval()
    model_rel = models.Relation_Linear(num_ent, num_rel, args.hidden_dim)
    model_rel.load_state_dict(torch.load('./models/Relation_Linear_{}.pt'.format(args.dataset)))
    model_rel.to(device)
    model_rel.eval()

num_test_samples = len(test_data)
test_array_data = np.asarray(test_data)
n_test_batch = (num_test_samples + batch_size - 1) // batch_size

mrr, hits1, hits3, hits10 = 0, 0, 0, 0
a, b = 0.4, 0.05
for idx in range(n_test_batch):
    batch_start = idx * batch_size
    batch_end = min(num_test_samples, (idx + 1) * batch_size)
    batch_data = test_array_data[batch_start: batch_end]
    labels = torch.LongTensor(batch_data[:, 2])
    score = model(static_graph, batch_data, device)
    if args.joint_model == 1:
        score = score + a * model_ent(batch_data) + b * model_rel(batch_data)
    tim_mrr, tim_hits1, tim_hits3, tim_hits10 = evoluation.calc_raw_mrr(score, labels.to(device), hits=[1, 3, 10])

    mrr += tim_mrr * len(batch_data)
    hits1 += tim_hits1 * len(batch_data)
    hits3 += tim_hits3 * len(batch_data)
    hits10 += tim_hits10 * len(batch_data)

mrr = mrr / test_array_data.shape[0]
hits1 = hits1 / test_array_data.shape[0]
hits3 = hits3 / test_array_data.shape[0]
hits10 = hits10 / test_array_data.shape[0]
print("MRR : {:.6f}".format(mrr))
print("Hits @ 1: {:.6f}".format(hits1))
print("Hits @ 3: {:.6f}".format(hits3))
print("Hits @ 10: {:.6f}".format(hits10))