import torch
import torch.nn as nn
from config import args
import utils
import models
from models import PKEM_Model
import numpy as np
import random
import evoluation

device = torch.device('cuda' if args.gpu == 1 else 'cpu')
batch_size = args.batch_size

train_data = utils.load_quadruples('./data/{}/train.txt'.format(args.dataset))
valid_data = utils.load_quadruples('./data/{}/valid.txt'.format(args.dataset))
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
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

num_train_samples = len(train_data)
train_array_data = np.asarray(train_data)
n_batch = (num_train_samples + batch_size - 1) // batch_size
num_valid_samples = len(valid_data)
valid_array_data = np.asarray(valid_data)
n_valid_batch = (num_valid_samples + batch_size - 1) // batch_size
best_mrr = 0

rd_idx = [_ for _ in range(num_train_samples)]
random.shuffle(rd_idx)

for epoch in range(args.n_epochs):
    model.train()
    for i in range(n_batch):
        optimizer.zero_grad()
        batch_start = i * batch_size
        batch_end = min(num_train_samples, (i + 1) * batch_size)
        train_batch_data = train_array_data[rd_idx[batch_start: batch_end]]
        labels = torch.LongTensor(train_batch_data[:, 2])
        score = model(static_graph, train_batch_data, device)
        loss = criterion(score, labels.to(device))
        loss.backward()
        optimizer.step()

    if epoch >= args.valid_epoch:
        model.eval()
        mrr, hits1, hits3, hits10 = 0, 0, 0, 0
        for i in range(n_valid_batch):
            batch_start = i * batch_size
            batch_end = min(num_valid_samples, (i + 1) * batch_size)
            valid_batch_data = valid_array_data[batch_start: batch_end]
            labels = torch.LongTensor(valid_batch_data[:, 2])
            score = model(static_graph, valid_batch_data, device)
            tim_mrr, tim_hits1, tim_hits3, tim_hits10 = evoluation.calc_raw_mrr(score, labels.to(device),
                                                                                hits=[1, 3, 10])

            mrr += tim_mrr * len(valid_batch_data)
            hits1 += tim_hits1 * len(valid_batch_data)
            hits3 += tim_hits3 * len(valid_batch_data)
            hits10 += tim_hits10 * len(valid_batch_data)

        mrr = mrr / valid_array_data.shape[0]
        if mrr > best_mrr:
            best_mrr = mrr
            print('epoch:{}, valid_mrr={}, Loss: {:.6f}'.format(epoch + 1, mrr, loss.item()))
            torch.save(model.state_dict(), './models/PKEM_Model_{}.pt'.format(args.dataset))

if args.joint_model == 1:
    print('start train entity linear model')

    model_ent = models.Entity_Linear(num_ent, args.hidden_dim)
    model_ent.to(device)
    optimizer = torch.optim.Adam(model_ent.parameters(), lr=args.lr)

    epoch_num = 8
    for epoch in range(epoch_num):
        model_ent.train()
        for i in range(n_batch):
            optimizer.zero_grad()
            batch_start = i * batch_size
            batch_end = min(num_train_samples, (i + 1) * batch_size)
            train_batch_data = train_array_data[batch_start: batch_end]
            labels = torch.LongTensor(train_batch_data[:, 2])
            score = model_ent(train_batch_data)
            loss = criterion(score, labels.to(device))
            loss.backward()
            optimizer.step()

        print('epoch:{}, Loss: {:.6f}'.format(epoch + 1, loss.item()))
    torch.save(model_ent.state_dict(), './models/Entity_Linear_{}.pt'.format(args.dataset))

    print('start train relation linear model')

    model_rel = models.Relation_Linear(num_ent, num_rel, args.hidden_dim)
    model_rel.to(device)
    optimizer = torch.optim.Adam(model_rel.parameters(), lr=args.lr)

    for epoch in range(epoch_num):
        model_rel.train()
        for i in range(n_batch):
            optimizer.zero_grad()
            batch_start = i * batch_size
            batch_end = min(num_train_samples, (i + 1) * batch_size)
            train_batch_data = train_array_data[batch_start: batch_end]
            labels = torch.LongTensor(train_batch_data[:, 2])
            score = model_rel(train_batch_data)
            loss = criterion(score, labels.to(device))
            loss.backward()
            optimizer.step()

        print('epoch:{}, Loss: {:.6f}'.format(epoch + 1, loss.item()))
    torch.save(model_rel.state_dict(), './models/Relation_Linear_{}.pt'.format(args.dataset))