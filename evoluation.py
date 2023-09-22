import torch


def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

def calc_raw_mrr(score, labels, hits=[]):
    with torch.no_grad():
        ranks = sort_and_rank(score, labels)
        ranks += 1
        mrr = torch.mean(1.0 / ranks.float())
        hits1 = torch.mean((ranks <= hits[0]).float())
        hits3 = torch.mean((ranks <= hits[1]).float())
        hits10 = torch.mean((ranks <= hits[2]).float())

    return mrr.item(), hits1.item(), hits3.item(), hits10.item()