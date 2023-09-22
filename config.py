import argparse

args = argparse.ArgumentParser()
args.add_argument('--dataset', type=str, default='ICEWS18')
args.add_argument('--lr', type=float, default=0.001)
args.add_argument('--n_epochs', type=int, default=60)
args.add_argument('--hidden_dim', type=int, default=200)
args.add_argument('--gpu', type=int, default=1)
args.add_argument('--valid_epoch', type=int, default=0)
args.add_argument('--batch_size', type=int, default=1024)
args.add_argument('--joint_model', type=int, default=1)

args = args.parse_args()
print(args)
