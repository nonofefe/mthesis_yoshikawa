import argparse

from models import VGAE, VGAEmf
from train import LinkPredTrainer
from utils import LinkPredData, apply_mask, generate_mask,apply_neighbor_mean_recursive, apply_neighbor_mean
from miss_struct import MissStruct


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    default='cora',
                    choices=['cora', 'citeseer', 'amacomp', 'amaphoto'],
                    help='dataset name')
parser.add_argument('--type',
                    default='uniform',
                    choices=['uniform', 'bias', 'struct'],
                    help="uniform randomly missing, biased randomly missing, and structurally missing")
parser.add_argument('--rate', default=0.1, type=float, help='missing rate')
parser.add_argument('--nhid', default=32, type=int, help='the number of hidden units')
parser.add_argument('--latent_dim', default=16, type=int, help='the dimension of latent variables')
parser.add_argument('--dropout', default=0., type=float, help='dropout rate')
parser.add_argument('--ncomp', default=5, type=int, help='the number of Gaussian components')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--wd', default=0., type=float, help='weight decay')
parser.add_argument('--epoch', default=1000, type=int, help='the number of training epochs')
parser.add_argument('--verbose', action='store_true', help='verbose')
parser.add_argument('--split', default=1, type=int, help='the number of split units')
parser.add_argument('--rec', default=1, type=int, help='the number of split units')

args = parser.parse_args()

if __name__ == '__main__':
    data = LinkPredData(args.dataset)
    mask = generate_mask(data.features, args.rate, args.type)
    miss_struct = MissStruct(mask, data.adj)
    apply_mask(data.features, mask)

    apply_neighbor_mean_recursive(data.features, mask, miss_struct, data.adj)
    #apply_neighbor_mean(data.features, mask, miss_struct, data.adj)
    
    model = VGAE(data, nhid=args.nhid, latent_dim=args.latent_dim, dropout=args.dropout)
    params = {
        'lr': args.lr,
        'weight_decay': args.wd,
        'epochs': args.epoch,
    }
    trainer = LinkPredTrainer(data, model, params, niter=20, verbose=args.verbose)
    trainer.run()
