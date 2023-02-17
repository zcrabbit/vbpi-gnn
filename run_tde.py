import argparse
import os

from utils import namenum
from nce import TDE
import numpy as np
import dill
import datetime

parser = argparse.ArgumentParser()

######### Model arguments
parser.add_argument('--hdim', type=int, default=100, help='hidden dimension for node embedding net')
parser.add_argument('--hL', type=int, default=2, help='number of hidden layers for node embedding net')
parser.add_argument('--gnn_type', type=str, default='gcn', help='gcn | sage | gin | ggnn')
parser.add_argument('--aggr', type=str, default='sum', help='sum | mean | max')


######### Optimizer arguments
parser.add_argument('--stepsz', type=float, default=0.001, help=' stepsz parameters ')
parser.add_argument('--maxIter', type=int, default=200000, help=' number of iterations for training, default=200000')
parser.add_argument('--batch_size', type=int, default=20, help='batch_size for gradient based optimization, default=10')
parser.add_argument('--ar', type=float, default=0.75, help='step size anneal rate, default=0.75')
parser.add_argument('--af', type=int, default=10000, help='step size anneal frequency, default=20000')
parser.add_argument('--tf', type=int, default=1000, help='monitor frequency during training, default=1000')
parser.add_argument('--klf', type=int, default=5000, help='kl divergence and nce loss test frequency, default=5000')


args = parser.parse_args()

args.result_folder = 'results/tde_simulation' 
if not os.path.exists(args.result_folder):
    os.makedirs(args.result_folder)
    
args.save_to_path = args.result_folder + '/' + 'batch_size_' + str(args.batch_size) + '_hdim_' + str(args.hdim) + '_hL_' + str(args.hL)
    
args.save_to_path = args.save_to_path + '_' + args.gnn_type + '_' + args.aggr

args.save_to_path = args.save_to_path + '_' + str(datetime.datetime.now()) + '.pt'

print('Training with the following settings: {}'.format(args))

taxa = list('ABCDEFGH')
with open('data/simulation/simulation_emp_tree_freq.dill', 'rb') as readin:
    emp_tree_freq = dill.load(readin, encoding="latin1")
    
trees, wts = zip(*emp_tree_freq.items())
for tree in trees:
    namenum(tree, taxa)

wts = np.array(wts)

model = TDE(trees, taxa, wts, hidden_dim=args.hdim, num_layers=args.hL, gnn_type=args.gnn_type, aggr=args.aggr)

print('Parameter Info:')
for param in model.parameters():
    print(param.dtype, param.size())
    
print('\nNCE running, results will be saved to: {}\n'.format(args.save_to_path))
nce_loss, test_kl_div = model.nce(args.stepsz, maxiter=args.maxIter, test_freq=args.tf, kl_test_freq=args.klf, batch_size=args.batch_size,
                                  anneal_freq=args.af, anneal_rate=args.ar, save_to_path=args.save_to_path)
         
np.save(args.save_to_path.replace('.pt', '_nce_loss.npy'), nce_loss)
np.save(args.save_to_path.replace('.pt', '_kl_div.npy'), test_kl_div)