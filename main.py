import argparse
import os

from dataManipulation import *
from utils import summary, summary_raw, mcmc_treeprob, get_support_from_mcmc, BitArray, tree_process
from vbpi import VBPI
import time
import numpy as np
import datetime
 
parser = argparse.ArgumentParser()

######### Data arguments
parser.add_argument('--dataset', required=True, help=' DS1 | DS2 | DS3 | DS4 | DS5 | DS6 | DS7 | DS8 ')
parser.add_argument('--supportType', type=str, default='ufboot', help=' ufboot | mcmc ')
parser.add_argument('--empFreq', default=False, action='store_true', help='emprical frequence for KL computation')


######### Model arguments
parser.add_argument('--psp', default=False, action='store_true', help=' turn on psp branch length feature')
parser.add_argument('--nf', type=int, default=2, help=' branch length feature embedding dimension ')
parser.add_argument('--hdim', type=int, default=100, help='hidden dimension for node embedding net')
parser.add_argument('--hL', type=int, default=2, help='number of hidden layers for node embedding net')
parser.add_argument('--brlen_model', type=str, default='gnn', help='branch length models')
parser.add_argument('--gnn_type', type=str, default='gcn', help='gcn | sage | gin | ggnn')
parser.add_argument('--aggr', type=str, default='sum', help='sum | mean | max')
parser.add_argument('--proj', default=False, action='store_true', help='use projection first in SAGEConv')
parser.add_argument('--test', default=False, action='store_true', help='turn on the test mode')
parser.add_argument('--datetime', type=str, default='2022-01-01', help=' 2020-04-01 | 2020-04-02 | ...... ')


######### Optimizer arguments
parser.add_argument('--stepszTree', type=float, default=0.001, help=' step size for tree topology parameters ')
parser.add_argument('--stepszBranch', type=float, default=0.001, help=' stepsz for branch length parameters ')
parser.add_argument('--maxIter', type=int, default=200000, help=' number of iterations for training, default=400000')
parser.add_argument('--invT0', type=float, default=0.001, help=' initial inverse temperature for annealing schedule, default=0.001')
parser.add_argument('--nwarmStart', type=float, default=100000, help=' number of warm start iterations, default=100000')
parser.add_argument('--nParticle', type=int, default=10, help='number of particles for variational objectives, default=10')
parser.add_argument('--ar', type=float, default=0.75, help='step size anneal rate, default=0.75')
parser.add_argument('--af', type=int, default=20000, help='step size anneal frequency, default=20000')
parser.add_argument('--tf', type=int, default=1000, help='monitor frequency during training, default=1000')
parser.add_argument('--lbf', type=int, default=5000, help='lower bound test frequency, default=5000')
parser.add_argument('--gradMethod', type=str, default='vimco', help=' vimco | rws ')

args = parser.parse_args()

args.result_folder = 'results/' + args.dataset + '/' + args.brlen_model
if not os.path.exists(args.result_folder):
    os.makedirs(args.result_folder)

args.save_to_path = args.result_folder + '/' + args.supportType + '_' + args.gradMethod + '_' + str(args.nParticle)
if args.brlen_model == 'gnn':
    args.save_to_path = args.save_to_path + '_' + args.gnn_type + '_' + args.aggr
if args.psp:
    args.save_to_path = args.save_to_path + '_psp'

if args.proj:
    args.save_to_path = args.save_to_path + '_proj'
    
if args.test:
    args.load_from_path = args.save_to_path + '_' + args.datetime + '.pt'

args.save_to_path = args.save_to_path + '_' + str(datetime.datetime.now()) + '.pt'
 
if not args.test:
    print('Training with the following settings: {}'.format(args))
else:
    print('Testing with the following settings: {}'.format(args))


ufboot_support_path = 'data/ufboot_data_DS1-11/'
data_path = 'data/hohna_datasets_fasta/'
ground_truth_path, samp_size = 'data/raw_data_DS1-11/', 750001

###### Load Data
print('\nLoading Data set: {} ......'.format(args.dataset))
run_time = -time.time()

if args.supportType == 'ufboot':
    tree_dict_support, tree_names_support = summary_raw(args.dataset, ufboot_support_path)
elif args.supportType == 'mcmc':
    tree_dict_support, tree_names_support, _ = mcmc_treeprob(mcmc_support_path + args.dataset + '.trprobs', 'nexus', taxon='keep')

data, taxa = loadData(data_path + args.dataset + '.fasta', 'fasta')

run_time += time.time()
print('Support loaded in {:.1f} seconds'.format(run_time))

if args.empFreq:
    print('\nLoading empirical posterior estimates ......')
    run_time = -time.time()
    tree_dict_total, tree_names_total, tree_wts_total = summary(args.dataset, ground_truth_path, samp_size=samp_size)
    emp_tree_freq = {tree_dict_total[tree_name]:tree_wts_total[i] for i, tree_name in enumerate(tree_names_total)}
    run_time += time.time()
    print('Empirical estimates from MrBayes loaded in {:.1f} seconds'.format(run_time))
else:
    emp_tree_freq = None

rootsplit_supp_dict, subsplit_supp_dict = get_support_from_mcmc(taxa, tree_dict_support, tree_names_support)
del tree_dict_support, tree_names_support

model = VBPI(taxa, rootsplit_supp_dict, subsplit_supp_dict, data, pden=np.ones(4)/4., subModel=('JC', 1.0),
                 emp_tree_freq=emp_tree_freq, feature_dim=args.nf, psp=args.psp, hidden_dim=args.hdim, num_layers=args.hL, branch_model=args.brlen_model, gnn_type=args.gnn_type, aggr=args.aggr, project=args.proj)

print('Parameter Info:')
for param in model.parameters():
    print(param.dtype, param.size())
    
if not args.test:
    print('\nVBPI running, results will be saved to: {}\n'.format(args.save_to_path))
    test_lb, test_kl_div = model.learn({'tree':args.stepszTree,'branch':args.stepszBranch}, args.maxIter, test_freq=args.tf, n_particles=args.nParticle, anneal_freq=args.af, init_inverse_temp=args.invT0,
                 warm_start_interval=args.nwarmStart, method=args.gradMethod, save_to_path=args.save_to_path)
             
    np.save(args.save_to_path.replace('.pt', '_test_lb.npy'), test_lb)
    if args.empFreq:
        np.save(args.save_to_path.replace('.pt', '_kl_div.npy'), test_kl_div)
else:
    print('Loading parameters from: {}\n'.format(args.load_from_path))
    model.load_from(args.load_from_path)
    
    print('Computing one sample lower bounds\n')
    lower_bound_1_sample = np.array([model.lower_bound(n_particles=1, n_runs=1000) for i in range(100)])
    np.save(args.load_from_path.replace('.pt', '_lower_bound_1_' + str(datetime.datetime.now()) + '.npy'), lower_bound_1_sample)
    
    print('Computing ten sample lower bounds\n')
    lower_bound_10_sample = np.array([model.lower_bound(n_particles=10, n_runs=1000) for i in range(100)])
    np.save(args.load_from_path.replace('.pt', '_lower_bound_10_' + str(datetime.datetime.now()) + '.npy'), lower_bound_10_sample)

    print('Computing marginal loglikelihood\n')
    marginal_likelihood_est = np.array([model.lower_bound(n_particles=1000, n_runs=1) for i in range(1000)])
    np.save(args.load_from_path.replace('.pt', '_marginal_likelihood_est_' + str(datetime.datetime.now()) +'.npy'), marginal_likelihood_est)


    if args.empFreq:
        tree_ci_index = np.argsort(tree_wts_total)[::-1]
        print('Computing 95% confidence interval tree lower bound\n')
        lower_bound_ci = []
        toBitArr = BitArray(taxa)
        for i in tree_ci_index[:42]:
            test_tree = tree_dict_total[tree_names_total[i]].copy()
            tree_process(test_tree, toBitArr)
            lower_bound_ci.append(model.tree_lower_bound(test_tree, n_runs=10000))
    
        np.save(args.save_to_path.replace('.pt', '_tree_lower_bound.npy'), lower_bound_ci)
