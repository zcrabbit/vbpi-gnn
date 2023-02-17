import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import math
import random
import numpy as np
from utils import namenum
from base_branchModel import BaseModel
from gnn_branchModel import GNNModel
from vector_sbnModel import SBN
from phyloModel import PHY

import pdb


class VBPI(nn.Module):
    EPS = np.finfo(float).eps
    def __init__(self, taxa, rootsplit_supp_dict, subsplit_supp_dict, data, pden, subModel, emp_tree_freq=None,
                 scale=0.1, psp=True, feature_dim=2, hidden_dim=100, num_layers=1, branch_model='base', gnn_type='gcn', aggr='sum', project=False):
        super().__init__()
        self.taxa, self.emp_tree_freq = taxa, emp_tree_freq
        if emp_tree_freq:
            self.trees, self.emp_freqs = zip(*emp_tree_freq.items())
            self.emp_freqs = np.array(self.emp_freqs)
            self.negDataEnt = np.sum(self.emp_freqs * np.log(np.maximum(self.emp_freqs, self.EPS)))
        
        self.ntips = len(data)
        self.scale = scale
        self.phylo_model = PHY(data, taxa, pden, subModel, scale=scale)
        self.log_p_tau = - np.sum(np.log(np.arange(3, 2*self.ntips-3, 2)))
        
        self.tree_model = SBN(taxa, rootsplit_supp_dict, subsplit_supp_dict)
        self.rs_embedding_map, self.ss_embedding_map = self.tree_model.rs_map, self.tree_model.ss_map       
        
        if branch_model == 'base':
            self.branch_model = BaseModel(self.ntips, self.rs_embedding_map, self.ss_embedding_map, psp=psp, feature_dim=feature_dim)
        elif branch_model == 'gnn':
            self.branch_model = GNNModel(self.ntips, hidden_dim, num_layers=num_layers, gnn_type=gnn_type, aggr=aggr, project=project)
        else:
            raise NotImplementedError
        
        torch.set_num_threads(1)
    
    def load_from(self, state_dict_path):
        self.load_state_dict(torch.load(state_dict_path))
        self.eval()
        self.tree_model.update_CPDs()
                
    
    def kl_div(self):
        kl_div = 0.0
        for tree, wt in self.emp_tree_freq.items():
            kl_div += wt * np.log(max(np.exp(self.tree_model.loglikelihood(tree)), self.EPS))
        kl_div = self.negDataEnt - kl_div
        return kl_div
    
    def logq_tree(self, tree):
        return self.tree_model(tree)
    
    def lower_bound(self, n_particles=1, n_runs=1000):
        lower_bounds = []
        with torch.no_grad():
            for run in range(n_runs):
                samp_trees = [self.tree_model.sample_tree() for particle in range(n_particles)]
                [namenum(tree, self.taxa) for tree in samp_trees]    
                samp_log_branch, logq_branch = self.branch_model(samp_trees)

                logll = torch.stack([self.phylo_model.loglikelihood(log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
                logp_prior = self.phylo_model.logprior(samp_log_branch)
                logq_tree = torch.stack([self.logq_tree(tree) for tree in samp_trees])       
                lower_bounds.append(torch.logsumexp(logll + logp_prior - logq_tree - logq_branch + self.log_p_tau - math.log(n_particles), 0))            
            
            lower_bound = torch.stack(lower_bounds).mean()
            
        return lower_bound.item()
    
    def tree_lower_bound(self, tree, n_particles=1, n_runs=1000, name_to_num=True):
        lower_bounds = []
        if name_to_num:
            namenum(tree, self.taxa)
        with torch.no_grad():
            for run in range(n_runs):
                test_trees = [tree for particle in range(n_particles)]
                samp_log_branch, logq_branch = self.branch_model(test_trees)

                logll = torch.stack([self.phylo_model.loglikelihood(log_branch, test_tree) for log_branch, test_tree in zip(*[samp_log_branch, test_trees])])
                logp_prior = self.phylo_model.logprior(samp_log_branch)
                lower_bounds.append(torch.logsumexp(logll + logp_prior - logq_branch, 0) - math.log(n_particles))
                
            lower_bound = torch.stack(lower_bounds).mean()

        return lower_bound.item()
    
    def vimco_lower_bound(self, inverse_temp=1.0, n_particles=10):
        samp_trees = [self.tree_model.sample_tree() for particle in range(n_particles)]
        [namenum(tree, self.taxa) for tree in samp_trees]
        
        samp_log_branch, logq_branch = self.branch_model(samp_trees)
        
        logll = torch.stack([self.phylo_model.loglikelihood(log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
        logp_prior = self.phylo_model.logprior(samp_log_branch)
        logp_joint = inverse_temp * logll + logp_prior
        logq_tree = torch.stack([self.logq_tree(tree) for tree in samp_trees])
        lower_bound = torch.logsumexp(logll + logp_prior - logq_tree - logq_branch + self.log_p_tau - math.log(n_particles), 0)
        
        l_signal = logp_joint - logq_tree - logq_branch
        mean_exclude_signal = (torch.sum(l_signal) - l_signal) / (n_particles-1.)
        control_variates = torch.logsumexp(l_signal.view(-1,1).repeat(1, n_particles) - l_signal.diag() + mean_exclude_signal.diag() - math.log(n_particles), dim=0)
        temp_lower_bound = torch.logsumexp(l_signal - math.log(n_particles), dim=0)
        vimco_fake_term = torch.sum((temp_lower_bound - control_variates).detach() * logq_tree, dim=0)
        return temp_lower_bound, vimco_fake_term, lower_bound, torch.max(logll)
        
        
    def rws_lower_bound(self, inverse_temp=1.0, n_particles=10):
        samp_trees = [self.tree_model.sample_tree() for particle in range(n_particles)]
        [namenum(tree, self.taxa) for tree in samp_trees]
        logq_tree = torch.stack([self.logq_tree(tree) for tree in samp_trees])
        
        samp_log_branch, logq_branch = self.branch_model(samp_trees)
        logll = torch.stack([self.phylo_model.loglikelihood(log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
        logp_prior = self.phylo_model.logprior(samp_log_branch)
        logp_joint = inverse_temp * logll + logp_prior
        lower_bound = torch.logsumexp(logll + logp_prior - logq_tree - logq_branch + self.log_p_tau - math.log(n_particles), 0)
        
        l_signal = logp_joint - logq_tree.detach() - logq_branch
        temp_lower_bound = torch.logsumexp(l_signal - math.log(n_particles), dim=0)
        snis_wts = torch.softmax(l_signal, dim=0)
        rws_fake_term = torch.sum(snis_wts.detach() * logq_tree, dim=0)
        return temp_lower_bound, rws_fake_term, lower_bound, torch.max(logll)
        
    
    def learn(self, stepsz, maxiter=100000, test_freq=1000, lb_test_freq=5000, anneal_freq=20000, anneal_rate=0.75, n_particles=10,
              init_inverse_temp=0.001, warm_start_interval=50000, method='vimco', save_to_path=None):
        lbs, lls = [], []
        test_kl_div, test_lb = [], []
        
        if not isinstance(stepsz, dict):
            stepsz = {'tree': stepsz, 'branch': stepsz}
        
        optimizer = torch.optim.Adam([
                    {'params': self.tree_model.parameters(), 'lr':stepsz['tree']},
                    {'params': self.branch_model.parameters(), 'lr': stepsz['branch']}
                ])
        run_time = -time.time()
        for it in range(1, maxiter+1):
            inverse_temp = min(1., init_inverse_temp + it * 1.0/warm_start_interval)
            if method == 'vimco':
                temp_lower_bound, vimco_fake_term, lower_bound, logll = self.vimco_lower_bound(inverse_temp, n_particles)
                loss = - temp_lower_bound - vimco_fake_term
            elif method == 'rws':
                temp_lower_bound, rws_fake_term, lower_bound, logll = self.rws_lower_bound(inverse_temp, n_particles)
                loss = - temp_lower_bound - rws_fake_term
            else:
                raise NotImplementedError

            lbs.append(lower_bound.item())
            lls.append(logll.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.tree_model.update_CPDs()
            
            
            if it % test_freq == 0:
                run_time += time.time()
                if self.emp_tree_freq:
                    test_kl_div.append(self.kl_div())
                    print('Iter {}:({:.1f}s) Lower Bound: {:.4f} | Loglikelihood: {:.4f} | KL: {:.6f}'.format(it, run_time, np.mean(lbs), np.max(lls), test_kl_div[-1]), flush=True)
                else:
                    print('Iter {}:({:.1f}s) Lower Bound: {:.4f} | Loglikelihood: {:.4f}'.format(it, run_time, np.mean(lbs), np.max(lls)), flush=True)
                if it % lb_test_freq == 0:
                    run_time = -time.time()
                    test_lb.append(self.lower_bound(n_particles=1))
                    run_time += time.time()
                    print('>>> Iter {}:({:.1f}s) Test Lower Bound: {:.4f}'.format(it, run_time, test_lb[-1]), flush=True)
                    
                run_time = -time.time()
                lbs, lls = [], []
            
            if it % anneal_freq == 0:
                for g in optimizer.param_groups:
                    g['lr'] *= anneal_rate
        
        if save_to_path is not None:
            torch.save(self.state_dict(), save_to_path)
            
        return test_lb, test_kl_div