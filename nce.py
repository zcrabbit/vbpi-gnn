import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import math
import numpy as np


from gnn_topologyModel import GNNModel



class TDE(nn.Module):
    EPS = np.finfo(float).eps
    def __init__(self, trees, taxa, emp_freq, hidden_dim=50, num_layers=1, gnn_type='gcn', aggr='sum'):
        super().__init__()
        self.trees = trees
        self.num_trees = len(trees)
        self.emp_freq = emp_freq
        self.ntips = len(taxa)
        self.negDataEnt = np.sum(self.emp_freq * np.log(np.maximum(self.emp_freq, self.EPS)))
        
        self.model = GNNModel(self.ntips, hidden_dim, num_layers=num_layers, gnn_type=gnn_type, aggr=aggr)
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        
        torch.set_num_threads(1)
        
#         self.uniform_dist_prob = np.ones(len(self.trees))/len(self.trees)

    def load_from(self, state_dict_path):
        self.load_state_dict(torch.load(state_dict_path))
        self.eval()

    def nce_loss(self, emp_freq=False):
        if emp_freq:
            prob_ratio = np.maximum(self.emp_freq / (self.emp_freq + 1./self.num_trees), self.EPS)
        else:
            with torch.no_grad():
                logits = self.batch_logits(self.trees).squeeze()
            prob_ratio = torch.sigmoid(logits).numpy()
        
        return - np.sum(self.emp_freq * np.log(prob_ratio)) - np.mean(np.log(1-prob_ratio))


    def kl_div(self):
        estimated_prob = torch.stack([self.estimated_logp(tree) for tree in self.trees])
        estimated_prob = torch.softmax(estimated_prob, dim=0)
        kl_div = self.negDataEnt - np.sum(self.emp_freq * np.log(estimated_prob.clamp(self.EPS).numpy()))
        
        return kl_div
        
    def sample_tree(self, batch_size=10, mode='data'):
        if mode == 'data':
            return np.random.choice(self.trees, batch_size, p=self.emp_freq)
        elif mode == 'noise':
            return np.random.choice(self.trees, batch_size)
        
    def batch_logits(self, tree_list):
        return torch.cat([self.model(tree) for tree in tree_list], dim=0)
    
    def estimated_logp(self, tree):
        with torch.no_grad():
            logp = self.model(tree).squeeze() - math.log(self.num_trees)
        return logp
        
    def nce(self, stepsz, maxiter=200000, batch_size=10, test_freq=1000, kl_test_freq=5000, anneal_freq=20000, anneal_rate=0.75, save_to_path=None):
        test_kl_div, NCE_loss = [], []
        optimizer = torch.optim.Adam(self.parameters(), lr=stepsz)
        loss = []
        
        run_time = -time.time()
        for it in range(1, maxiter+1):
            valid, fake = torch.ones(batch_size, 1), torch.zeros(batch_size, 1)
            samp_data = self.sample_tree(batch_size, 'data')
            samp_noise = self.sample_tree(batch_size, 'noise')
                       
            nce_loss = 0.5*(self.adversarial_loss(self.batch_logits(samp_data), valid) + self.adversarial_loss(self.batch_logits(samp_noise), fake))
            loss.append(nce_loss.item())
            
            optimizer.zero_grad()
            nce_loss.backward()
            optimizer.step()
            
            if it % test_freq == 0:
                run_time += time.time()
                print('Iter {}:({:.1f}s) NCE Loss {:.4f}'.format(it, run_time, np.mean(loss)))               
                
                if it % kl_test_freq == 0:
                    run_time = -time.time()
                    test_kl_div.append(self.kl_div())
                    NCE_loss.append(self.nce_loss())
                    run_time += time.time()
                    print('>>> Iter {}:({:.1f}s) NCE Loss {:.4f} | KL {:.6f}'.format(it, run_time, NCE_loss[-1], test_kl_div[-1]))
                
                run_time = -time.time()
                loss = []             
                
            if it % anneal_freq == 0:
                for g in optimizer.param_groups:
                    g['lr'] *= anneal_rate
        
        if save_to_path is not None:
            torch.save(self.state_dict(), save_to_path)
                           
        return NCE_loss, test_kl_div