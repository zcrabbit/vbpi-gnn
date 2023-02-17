import torch
import numpy as np
from rateMatrix import *
import pdb

class PHY(object):
    nuc2vec = {'A':[1.,0.,0.,0.], 'G':[0.,1.,0.,0.], 'C':[0.,0.,1.,0.], 'T':[0.,0.,0.,1.],
           '-':[1.,1.,1.,1.], '?':[1.,1.,1.,1.], 'N':[1.,1.,1.,1.], 'R':[1.,1.,0.,0.],
           'Y':[0.,0.,1.,1.], 'S':[0.,1.,1.,0.], 'W':[1.,0.,0.,1.], 'K':[0.,1.,0.,1.],
           'M':[1.,0.,1.,0.], 'B':[0.,1.,1.,1.], 'D':[1.,1.,0.,1.], 'H':[1.,0.,1.,1.],
           'V':[1.,1.,1.,0.], '.':[1.,1.,1.,1.], 'U':[0.,0.,0.,1.]}
    
    def __init__(self, data, taxa, pden, subModel, scale=0.1, unique_site=True):
        self.ntips = len(data)
        self.nsites = len(data[0])
        self.taxa = taxa
        Qmodel, Qpara = subModel
        if Qmodel == "JC":
            self.D, self.U, self.U_inv, self.rateM = decompJC()
        if Qmodel == "HKY":
            self.D, self.U, self.U_inv, self.rateM = decompHKY(pden, Qpara)
        if Qmodel == "GTR":
            AG, AC, AT, GC, GT, CT = Qpara
            self.D, self.U, self.U_inv, self.rateM = decompGTR(pden, AG, AC, AT, GC, GT, CT)
        
        self.pden = torch.from_numpy(pden).float()
        self.D = torch.from_numpy(self.D).float()
        self.U = torch.from_numpy(self.U).float()
        self.U_inv = torch.from_numpy(self.U_inv).float()
        
        if unique_site:
            self.L, self.site_counts = map(torch.FloatTensor, self.initialCLV(data, unique_site=True))
        else:
            self.L, self.site_counts = torch.FloatTensor(self.initialCLV(data)), 1.0
        self.scale= scale
    
    def initialCLV(self, data, unique_site=False):
        if unique_site:
            data_arr = np.array(list(zip(*data)))
            unique_sites, counts = np.unique(data_arr, return_counts=True, axis=0)
            unique_data = unique_sites.T
            
            return [np.transpose([self.nuc2vec[c] for c in unique_data[i]]) for i in range(self.ntips)], counts
        else:
            return [np.transpose([self.nuc2vec[c] for c in data[i]]) for i in range(self.ntips)]

    def logprior(self, log_branch):
        return -torch.sum(torch.exp(log_branch)/self.scale + np.log(self.scale) - log_branch, -1)
    
    def loglikelihood(self, log_branch, tree):
        branch_D = torch.einsum("i,j->ij", (log_branch.exp(), self.D))
        transition_matrix = torch.matmul(torch.einsum("ij,kj->kij", (self.U, torch.exp(branch_D))), self.U_inv).clamp(0.0)
        scaler_list = []
        for node in tree.traverse("postorder"):
            if node.is_leaf():
                node.state = self.L[node.name].detach()
            else:
                node.state = 1.0
                for child in node.children:
                    node.state *= transition_matrix[child.name].mm(child.state)
                scaler = torch.sum(node.state, 0)
                node.state /= scaler
                scaler_list.append(scaler)

        scaler_list.append(torch.mm(self.pden.view(-1,4), tree.state).squeeze())
        logll = torch.sum(torch.log(torch.stack(scaler_list)) * self.site_counts)
        return logll
    
    def logp_joint(self, log_branch, tree):
        return self.logprior(log_branch) + self.loglikelihood(log_branch, tree)