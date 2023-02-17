import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

                    
    
class BaseModel(nn.Module):
    """
    The base VBPI branch length model.
    Use psp to turn on/off the primary subsplit pair (PSP) parameterization. The Default is true.
    
    
    """
    def __init__(self, ntips, rootsplit_embedding_map, subsplit_embedding_map, psp=True, feature_dim=2, **kwargs):
        super().__init__()
        self.ntips = ntips
        self.embedding_map = rootsplit_embedding_map
        self.psp = psp
        self.feature_dim = feature_dim
        
        self.embedding_dim = len(self.embedding_map)
        if self.psp:
            for parent in subsplit_embedding_map:
                if parent[:self.ntips].count('1') + parent[self.ntips:].count('1') == self.ntips:
                    for child, i in subsplit_embedding_map[parent].items():
                        self.embedding_map[parent+child] = self.embedding_dim + i
                    self.embedding_dim += len(subsplit_embedding_map[parent])
        
        self.sx = nn.Parameter(torch.zeros(self.embedding_dim, self.feature_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.sx.data)         
        # self.padding_dim = self.embedding_dim
        self.padding_dim = -1
    
    def pad_feature(self):
        self.feature_padded = torch.cat((self.sx, torch.zeros(1, self.feature_dim)), dim=0)              
    
    def grab_geo_idxes(self, tree):
        neigh_ss_idxes = []
        idx_list = []
        
        for node in tree.traverse("postorder"):
            if not node.is_root():
                neigh_ss_idx = []
                neigh_ss_idx.append(self.embedding_map[node.split_bitarr])
                
                if self.psp:                
                    comb_parent_bipart_bitarr_root_to_leaf = node.clade_bitarr + ~node.clade_bitarr
                    comb_parent_bipart_bitarr_leaf_to_root = ~node.clade_bitarr + node.clade_bitarr
                    if node.up.is_root():
                        child_bipart_bitarr = min(sister.clade_bitarr for sister in node.get_sisters())
                    else:
                        child_bipart_bitarr = min([node.get_sisters()[0].clade_bitarr, ~node.up.clade_bitarr])
                    neigh_ss_idx.append(self.embedding_map[comb_parent_bipart_bitarr_root_to_leaf.to01() + child_bipart_bitarr.to01()])
                    
                    if not node.is_leaf():
                        child_bipart_bitarr = min(child.clade_bitarr for child in node.children)
                        neigh_ss_idx.append(self.embedding_map[comb_parent_bipart_bitarr_leaf_to_root.to01() + child_bipart_bitarr.to01()])
                    else:
                        neigh_ss_idx.append(self.padding_dim)
                        
                neigh_ss_idxes.append(neigh_ss_idx)
                idx_list.append(node.name)
        
        return neigh_ss_idxes, idx_list
        
    
    def mean_std(self, tree, return_adj_matrix=False):
        neigh_ss_idxes, idx_list = self.grab_geo_idxes(tree)

        neigh_ss_idxes = torch.LongTensor(neigh_ss_idxes)
        branch_idx_map = torch.sort(torch.LongTensor(idx_list), dim=0, descending=False)[1]
        mean_std = torch.index_select(self.feature_padded[neigh_ss_idxes].sum(1), 0, branch_idx_map)
        
        if not return_adj_matrix:
            return mean_std[:, 0], mean_std[:, 1]
        else:
            return mean_std[:, 0], mean_std[:, 1], neigh_ss_idxes[branch_idx_map]
    
    
    def sample_branch_base(self, n_particles):
        samp_log_branch = torch.randn(n_particles, 2*self.ntips-3)
        return samp_log_branch, torch.sum(-0.5*math.log(2*math.pi) - 0.5*samp_log_branch**2, -1)
    
    
    def forward(self, tree_list):
        self.pad_feature()
        mean, std = zip(*map(lambda x: self.mean_std(x), tree_list))
        mean, std = torch.stack(mean, dim=0), torch.stack(std, dim=0)
        samp_log_branch, logq_branch = self.sample_branch_base(len(tree_list))
        samp_log_branch, logq_branch = samp_log_branch * std.exp() + mean - 2.0, logq_branch - torch.sum(std, -1)
        return samp_log_branch, logq_branch       