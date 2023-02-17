import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

from gnnModels_slim import GNNStack, GatedGraphConv, IDConv, MeanStdPooling



class GNNModel(nn.Module):          
    def __init__(self, ntips, hidden_dim=100, num_layers=1, gnn_type='gcn', aggr='sum', project=False, bias=True, **kwargs):
        super().__init__()
        self.ntips = ntips
        self.leaf_features = torch.eye(self.ntips)
        
        if gnn_type == 'identity':
            self.gnn = IDConv()
        elif gnn_type != 'ggnn':
            self.gnn = GNNStack(self.ntips, hidden_dim, num_layers=num_layers, bias=bias, gnn_type=gnn_type, aggr=aggr, project=project)
        else:
            self.gnn = GatedGraphConv(hidden_dim, num_layers=num_layers, bias=bias)
            
        if gnn_type == 'identity':
            self.mean_std_net = MeanStdPooling(self.ntips, hidden_dim, bias=bias)
        else:
            self.mean_std_net = MeanStdPooling(hidden_dim, hidden_dim, bias=bias)
                
    
    def node_embedding(self, tree):
        for node in tree.traverse('postorder'):
            if node.is_leaf():
                node.c = 0
                node.d = self.leaf_features[node.name]
            else:
                child_c, child_d = 0., 0.
                for child in node.children:
                    child_c += child.c
                    child_d += child.d
                node.c = 1./(3. - child_c)
                node.d = node.c * child_d
        
        node_features, node_idx_list, edge_index = [], [], []            
        for node in tree.traverse('preorder'):
            neigh_idx_list = []
            if not node.is_root():
                node.d = node.c * node.up.d + node.d
                # parent_idx_list.append(node.up.name)
                neigh_idx_list.append(node.up.name)
                
                if not node.is_leaf():
                    neigh_idx_list.extend([child.name for child in node.children])
                else:
                    neigh_idx_list.extend([-1, -1])              
            else:
                neigh_idx_list.extend([child.name for child in node.children])
            
            edge_index.append(neigh_idx_list)                
            node_features.append(node.d)
            node_idx_list.append(node.name)
        
        branch_idx_map = torch.sort(torch.LongTensor(node_idx_list), dim=0, descending=False)[1]
        # parent_idxes = torch.LongTensor(parent_idx_list)
        edge_index = torch.LongTensor(edge_index)
        # pdb.set_trace()
        
        return torch.index_select(torch.stack(node_features), 0, branch_idx_map), edge_index[branch_idx_map]
    
    
    def mean_std(self, tree, **kwargs):
        node_features, edge_index = self.node_embedding(tree)
        node_features = self.gnn(node_features, edge_index)

        return self.mean_std_net(node_features, edge_index[:-1, 0])
            
    
    def sample_branch_base(self, n_particles):
        samp_log_branch = torch.randn(n_particles, 2*self.ntips-3)
        return samp_log_branch, torch.sum(-0.5*math.log(2*math.pi) - 0.5*samp_log_branch**2, -1)
    
    
    def forward(self, tree_list):
        mean, std = zip(*map(lambda x: self.mean_std(x), tree_list))
        mean, std = torch.stack(mean, dim=0), torch.stack(std, dim=0)
        samp_log_branch, logq_branch = self.sample_branch_base(len(tree_list))
        samp_log_branch, logq_branch = samp_log_branch * std.exp() + mean - 2.0, logq_branch - torch.sum(std, -1)
        return samp_log_branch, logq_branch 