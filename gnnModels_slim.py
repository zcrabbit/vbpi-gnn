import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pdb


from torch import Tensor

def uniform(size, value):
    if isinstance(value, Tensor):
        bound = 1. / math.sqrt(size)
        value.data.uniform_(-bound, bound)


        
class IDConv(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, *args):
        return x
        

class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.transform = nn.Linear(self.in_channels, self.out_channels, bias=False)
        
    def forward(self, x, edge_index, *args):
        node_degree = torch.sum(edge_index >= 0, dim=-1, keepdim=True, dtype=torch.float) + 1.0
        x = self.transform(x) / node_degree**0.5
        
        node_feature_padded = torch.cat((x, torch.zeros(1, self.out_channels)), dim=0)
        neigh_feature = node_feature_padded[edge_index]
        node_and_neigh_feature = torch.cat((neigh_feature, x.view(-1, 1, self.out_channels)), dim=1)
        
        return torch.sum(node_and_neigh_feature, dim=1) / node_degree**0.5
        

class SAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, aggr='mean', project=False, **kwargs):
        super().__init__()
        self.aggr = aggr
        self.in_channels, self.out_channels = in_channels, out_channels        
        self.proj = project
        if project:
            self.mlp = nn.Sequential(nn.Linear(self.in_channels, self.out_channels, bias=bias),
                                     nn.ELU(),)
            self.transform = nn.Linear(self.in_channels + self.out_channels, self.out_channels, bias=False)
        else:
            self.transform = nn.Linear(2*self.in_channels, self.out_channels, bias=bias)
            
        
    def forward(self, x, edge_index, *args):
        if self.proj:
            node_feature_padded = torch.cat((self.mlp(x), torch.zeros(1, self.out_channels)), dim=0)
        else:
            node_feature_padded = torch.cat((x, torch.zeros(1, self.in_channels)), dim=0)
        neigh_feature = node_feature_padded[edge_index]
        
        if self.aggr == 'mean':
            node_degree = torch.sum(edge_index >= 0, dim=-1, keepdim=True)
            neigh_feature_agg = torch.sum(neigh_feature, dim=1) / node_degree
        elif self.aggr == 'sum':
            neigh_feature_agg = torch.sum(neigh_feature, dim=1)
        elif self.aggr == 'max':
            neigh_feature_agg = torch.max(neigh_feature, dim=1)[0]
        else:
            raise NotImplementedError
        
        node_and_neigh_feature = torch.cat((x, neigh_feature_agg), dim=1)
        return self.transform(node_and_neigh_feature)
        
        
class GINConv(nn.Module):
    def __init__(self, in_channels, out_channels, eps=0., train_eps=True, bias=True, **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps]))
        else:
            self.register_buffer('eps', torch.tensor([eps]))
        
        self.mlp = nn.Sequential(nn.Linear(self.in_channels, self.out_channels, bias=bias),
                                 # nn.ELU(),
                                 # nn.Linear(self.out_channels, self.out_channels, bias=bias),
                                 )
    
    def forward(self, x, edge_index, *args):
        node_feature_padded = torch.cat((x, torch.zeros(1, self.in_channels)), dim=0)
        neigh_feature = node_feature_padded[edge_index]
        
        return self.mlp((1+self.eps)*x + neigh_feature.sum(1))


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.mlp = nn.Sequential(nn.Linear(2*self.in_channels, self.out_channels, bias=bias),
                                 nn.ELU(),)
        
    def forward(self, x, edge_index, *args):
        node_feature_padded = torch.cat((x, torch.zeros(1, self.in_channels)), dim=0)
        neigh_feature = node_feature_padded[edge_index]
        
        x_ = x.repeat(1, edge_index.shape[-1]).view(-1, self.in_channels)
        node_and_neigh_feature = torch.cat((x_, neigh_feature.view(-1, self.in_channels)-x_), dim=-1)
        output = self.mlp(node_and_neigh_feature)
        # output[edge_index.flatten()==-1] = -math.inf
        output = torch.where(edge_index.flatten().view(-1, 1)!=-1, output, torch.tensor(-math.inf))
        
        return torch.max(output.view(-1, edge_index.shape[-1], self.out_channels), dim=1)[0]
    
        

class GatedGraphConv(nn.Module):
    def __init__(self, out_channels, num_layers=1, bias=True, **kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.weight = nn.Parameter(torch.randn(num_layers, self.out_channels, self.out_channels))
        
        self.rnn = nn.GRUCell(out_channels, out_channels, bias=bias)
        
    def reset_parameters(self):
        uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()
        
    def forward(self, x, edge_index, *args):
        if x.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')
        
        if x.size(-1) < self.out_channels:
            x = torch.cat((x, x.new_zeros(x.size(0), self.out_channels - x.size(-1))), dim=1)
        
        for i in range(self.num_layers):
            m = torch.matmul(x, self.weight[i])
            
            node_feature_padded = torch.cat((m, torch.zeros(1, self.out_channels)))
            neigh_feature = node_feature_padded[edge_index]
            m = neigh_feature.sum(1)
            
            x = self.rnn(m, x)
        
        return x
        
        
        
class GNNStack(nn.Module):
    gnnModels = {'gcn': GCNConv,
                 'sage': SAGEConv,
                 'gin': GINConv,
                 'edge': EdgeConv,
             }
    def __init__(self, in_channels, out_channels, num_layers=1, bias=True, aggr='sum', gnn_type='gcn', project=False, **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.num_layers = num_layers
        
        self.gconvs = nn.ModuleList()
        self.gconvs.append(self.gnnModels[gnn_type](self.in_channels, self.out_channels, bias=bias, aggr=aggr, project=project))
        for i in range(self.num_layers-1):
            self.gconvs.append(self.gnnModels[gnn_type](self.out_channels, self.out_channels, bias=bias, aggr=aggr, project=project))
        
    
    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.gconvs[i](x, edge_index)
            x = F.elu(x)
        
        return x
        
        

class MeanStdPooling(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        
        self.net = nn.Sequential(nn.Linear(self.in_channels, self.out_channels, bias=bias),
                                 nn.ELU(),
                                 nn.Linear(self.out_channels, self.out_channels, bias=bias),
                                 nn.ELU(),)
        
        self.readout = nn.Sequential(nn.Linear(self.out_channels, self.out_channels, bias=bias),
                                     nn.ELU(),
                                     nn.Linear(self.out_channels, 2, bias=bias),)
                                     
    
    def forward(self, x, parent_index):
        mean_std = self.net(x)           
        mean_std = torch.max(mean_std[:-1], mean_std[parent_index])                
        mean_std = self.readout(mean_std)
        
        return mean_std[:, 0], mean_std[:, 1]
        

class GraphPooling(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, aggr='mean', **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.aggr = aggr
        
        self.net = nn.Sequential(nn.Linear(self.in_channels, self.out_channels, bias=bias),
                                 nn.ELU(),
                                 nn.Linear(self.out_channels, self.out_channels, bias=bias),
                                 nn.ELU(),)
        
        self.readout = nn.Sequential(nn.Linear(self.out_channels, self.out_channels, bias=bias),
                                     nn.ELU(),
                                     nn.Linear(self.out_channels, 1, bias=bias),)
    
    def forward(self, x):
        output = self.net(x)
        if self.aggr == 'mean':
            output = torch.mean(output, dim=0, keepdim=True)
        elif self.aggr == 'sum':
            output = torch.sum(output, dim=0, keepdim=True)
        elif self.aggr == 'max':
            output = torch.max(output, dim=0, keepdim=True)
        else:
            raise NotImplementedError
        
        return self.readout(output)            
        
