import numpy as np
import copy
from Bio import Phylo
from io import StringIO
# from cStringIO import StringIO
from ete3 import Tree
from bitarray import bitarray
from treeManipulation import init, namenum, nametaxon
from collections import defaultdict, OrderedDict
import pdb


class BitArray(object):
    def __init__(self, taxa):
        self.taxa = taxa
        self.ntaxa = len(taxa)
        self.map = {taxon: i for i, taxon in enumerate(taxa)}
        
    def combine(self, arrA, arrB):
        if arrA < arrB:
            return arrA + arrB
        else:
            return arrB + arrA 
        
    def merge(self, key):
        return bitarray(key[:self.ntaxa]) | bitarray(key[self.ntaxa:])
        
    def decomp_minor(self, key):
        return min(bitarray(key[:self.ntaxa]), bitarray(key[self.ntaxa:]))
        
    def minor(self, arrA):
        return min(arrA, ~arrA)
        
    def from_clade(self, clade):
        bit_list = ['0'] * self.ntaxa
        for taxon in clade:
            bit_list[self.map[taxon]] = '1'
        return bitarray(''.join(bit_list))

    
def taxa2num(taxa):
    taxa2numMap = {}
    for i, taxon in enumerate(taxa):
        taxa2numMap[taxon] = i    
    return taxa2numMap
    
def logsumexp(x):
    max_x = np.max(x)
    return np.log(np.sum(np.exp(x - max_x))) + max_x    
    

def mcmc_treeprob(filename, data_type, truncate=None, taxon=None):
    mcmc_samp_tree_stats = Phylo.parse(filename, data_type)
    mcmc_samp_tree_dict = OrderedDict()
    mcmc_samp_tree_name = []
    mcmc_samp_tree_wts = []
    num_hp_tree = 0
    if taxon:
        taxon2idx = {taxon: i for i, taxon in enumerate(taxon)}
        
    for tree in mcmc_samp_tree_stats:
        handle = StringIO()
        Phylo.write(tree, handle,'newick')
        mcmc_samp_tree_dict[tree.name] = Tree(handle.getvalue().strip())
        if taxon:
            if taxon != 'keep':
                namenum(mcmc_samp_tree_dict[tree.name],taxon)
        else:
            init(mcmc_samp_tree_dict[tree.name],name='interior')
            
        handle.close()
        mcmc_samp_tree_name.append(tree.name)
        mcmc_samp_tree_wts.append(tree.weight)
        num_hp_tree += 1
        
        if truncate and num_hp_tree >= truncate:
            break
    
    return mcmc_samp_tree_dict, mcmc_samp_tree_name, mcmc_samp_tree_wts
    
    
def summary(dataset, file_path, samp_size=750001):
    tree_dict_total = OrderedDict()
    tree_dict_map_total = defaultdict(float)
    tree_names_total = []
    tree_wts_total = []
    n_samp_tree = 0
    for i in range(1,11):
        tree_dict_rep, tree_name_rep, tree_wts_rep = mcmc_treeprob(file_path + dataset + '/rep_{}/'.format(i) + dataset + '.trprobs', 'nexus', taxon='keep')
        tree_wts_rep = np.round(np.array(tree_wts_rep)*samp_size)
 
        for i, name in enumerate(tree_name_rep):
            tree_id = tree_dict_rep[name].get_topology_id()
            if tree_id not in tree_dict_map_total:
                n_samp_tree += 1
                tree_names_total.append('tree_{}'.format(n_samp_tree))
                tree_dict_total[tree_names_total[-1]] = tree_dict_rep[name]

            tree_dict_map_total[tree_id] += tree_wts_rep[i]
    
    for key in tree_dict_map_total:
        tree_dict_map_total[key] /= 10*samp_size

    for name in tree_names_total:
        tree_wts_total.append(tree_dict_map_total[tree_dict_total[name].get_topology_id()])  
        
    return tree_dict_total, tree_names_total, tree_wts_total


def get_tree_list_raw(filename, burnin=0, truncate=None, hpd=0.95):
    tree_dict = {}
    tree_wts_dict = defaultdict(float)
    tree_names = []
    i, num_trees = 0, 0
    with open(filename, 'r') as input_file:
        while True:
            line = input_file.readline()
            if line == "":
                break
            num_trees += 1
            if num_trees < burnin:
                continue
            tree = Tree(line.strip())
            tree_id = tree.get_topology_id()
            if tree_id not in tree_wts_dict:
                tree_name = 'tree_{}'.format(i)
                tree_dict[tree_name] = tree
                tree_names.append(tree_name)
                i += 1            
            tree_wts_dict[tree_id] += 1.0
            
            if truncate and num_trees == truncate + burnin:
                break
    tree_wts = [tree_wts_dict[tree_dict[tree_name].get_topology_id()]/(num_trees-burnin) for tree_name in tree_names]
    if hpd < 1.0:
        ordered_wts_idx = np.argsort(tree_wts)[::-1]
        cum_wts_arr = np.cumsum([tree_wts[k] for k in ordered_wts_idx])
        cut_at = next(x[0] for x in enumerate(cum_wts_arr) if x[1] > hpd)
        tree_wts = [tree_wts[k] for k in ordered_wts_idx[:cut_at]]
        tree_names = [tree_names[k] for k in ordered_wts_idx[:cut_at]]
        
    return tree_dict, tree_names, tree_wts
    

def summary_raw(dataset, file_path, truncate=None, hpd=0.95, n_rep=10):
    tree_dict_total = {}
    tree_id_set_total = set()
    tree_names_total = []
    n_samp_tree = 0
    
    for i in range(1, n_rep+1):
        tree_dict_rep, tree_names_rep, tree_wts_rep = get_tree_list_raw(file_path + dataset + '/' + dataset + '_ufboot_rep_{}'.format(i), truncate=truncate, hpd=hpd)
        for j, name in enumerate(tree_names_rep):
            tree_id = tree_dict_rep[name].get_topology_id()
            if tree_id not in tree_id_set_total:
                n_samp_tree += 1
                tree_names_total.append('tree_{}'.format(n_samp_tree))
                tree_dict_total[tree_names_total[-1]] = tree_dict_rep[name]
                tree_id_set_total.add(tree_id)
    
    return tree_dict_total, tree_names_total
    

def get_support_from_mcmc(taxa, tree_dict_total, tree_names_total, tree_wts_total=None):
    rootsplit_supp_dict = OrderedDict()
    subsplit_supp_dict = OrderedDict()
    toBitArr = BitArray(taxa)
    for i, tree_name in enumerate(tree_names_total):
        tree = tree_dict_total[tree_name]
        wts = tree_wts_total[i] if tree_wts_total else 1.0
        nodetobitMap = {node:toBitArr.from_clade(node.get_leaf_names()) for node in tree.traverse('postorder') if not node.is_root()}
        for node in tree.traverse('levelorder'):
            if not node.is_root():
                rootsplit = toBitArr.minor(nodetobitMap[node]).to01()
                # rootsplit_supp_dict[rootsplit] += wts
                if rootsplit not in rootsplit_supp_dict:
                    rootsplit_supp_dict[rootsplit] = 0.0
                rootsplit_supp_dict[rootsplit] += wts
                if not node.is_leaf():
                    child_subsplit = min([nodetobitMap[child] for child in node.children]).to01()
                    for sister in node.get_sisters():
                        parent_subsplit = (nodetobitMap[sister] + nodetobitMap[node]).to01()
                        if parent_subsplit not in subsplit_supp_dict:
                            subsplit_supp_dict[parent_subsplit] = OrderedDict()
                        if child_subsplit not in subsplit_supp_dict[parent_subsplit]:
                            subsplit_supp_dict[parent_subsplit][child_subsplit] = 0.0
                        subsplit_supp_dict[parent_subsplit][child_subsplit] += wts
                    if not node.up.is_root():
                        parent_subsplit = (~nodetobitMap[node.up] + nodetobitMap[node]).to01()
                        if parent_subsplit not in subsplit_supp_dict:
                            subsplit_supp_dict[parent_subsplit] = OrderedDict()
                        if child_subsplit not in subsplit_supp_dict[parent_subsplit]:
                            subsplit_supp_dict[parent_subsplit][child_subsplit] = 0.0                        
                        subsplit_supp_dict[parent_subsplit][child_subsplit] += wts
                        
                    parent_subsplit = (~nodetobitMap[node] + nodetobitMap[node]).to01()
                    if parent_subsplit not in subsplit_supp_dict:
                        subsplit_supp_dict[parent_subsplit] = OrderedDict()
                    if child_subsplit not in subsplit_supp_dict[parent_subsplit]:
                        subsplit_supp_dict[parent_subsplit][child_subsplit] = 0.0
                    subsplit_supp_dict[parent_subsplit][child_subsplit] += wts
                
                if not node.up.is_root():
                    bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()] + [~nodetobitMap[node.up]])
                else:
                    bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()])
                child_subsplit = bipart_bitarr.to01()
                if not node.is_leaf():
                    for child in node.children:
                        parent_subsplit = (nodetobitMap[child] + ~nodetobitMap[node]).to01()
                        if parent_subsplit not in subsplit_supp_dict:
                            subsplit_supp_dict[parent_subsplit] = OrderedDict()
                        if child_subsplit not in subsplit_supp_dict[parent_subsplit]:
                            subsplit_supp_dict[parent_subsplit][child_subsplit] = 0.0
                        subsplit_supp_dict[parent_subsplit][child_subsplit] += wts
                
                parent_subsplit = (nodetobitMap[node] + ~nodetobitMap[node]).to01()
                if parent_subsplit not in subsplit_supp_dict:
                    subsplit_supp_dict[parent_subsplit] = OrderedDict()
                if child_subsplit not in subsplit_supp_dict[parent_subsplit]:
                    subsplit_supp_dict[parent_subsplit][child_subsplit] = 0.0
                subsplit_supp_dict[parent_subsplit][child_subsplit] += wts

    return rootsplit_supp_dict, subsplit_supp_dict 


def tree_process(tree, toBitArr):
    for node in tree.traverse("postorder"):
        if not node.is_root():
            node.clade_bitarr = toBitArr.from_clade(node.get_leaf_names())
            node.split_bitarr = min([node.clade_bitarr, ~node.clade_bitarr]).to01()
            
            
def generate(taxa):
    if len(taxa)==3:
        return [Tree('('+','.join(taxa)+');')]
    else:
        res = []
        sister = Tree('('+taxa[-1]+');')
        for tree in generate(taxa[:-1]):
            for node in tree.traverse('preorder'):
                if not node.is_root():
                    node.up.add_child(sister)
                    node.detach()
                    sister.add_child(node)
                    res.append(copy.deepcopy(tree))
                    node.detach()
                    sister.up.add_child(node)
                    sister.detach()
        
        return res