import torch
import numpy as np
import math
from collections import defaultdict
import dgl
import utils
import pickle
import torch.nn as nn
import pdb

def gen_DGLGraph(args, ground):
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    he = []
    hv = []
    for i, edge in enumerate(ground):
        for v in edge :
            he.append(i)
            hv.append(v)
    data_dict = {
        ('node', 'in', 'edge'): (hv, he),        
        ('edge', 'con', 'node'): (he, hv)
    }
    g = dgl.heterograph(data_dict)
    return g.to(device)

def add_edge(args, batch, g):
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    edge_num = g.nodes('edge')[-1].item()
    he = []
    hv = []
    
    for i, edge in enumerate(batch):
        for v in edge :
            he.append(edge_num+i+1)
            hv.append(v.item())
    
    g.add_edges(torch.tensor(hv).to(device), torch.tensor(he).to(device), etype=('node', 'in', 'edge'))
    g.add_edges(torch.tensor(he).to(device), torch.tensor(hv).to(device), etype=('edge', 'con', 'node'))

def del_edge(args, batch, g):
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    edge_num = g.nodes('edge')[-1].item()-len(batch)
    he = []
    hv = []
    edges = []
    for i, edge in enumerate(batch):
        edges.append(edge_num+i+1)
        for v in edge :
            hv.append([v.item(), edge_num+i+1])
            he.append([edge_num+i+1, v.item()])
    #pdb.set_trace()
    g.remove_edges(g.edges(form='eid', etype='in')[-len(hv):], etype='in')
    g.remove_edges(g.edges(form='eid', etype='con')[-len(he):], etype='con')
    g.remove_nodes(torch.tensor(edges).to(device), ntype='edge')

def gen_data(args, dataset_name='cora', do_val=False):
    '''
    Retrieve and process data, can be used generically for any dataset with predefined data format, eg cora, citeseer, etc.
    flip_edge_node: whether to flip edge and node in case of relation prediction.
    '''
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    data_path = None
    if dataset_name == 'cora':
        data_path = '/workspace/jupyter/HyFER/data/cora7cls1433.pt'
    elif dataset_name == 'citeseer':
        data_path = '/workspace/jupyter/HyFER/data/citeseer6cls3703.pt'            
    elif dataset_name == 'dblp' :
        data_path = '/workspace/jupyter/HyFER/data/dblp6cls1425.pt'
    elif dataset_name == 'pubmed' :
        data_path = '/workspace/jupyter/HyFER/data/pubmed3cls500.pt'            
    else :
        raise Exception('dataset {} not supported!'.format(dataset_name))

    data_dict = torch.load(data_path)
    paper_author = torch.LongTensor(data_dict['paper_author'])
    author_paper = torch.LongTensor(data_dict['author_paper'])
    n_author = data_dict['n_author']
    n_paper = data_dict['n_paper']
    classes = data_dict['classes']
    paper_X = data_dict['paper_X']
    paperwt = data_dict['paperwt']
    authorwt = data_dict['authorwt']
    cls_l = list(set(classes))
    
    cls2int = {k:i for (i, k) in enumerate(cls_l)}
    classes = [cls2int[c] for c in classes]
    args.input_dim = paper_X.shape[-1]
    args.ne = n_author
    args.nv = n_paper
    ne = args.ne
    nv = args.nv
    args.n_cls = len(cls_l)
    # Training/validation set selection
    dataset_percent = args.label_percent if args.label_percent != -1 else utils.get_label_percent(dataset_name)
    n_labels = max(1, math.ceil(nv*dataset_percent))
    args.all_labels = torch.LongTensor(classes)
    proportional_select = True
    if proportional_select:
        n_labels = int(math.ceil(n_labels/args.n_cls)*args.n_cls)
        all_cls_idx = []
        if do_val :
            all_val_idx = []
        n_label_per_cls = n_labels//args.n_cls
        for i in range(args.n_cls):
            cur_idx = torch.LongTensor(list(range(nv)))[args.all_labels == i]
            # if do-val : select 2 times of training dataset - split them in half
            if do_val :
                rand_idx = torch.from_numpy(np.random.choice(len(cur_idx), size=(2*n_label_per_cls,), replace=False )).to(torch.int64)
                cur_idx = cur_idx[rand_idx]
                train_idx, val_idx = torch.split(cur_idx, [n_label_per_cls, n_label_per_cls])
                all_cls_idx.append(train_idx)
                all_val_idx.append(val_idx)
            else :
                rand_idx = torch.from_numpy(np.random.choice(len(cur_idx), size=(n_label_per_cls,), replace=False )).to(torch.int64)
                cur_idx = cur_idx[rand_idx]
                all_cls_idx.append*(cur_idx)
        args.label_idx = torch.cat(all_cls_idx, 0)
        if do_val :
            args.val_idx = torch.cat(all_val_idx, 0)
    else:
        args.label_idx = torch.from_numpy(np.random.choice(nv, size=(n_labels,), replace=False )).to(torch.int64)    

    args.labels = args.all_labels[args.label_idx].to(device) #torch.ones(n_labels, dtype=torch.int64)
    if do_val :
        args.valid_labels = args.all_labels[args.val_idx].to(device)
    args.all_labels = args.all_labels.to(device)
 
    if isinstance(paper_X, np.ndarray):
        args.v = torch.from_numpy(paper_X.astype(np.float32)).to(device)
    else:
        args.v = torch.from_numpy(np.array(paper_X.astype(np.float32).todense())).to(device)
        
    args.vidx = paper_author[:, 0].to(device)
    args.eidx = paper_author[:, 1].to(device)
    args.paper_author = paper_author
    
   
    args.incidence = torch.zeros(ne, nv)
    for elem in author_paper :
        e, v = elem
        args.incidence[e, v]=1
    data_dict = {
        ('node', 'in', 'edge'): (args.paper_author[:,0], args.paper_author[:,1]),        
        ('edge', 'con', 'node'): (args.paper_author[:,1], args.paper_author[:,0])
    }
    g = dgl.heterograph(data_dict)
    g.ndata['h'] = {'node' : torch.tensor(paper_X).type('torch.FloatTensor'), 'edge' : torch.ones(args.ne, args.dim_edge)}
    args.v_feat = torch.tensor(paper_X).type('torch.FloatTensor').to(device)
    args.e_feat = torch.ones(args.ne, args.dim_edge).to(device)
    
    # HGNN
    H = torch.tensor(args.incidence.T)
    W = torch.ones(H.shape[1]) # H.shape[1] = n_edge
    args.DV2 = torch.pow(torch.sum(H * W, axis=1), -0.5).to(device)
    args.invDE = torch.pow(torch.sum(H, axis=0), -1).to(device)
    
    ########### For HNHN reproducibility issue ############
    #args.alpha = 0.15 #.1 #-.1
    #
    #weights for regularization
    #'''
     
    args.v_weight = torch.Tensor([(1/w if w > 0 else 1) for w in paperwt]).unsqueeze(-1).to(device) #torch.ones((nv, 1)) / 2 #####
    args.e_weight = torch.Tensor([(1/w if w > 0 else 1) for w in authorwt]).unsqueeze(-1).to(device) # 1)) / 2 #####torch.ones(ne, 1) /
    paper2sum = defaultdict(list)
    author2sum = defaultdict(list)
    e_reg_weight = torch.zeros(args.ne) ###
    v_reg_weight = torch.zeros(args.nv) ###
    #a switch to determine whether to have wt in exponent or base
    use_exp_wt = args.use_exp_wt #True #False
    for i, (paper_idx, author_idx) in enumerate(paper_author.tolist()):
        e_wt = args.e_weight[author_idx]
        e_reg_wt = torch.exp(args.alpha_e*e_wt) if use_exp_wt else e_wt**args.alpha_e 
        e_reg_weight[author_idx] = e_reg_wt
        paper2sum[paper_idx].append(e_reg_wt) ###
        
        v_wt = args.v_weight[paper_idx]
        v_reg_wt = torch.exp(args.alpha_v*v_wt) if use_exp_wt else v_wt**args.alpha_v
        v_reg_weight[paper_idx] = v_reg_wt
        author2sum[author_idx].append(v_reg_wt) ###        
    #'''
    v_reg_sum = torch.zeros(nv) ###
    e_reg_sum = torch.zeros(ne) ###
    for paper_idx, wt_l in paper2sum.items():
        v_reg_sum[paper_idx] = sum(wt_l)
    for author_idx, wt_l in author2sum.items():
        e_reg_sum[author_idx] = sum(wt_l)

    #pdb.set_trace()
    #this is used in denominator only
    e_reg_sum[e_reg_sum==0] = 1
    v_reg_sum[v_reg_sum==0] = 1
    args.e_reg_weight = torch.Tensor(e_reg_weight).unsqueeze(-1).to(device)
    args.v_reg_sum = torch.Tensor(v_reg_sum).unsqueeze(-1).to(device)
    args.v_reg_weight = torch.Tensor(v_reg_weight).unsqueeze(-1).to(device)
    args.e_reg_sum = torch.Tensor(e_reg_sum).unsqueeze(-1).to(device)
    ########## End for HNHN Reproducibility #############
    
    return args