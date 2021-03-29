import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", dest='fix_seed', action='store_const', default=False, const=True, help='Fix seed for reproducibility and fair comparison.')
    parser.add_argument("--do_svd", action='store_const', default=False, const=True, help='use svd')
    parser.add_argument("--kfold", default=-1, type=int, help='for k-fold cross validation')
    parser.add_argument("--num_heads", default=1, type=int, help='set transformer multi head attention')
    parser.add_argument("--dropout", default=0.3, type=float, help='dropout')
    parser.add_argument("--lr", default=0.004, type=float, help='learning rate')
    parser.add_argument("--bs", default=64, type=int, help='batch size')
    parser.add_argument("--n_layers", default=1, type=int, help='number of layers')
    parser.add_argument("--exp_num", default=10, type=int, help='number of experiments')
    parser.add_argument("--exp_wt", dest='use_exp_wt', action='store_const', default=False, const=True, help='Fix seed for reproducibility and fair comparison.')
    parser.add_argument("--alpha_e", default=0, type=float, help='alpha')
    parser.add_argument("--alpha_v", default=0, type=float, help='alpha')
    parser.add_argument("--dataset_name", type=str, default='cora', help='dataset name')
    parser.add_argument("--n_hidden", default=400, type=int, help='Dimension of hidden vector')
    parser.add_argument("--dim_query", default=64, type=int, help='Dimension of vertex hidden vector')
    parser.add_argument("--dim_vertex", default=400, type=int, help='Dimension of vertex hidden vector')
    parser.add_argument("--dim_edge", default=400, type=int, help='Dimension of edge hidden vector')
    parser.add_argument("--epochs", default=200, type=int, help='number of epochs')
    parser.add_argument("--model", default='hnhn', type=str, help='Use which model')
    parser.add_argument("--label_percent", default=-1, type=float, help='control label ratio')
    parser.add_argument("--alpha_exp", dest='alpha_exp', action='store_const',default=False, const=True, help='find hyperparameter alpha_v, alpha_e')
    parser.add_argument("--scheduler", type=str, default='multi', help='schedular: multi or plateau')
    parser.add_argument("--test_ratio", type=float, default=0.2, help='test set ratio')
    parser.add_argument("--nsampler", type=str, default='MNS', help='negative sampler')
    parser.add_argument("--neg_ratio", type=int, default=5, help='negative sample ratio')
    parser.add_argument("--gpu", type=int, default=-1, help='gpu number. -1 if cpu else gpu number')
    parser.add_argument("--splits", type=bool, default=False, help='use previously splitted/negative sampled dataset')
    parser.add_argument("--complex", type=bool, default=False, help='use complex classifier or not')
    parser.add_argument("--aggr", type=str, default='mean', help='aggregation method: mean, set, sagnn')
    parser.add_argument("--init_feat", type=str, default='default', help='use of initial feature')
    opt = parser.parse_args()
    return opt
 
def readlines(path):
    with open(path, 'r') as f:
        return f.readlines()
    

def get_label_percent(dataset_name):
    if dataset_name == 'cora':
        return .052
    elif dataset_name == 'citeseer':
        return .15 
    elif dataset_name == 'dblp':
        return .04
    elif dataset_name == "pubmed":
        return .02
    else:
        raise Exception('dataset not supported')
    
def gen_HE_data(args):
    incidence = {}
    paper_author = args.paper_author.numpy()
    for tup in paper_author:
        v, e = tup
        if e not in incidence :
            incidence[e]=[]
        incidence[e].append(v)
    HE = []
    for e in incidence.keys():
        HE.append(frozenset(incidence[e]))
    test_num = int(args.test_ratio*len(HE))
    total_idx = list(range(len(HE)))
    test_idx = random.sample(total_idx, test_num)
    train_data = []
    test_data = []
    for idx in total_idx :
        if idx in test_idx :
            test_data.append(HE[idx])
        else :
            train_data.append(HE[idx])
    return train_data, test_data