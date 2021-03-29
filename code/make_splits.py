
'''
Processing data
'''
import torch
import numpy as np
import os
from collections import defaultdict
import re
import sklearn
import sklearn.feature_extraction as feat_extract
from sklearn.decomposition import TruncatedSVD
import utils
import pickle
from sampler import *
import random

def split_coauthor(args, dataset, remove_isolated=True):
    '''
    In co-authorship dataset, 
    hypergraph.pickle of form dictionary of "author" : [list of paper] format
    features.pickle of form list of paper features format
    labels.pickle of the form list of paper label format
    author's ids are hyperedges, papers' ids are hypernodes    
    '''
    feature_path = 'data/coauthorship/{}/features.pickle'.format(dataset)
    hypergraph_path = 'data/coauthorship/{}/hypergraph.pickle'.format(dataset)
    label_path = 'data/coauthorship/{}/labels.pickle'.format(dataset)
    paper2citing = defaultdict(set)
    citing2paper = defaultdict(set)
    f_feat = open(feature_path, 'rb')
    f_hgraph = open(hypergraph_path, 'rb')
    f_label = open(label_path, 'rb')
    feat_val = pickle.load(f_feat).todense()
    hgraph_val = pickle.load(f_hgraph)
    label_val = pickle.load(f_label)
    author_list = [] # list of author - translates author name into index
    paper_list = [] # list of paper - translates paper number into index
    author_paper = {} # dict of int:list - author idx i wrote papers with idx [j, k, l, .. ]
    paper_author = {} # dict of int:list - paper idx i were written by authors with idx [j, k, l, ..]
    idx2paper = {} # dict of int:int - translate paper idx i to paper number
    num_papers = feat_val.shape[0]
    for author in hgraph_val.keys() :
        papers = hgraph_val[author]
        if author not in author_list :
            author_list.append(author)
        idx = author_list.index(author)
        if idx not in author_paper :
            author_paper[idx]=[]
        translated_p = []
        for paper in papers :
            if paper not in paper_list :
                paper_list.append(paper)
            p_idx = paper_list.index(paper)
            idx2paper[p_idx]=paper
            translated_p.append(p_idx) # translate paper number into index
            if p_idx not in paper_author :
                paper_author[p_idx]=[]
            paper_author[p_idx].append(idx)
        author_paper[idx]=translated_p

    paper_citing = [] # list of [paper author]
    citing_paper = [] # list of [author paper]
    for author in author_paper.keys():
        papers = author_paper[author]
        for paper in papers :
            paper_citing.append([paper, author])
            citing_paper.append([author, paper])

    if args.nsampler == 'UNS':
        train_sampler = UNSSampler(args.neg_ratio)
        test_sampler = UNSSampler(1)
    elif args.nsampler == 'MNS':
        train_sampler = MNSSampler(args.neg_ratio)
        test_sampler = MNSSampler(1)
    elif args.nsampler == 'SNS':
        train_sampler = SNSSampler(args.neg_ratio)
        test_sampler = SNSSampler(1)
    elif args.nsampler == 'CNS':
        train_sampler = CNSSampler(args.neg_ratio)
        test_sampler = CNSSampler(1)
    elif args.nsampler == 'Mixed':
        train_sampler = MixedSampler(args.neg_ratio)
        test_sampler = MixedSampler(1)
    for split in range(10):
        incidence={}
        for tup in paper_citing:
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
        
        train_label = [1]*len(train_data)+[0]*(args.neg_ratio*len(train_data))
        test_label = [1]*len(test_data)+[0]*(args.neg_ratio*len(test_data))
        train_neg = train_sampler(set(train_data))
        test_neg = test_sampler(set(test_data))
        
        train_data += list(train_neg)
        test_data += list(test_neg)
        
        train_data = [list(edge) for edge in train_data]
        test_data = [list(edge) for edge in test_data]
        
        torch.save({'train_data': train_data, 'train_label': train_label, 'test_data': test_data, 'test_label': test_label,}, 'data/{}{}{}split{}.pt'.format(dataset, args.nsampler, args.neg_ratio, split))

def split_citation(args, dataset, remove_isolated=True):
    '''
    In co-authorship dataset, 
    hypergraph.pickle of form dictionary of "author" : [list of paper] format
    features.pickle of form list of paper features format
    labels.pickle of the form list of paper label format
    author's ids are hyperedges, papers' ids are hypernodes    
    '''
    feature_path = 'data/cocitation/{}/features.pickle'.format(dataset)
    hypergraph_path = 'data/cocitation/{}/hypergraph.pickle'.format(dataset)
    label_path = 'data/cocitation/{}/labels.pickle'.format(dataset)
    paper2citing = defaultdict(set)
    citing2paper = defaultdict(set)
    f_feat = open(feature_path, 'rb')
    f_hgraph = open(hypergraph_path, 'rb')
    f_label = open(label_path, 'rb')
    feat_val = pickle.load(f_feat).todense()
    hgraph_val = pickle.load(f_hgraph)
    label_val = pickle.load(f_label)
    author_list = [] # list of author - translates author name into index
    paper_list = [] # list of paper - translates paper number into index
    author_paper = {} # dict of int:list - author idx i wrote papers with idx [j, k, l, .. ]
    paper_author = {} # dict of int:list - paper idx i were written by authors with idx [j, k, l, ..]
    idx2paper = {} # dict of int : int - translate paper index i into paper number
    idx2citation = {} # dict of int : int - translate citing paper index i into paper number
    num_papers = feat_val.shape[0]
    for author in hgraph_val.keys() :
        papers = hgraph_val[author]
        if author not in author_list :
            author_list.append(author)
        idx = author_list.index(author)
        idx2citation[idx] = author
        if idx not in author_paper :
            author_paper[idx]=[]
        translated_p = []
        for paper in papers :
            if paper not in paper_list :
                paper_list.append(paper)
            p_idx = paper_list.index(paper)
            idx2paper[p_idx]=paper
            translated_p.append(p_idx) # translate paper number into index
            if p_idx not in paper_author :
                paper_author[p_idx]=[]
            paper_author[p_idx].append(idx)
        author_paper[idx]=translated_p
    # paper : corresponds to hypernode
    # citing : corresponds to hyperedge(author)
    print("number of vertex : {} / number of hyperedge : {}".format(len(paper_list), len(author_list)))
    paper_citing = [] # list of [paper author]
    citing_paper = [] # list of [author paper]
    paperwt = torch.zeros(len(paper_list)) # corresponds to hypernode
    citingwt = torch.zeros(len(author_list)) # corresponds to hyperedge
    n_paper = len(paper_list)
    n_hedge = len(author_list)
    feat_dim = feat_val.shape[1]
    for author in author_paper.keys():
        papers = author_paper[author]
        if len(papers) == 1 :
            continue # remove one-cited papers
        for paper in papers :
            paper_citing.append([paper, author])
            citing_paper.append([author, paper])
            paperwt[paper]+=1
            citingwt[author]+=1
    if args.nsampler == 'UNS':
        train_sampler = UNSSampler(args.neg_ratio)
        test_sampler = UNSSampler(1)
    elif args.nsampler == 'MNS':
        train_sampler = MNSSampler(args.neg_ratio)
        test_sampler = MNSSampler(1)
    elif args.nsampler == 'SNS':
        train_sampler = SNSSampler(args.neg_ratio)
        test_sampler = SNSSampler(1)
    elif args.nsampler == 'CNS':
        train_sampler = CNSSampler(args.neg_ratio)
        test_sampler = CNSSampler(1)
    elif args.nsampler == 'Mixed':
        train_sampler = MixedSampler(args.neg_ratio)
        test_sampler = MixedSampler(1)
    for split in range(10):
        incidence={}
        for tup in paper_citing:
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
        
        train_label = [1]*len(train_data)+[0]*(args.neg_ratio*len(train_data))
        test_label = [1]*len(test_data)+[0]*(args.neg_ratio*len(test_data))
        train_neg = train_sampler(set(train_data))
        test_neg = test_sampler(set(test_data))
        
        train_data += list(train_neg)
        test_data += list(test_neg)
        
        train_data = [list(edge) for edge in train_data]
        test_data = [list(edge) for edge in test_data]
        
        torch.save({'train_data': train_data, 'train_label': train_label, 'test_data': test_data, 'test_label': test_label,}, 'data/splits/{}{}{}split{}.pt'.format(dataset, args.nsampler, args.neg_ratio, split))

if __name__ == '__main__':
    args = utils.parse_args()
    #dataset_name = 'citeseer' #'cora'
    #dataset_name = 'cora'
    """if args.dataset_name == 'cora':
        #process_cora_cls('data/cora/classifications')
        process_meta_files(args)
    elif args.dataset_name == 'citeseer':
        process_citeseer(args)"""
    if args.dataset_name == 'cora' or args.dataset_name == 'citeseer' or args.dataset_name == 'pubmed' :
        split_citation(args, args.dataset_name)
    else :
        split_coauthor(args, args.dataset_name)