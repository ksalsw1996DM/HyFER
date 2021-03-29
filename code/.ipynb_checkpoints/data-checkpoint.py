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

def process_coauthor(args, dataset, remove_isolated=True):
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
        for paper in papers :
            paper_citing.append([paper, author])
            citing_paper.append([author, paper])
            paperwt[paper]+=1
            citingwt[author]+=1
    
    X = np.zeros((n_paper, feat_dim))
    classes = []
    for i in range(n_paper):
        paper_num = idx2paper[i]
        X[i]=feat_val[paper_num][0]
        classes.append(label_val[paper_num])
    if args.do_svd:
        feat_dim = 300
        svd = TruncatedSVD(n_components=feat_dim, n_iter=12)
        X = svd.fit_transform(X)
    print("Input feature has shape {}".format(X.shape))
    cls2idx = {}
    torch.save({'n_author': n_hedge, 'n_paper': n_paper, 'classes':classes, 'paper_author': paper_citing, 'author_paper': citing_paper, 'paperwt': paperwt, 'authorwt': citingwt, 'paper_X': X, }, 'data/{}{}cls{}.pt'.format(dataset, len(set(classes)), feat_dim) ) #_{}cls.pt'.format(len(set(classes))))

def process_citation(args, dataset, remove_isolated=True):
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
    X = np.zeros((n_paper, feat_dim))
    citing_X = np.zeros((n_hedge, feat_dim))
    classes = []
    citing_classes = []
    for i in range(n_paper):
        paper_num = idx2paper[i]
        X[i]=feat_val[paper_num][0]
        classes.append(label_val[paper_num])
    for i in range(n_hedge):
        paper_num = idx2citation[i]
        citing_X[i]=feat_val[paper_num][0]
        citing_classes.append(label_val[paper_num])
    torch.save({'n_author': n_hedge, 'n_paper': n_paper, 'classes':classes, 'author_classes': citing_classes, 'paper_author': paper_citing, 'author_paper': citing_paper, 'paperwt': paperwt, 'authorwt': citingwt, 'paper_X': X, 'author_X': citing_X}, 'data/{}{}cls{}.pt'.format(dataset, len(set(classes)), feat_dim) ) #_{}cls.pt'.format(len(set(classes))))

if __name__ == '__main__':
    args = utils.parse_args()
    if args.dataset_name == 'cora' or args.dataset_name == 'citeseer' or args.dataset_name == 'pubmed' :
        process_citation(args, args.dataset_name)
    else :
        process_coauthor(args, args.dataset_name)