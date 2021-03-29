###### This code originates from Author's official implementation, on https://github.com/prasanna08/Negative-Sampling-for-Hyperedge-Prediction ######

from collections import defaultdict
import math

import numpy as np
from tqdm.autonotebook import tqdm
import pdb
def sample_initial_edge(nodes_to_neighbors):
    edgeidx = np.random.choice(
        sum(len(nodes_to_neighbors[n]) for n in nodes_to_neighbors))
    carry = 0
    for n in nodes_to_neighbors:
        if edgeidx < carry + len(nodes_to_neighbors[n]):
            edge = [n, list(nodes_to_neighbors[n])[edgeidx - carry]]
            break
        carry += len(nodes_to_neighbors[n])
    return edge

def clique_negative_sampling(
        hyperedges, nodes_to_neighbors, num_negative,
        list_hyperedges, node_set):
    edgeidx = np.random.choice(len(hyperedges), size=1)[0]
    neg = list_hyperedges[edgeidx]

    while neg in hyperedges:
        edgeidx = np.random.choice(len(hyperedges), size=1)[0]
        edge = list(list_hyperedges[edgeidx])
        node_to_remove = np.random.choice(len(edge), size=1)[0]
        nodes_to_keep = edge[:node_to_remove] + edge[node_to_remove+1:]
        
        probable_neighbors = node_set
        for node in nodes_to_keep:
            probable_neighbors = probable_neighbors.intersection(
                nodes_to_neighbors[node])
        
        if len(probable_neighbors) == 0:
            continue
        probable_neighbors = list(probable_neighbors)
        neighbor_node = np.random.choice(probable_neighbors, size=1)[0]
        
        edge.append(neighbor_node)
        neg = frozenset(edge)

    edges = {
        frozenset([node1, node2])
        for node1 in neg for node2 in neg if node1 < node2
    }
    return neg, edges

def mfinder_sampling(nodes_to_neighbors, k):
    neighbor_edges = []
    induced_edges = set()
    sampled_nodes = set()
    
    while len(sampled_nodes) < k:
        while len(neighbor_edges) == 0:
            edge = sample_initial_edge(nodes_to_neighbors)
            sampled_nodes = set(edge)
            neighbor_edges = set([
                frozenset([node, nnode])
                for node in sampled_nodes for nnode in nodes_to_neighbors[node]
                if nnode not in sampled_nodes])
            neighbor_edges = list(neighbor_edges)

            induced_edges = set()
            induced_edges.add(frozenset(edge))
        
        selected_edge = neighbor_edges[np.random.choice(len(neighbor_edges))]
        induced_edges.add(selected_edge)
        new_node = [n for n in selected_edge.difference(sampled_nodes)][0]
        sampled_nodes.add(new_node)
        
        neighbor_edges = [
            edge for edge in neighbor_edges if new_node not in edge]
        
        new_edges = set()
        
        for node in nodes_to_neighbors[new_node]:
            if node not in sampled_nodes:
                new_edges.add(frozenset([new_node, node]))
            else:
                induced_edges.add(frozenset([new_node, node]))
        
        neighbor_edges.extend(list(new_edges))    
        #assert len(neighbor_edges) == len(set(neighbor_edges))

    return sampled_nodes, induced_edges

def sized_mf_sampling(size_dist, nodes, nodes_to_neighbors, hyperedges):
    vals = [v for v, p in size_dist]
    p = [p for v, p in size_dist]
    sampled_size = np.random.choice(vals, p=p)
    sampled_nodes = {'a', 'b'}
    hyperedges.add(frozenset(sampled_nodes))
    while frozenset(sampled_nodes) in hyperedges:
        sampled_nodes, sampled_edge = mfinder_sampling(
            nodes_to_neighbors, sampled_size)
    hyperedges.remove(frozenset({'a', 'b'}))
    return sampled_nodes, sampled_edge

def sized_random_sampling(size_dist, nodes, nodes_to_neighbors, hyperedges):
    vals = [v for v, p in size_dist]
    p = [p for v, p in size_dist]
    sampled_nodes = frozenset({'a', 'b'})
    hyperedges.add(sampled_nodes)
    sampled_size = np.random.choice(vals, p=p)
    while frozenset(sampled_nodes) in hyperedges:
        sampled_nodes = [
            nodes[idx] for idx in np.random.choice(
                len(nodes), size=sampled_size, replace=False)]
    edges = {
        frozenset([node, node2])
        for node in sampled_nodes for node2 in sampled_nodes
        if node2 in nodes_to_neighbors[node] and node2 < node}
    hyperedges.remove(frozenset({'a', 'b'}))
    return sampled_nodes, edges

def get_pure_sample_size_dist(num_nodes):
    N = num_nodes
    nck = 1
    size_dist = dict()
    size_dist[0] = 1
    for idx in range(1, num_nodes):
        nck *= (N - (idx - 1)) / idx
        size_dist[idx] = nck
    size_dist[N] = 1
    total = sum(v for k, v in size_dist.items())
    for i in size_dist:
        size_dist[i] = float(size_dist[i]) / total
    return size_dist

def negative_sample(
        nodes_to_neighbors, size_dist, num_negative, hyperedges, method):
    nodes = list(nodes_to_neighbors.keys())
    neg_samples = []
    
    if method == 'UNS':
        size_dist = get_pure_sample_size_dist(len(nodes))
    
    size_dist = size_dist.items()  
    if method in ['SNS', 'UNS']:
        for i in tqdm(range(num_negative), leave=False):
            sampled_edge = sized_random_sampling(
                size_dist, nodes, nodes_to_neighbors, hyperedges)
            neg_samples.append(sampled_edge)
    elif method == 'MNS':
        for i in tqdm(range(num_negative), leave=False):
            sampled_edge = sized_mf_sampling(
                size_dist, nodes, nodes_to_neighbors, hyperedges)
            neg_samples.append(sampled_edge)
    elif method == 'CNS':
        list_hyperedges = list(hyperedges)
        node_set = set(nodes_to_neighbors.keys())
        for i in tqdm(range(num_negative), leave=False):
            sampled_edge = clique_negative_sampling(
                hyperedges, nodes_to_neighbors, num_negative, list_hyperedges,
                node_set)
            neg_samples.append(sampled_edge)
    return neg_samples


def generate_hyperedge_size_dist(hyperedges):
    size_dist = defaultdict(int)
    for edge in hyperedges:
        size_dist[len(edge)] += 1
    if 1 in size_dist:
        del size_dist[1]
    if 2 in size_dist:
        del size_dist[2]
    total = sum(v for k, v in size_dist.items())
    for i in size_dist:
        size_dist[i] = float(size_dist[i]) / total
    return size_dist

def generate_negative_samples_for_hyperedges(
        hyperedges, method, neg_samples_size):
    #print(hyperedges)
    edges = {
        frozenset({u, v}) for hedge in hyperedges
        for u in hedge for v in hedge if u > v}
    nodes_to_neighbors = defaultdict(set)
    for edge in edges:
        u, v = edge
        nodes_to_neighbors[u].add(v)
        nodes_to_neighbors[v].add(u)

    size_dist = generate_hyperedge_size_dist(hyperedges)
    
    #print('Generating Negative Samples')
    total = math.ceil(neg_samples_size)
    neg_samples = negative_sample(
        nodes_to_neighbors, size_dist, total, hyperedges, method)
    negative_hyperedges = [frozenset(x) for x, y in neg_samples]
    
    return negative_hyperedges

class UNSSampler(object):
    def __init__(self, ratio):
        self.ratio = ratio
    def __call__(self, hedges):
        neg_samples_size = int(self.ratio*len(hedges))
        neg_samples = generate_negative_samples_for_hyperedges(hedges, 'UNS', neg_samples_size)
        return neg_samples

class SNSSampler(object):
    def __init__(self, ratio):
        self.ratio = ratio
    def __call__(self, hedges):
        neg_samples_size = int(self.ratio*len(hedges))
        neg_samples = generate_negative_samples_for_hyperedges(hedges, 'SNS', neg_samples_size)
        return neg_samples

class MNSSampler(object):
    def __init__(self, ratio):
        self.ratio = ratio
    def __call__(self, hedges):
        neg_samples_size = int(self.ratio*len(hedges))
        #pdb.set_trace()
        neg_samples = generate_negative_samples_for_hyperedges(hedges, 'MNS', neg_samples_size)
        return neg_samples

class CNSSampler(object):
    def __init__(self, ratio):
        self.ratio = ratio
    def __call__(self, hedges):
        neg_samples_size = int(self.ratio*len(hedges))
        neg_samples = generate_negative_samples_for_hyperedges(hedges, 'CNS', neg_samples_size)
        return neg_samples

class MixedSampler(object):
    def __init__(self, ratio):
        self.ratio = ratio
    def __call__(self, hedges):
        neg_samples_size = int(self.ratio*len(hedges)/3)
        neg_samples2 = generate_negative_samples_for_hyperedges(hedges, 'SNS', neg_samples_size)
        neg_samples3 = generate_negative_samples_for_hyperedges(hedges, 'MNS', neg_samples_size)
        neg_samples4 = generate_negative_samples_for_hyperedges(hedges, 'CNS', neg_samples_size)
        neg_samples = neg_samples2 + neg_samples3 + neg_samples4
        return neg_samples
