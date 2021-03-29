import torch
import numpy as np
import pdb
import random

class DataLoader(object):

    def __init__(self, hyperedges, labels, batch_size, device, test=False):
        self.batch_size = batch_size
        self.hyperedges = hyperedges
        self.labels = labels
        self.idx = 0
        self.device = device
        self.test = test
        self.shuffle()
    
    def eval(self):
        self.test = True

    def train(self):
        self.test = False

    def shuffle(self):
        tmp = list(zip(self.hyperedges, self.labels))
        random.shuffle(tmp)
        self.hyperedges, self.labels = zip(*tmp)
  
    def __iter__(self):
        self.idx = 0
        return self
    
    def next(self):
        return self.__next__()

    def __next__(self):
        return self.next_batch()

    def next_batch(self):
        #start = self.idx
        is_last = False
        #end = self.idx+self.batch_size if self.idx+self.batch_size < len(self.hyperedges) else len(self.hyperedges)
        end = self.idx+self.batch_size
        if end >= len(self.hyperedges):
            is_last = True
            if self.test :
                hyperedges = self.hyperedges[self.idx:]
                labels = self.labels[self.idx:]          
                self.idx = end - len(self.hyperedges)
            else :
                hyperedges = self.hyperedges[self.idx:] + self.hyperedges[:end - len(self.hyperedges)]
                labels = self.labels[self.idx:] + self.labels[:end - len(self.labels)]
                self.idx = 0
        else :
            hyperedges = self.hyperedges[self.idx:self.idx + self.batch_size]
            labels = self.labels[self.idx:self.idx + self.batch_size]
        
        hyperedges = [torch.LongTensor(edge).to(self.device) for edge in hyperedges]
        labels = torch.LongTensor(labels).to(self.device)
        self.idx = end % len(self.hyperedges)
        if is_last :
            self.shuffle()        
        return hyperedges, labels, is_last