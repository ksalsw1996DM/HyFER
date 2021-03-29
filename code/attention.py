import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

class Multihead(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads):
        super(Multihead, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.q_lin = nn.Linear(dim_Q, dim_V)
        self.k_lin = nn.Linear(dim_K, dim_V)
        self.v_lin = nn.Linear(dim_K, dim_V)
        self.m_lin = nn.Linear(dim_V, dim_V)

    def forward(self, tgt, src):
        Q = self.q_lin(tgt)
        K, V = self.k_lin(src), self.v_lin(src)

        dim_split = self.dim_V // self.num_heads
        Q = torch.cat(Q.split(dim_split, 1), 0)
        K = torch.cat(K.split(dim_split, 1), 0)
        V = torch.cat(V.split(dim_split, 1), 0)

        Attention = torch.softmax(torch.matmul(Q, K.transpose(0,1))/math.sqrt(self.dim_V), 1)
        Multi = torch.cat((Q + torch.matmul(Attention, V)).split(Q.size(0), 0), 1)
        Multi = Multi + F.relu(self.m_lin(Multi))
        return Multi



