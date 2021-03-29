import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pdb

class multilayers(nn.Module):
    def __init__(self, model, inputs, n_layers):
        super(multilayers, self).__init__()
        self.layers = []
        self.model = model
        for i in range(n_layers):
            self.layers.append(self.model(*inputs))
    
    def to(self, device):
        for layer in self.layers:
            layer.to(device)
        return self
        
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
            
    def forward(self, inputs, n_layers):
        #pdb.set_trace()
        first_layer, last_layer = True, False
        for i, layer in enumerate(self.layers):
            if i == n_layers-1:
                last_layer = True
                return layer(*inputs, first_layer, last_layer)
            else:
                inputs = layer(*inputs, first_layer, last_layer)
            first_layer = False
        print('error')
            
class HNHN(nn.Module):
    def __init__(self, input_dim, vertex_dim, edge_dim, num_class, dropout=0.5):
        super(HNHN, self).__init__()
        self.dropout = dropout
        
        self.vtx_lin_1layer = torch.nn.Linear(input_dim, vertex_dim)
        self.vtx_lin = torch.nn.Linear(vertex_dim, vertex_dim)
        
        self.ve_lin = torch.nn.Linear(vertex_dim, edge_dim)
        self.ev_lin = torch.nn.Linear(edge_dim, vertex_dim)
        
        self.cls = nn.Linear(vertex_dim, num_class)

    def weight_fn(self, edges):
        weight = edges.src['reg_weight']/edges.dst['reg_sum']
        return {'weight': weight}
    
    def message_func(self, edges):
        #pdb.set_trace()
        return {'Wh': edges.src['Wh'], 'weight': edges.data['weight']}

    def reduce_func(self, nodes):
        Weight = nodes.mailbox['weight']
        #pdb.set_trace()
        aggr = torch.sum(Weight * nodes.mailbox['Wh'], dim=1)
        return {'h': aggr}

    def forward(self, g, vfeat, efeat, v_reg_weight, v_reg_sum, e_reg_weight, e_reg_sum, first_layer, last_layer):
        #pdb.set_trace()
        with g.local_scope():
            # feat_v = self.vtx_lin(vfeat)
            if first_layer:
                feat_v = self.vtx_lin_1layer(vfeat)
            else:
                feat_v = self.vtx_lin(vfeat)  
                
            feat_e = efeat

            g.ndata['h'] = {'node': feat_v}
            g.ndata['Wh'] = {'node' : self.ve_lin(feat_v)}
            g.ndata['reg_weight'] = {'node':v_reg_weight, 'edge':e_reg_weight}
            g.ndata['reg_sum'] = {'node':v_reg_sum, 'edge':e_reg_sum}
            
            # edge aggregation
            g.apply_edges(self.weight_fn, etype='in')
            g.update_all(self.message_func, self.reduce_func, etype='in')            
            feat_e = g.ndata['h']['edge']
            
            g.ndata['Wh'] = {'edge' : self.ev_lin(feat_e)}
            
            # node aggregattion
            g.apply_edges(self.weight_fn, etype='con')
            g.update_all(self.message_func, self.reduce_func, etype='con')
            feat_v = g.ndata['h']['node']
            if not last_layer :
                feat_v = F.dropout(feat_v, self.dropout)
            if last_layer:
                pred=self.cls(feat_v)
                return feat_v, feat_e, pred
            else:
                return [g, feat_v, feat_e, v_reg_weight, v_reg_sum, e_reg_weight, e_reg_sum]

class HyperAttn_ne(nn.Module):
    # edge attention  version
    def __init__(self, input_dim, query_dim, vertex_dim, edge_dim, num_class, dropout = 0.5):
        super(HyperAttn_ne, self).__init__()
        self.dropout = dropout
        
        self.query_dim = query_dim
        self.vtx_lin_1layer = torch.nn.Linear(input_dim, vertex_dim)
        self.vtx_lin = torch.nn.Linear(vertex_dim, vertex_dim)
        
        self.qe_lin = torch.nn.Linear(edge_dim, query_dim)
        self.kv_lin = torch.nn.Linear(vertex_dim, query_dim)
        self.vv_lin = torch.nn.Linear(vertex_dim, edge_dim)
        
        self.qv_lin = torch.nn.Linear(vertex_dim, query_dim)
        self.ke_lin = torch.nn.Linear(edge_dim, query_dim)
        self.ve_lin = torch.nn.Linear(edge_dim, vertex_dim)
        
        self.cls = nn.Linear(vertex_dim, num_class)

    def attention(self, edges):
        attn_score = F.leaky_relu((edges.src['k'] * edges.dst['q']).sum(-1))
        return {'Attn': attn_score/np.sqrt(self.query_dim)}
    
    def message_func(self, edges):
        return {'v': edges.src['v'], 'Attn': edges.data['Attn']}

    def reduce_func(self, nodes):
        attention_score = F.softmax((nodes.mailbox['Attn']), dim=1)
        aggr = torch.sum(attention_score.unsqueeze(-1) * nodes.mailbox['v'], dim=1)
        return {'h': aggr}

    def forward(self, g, vfeat, efeat, first_layer, last_layer):
        with g.local_scope():
            if first_layer:
                feat_v = self.vtx_lin_1layer(vfeat)
            else:
                feat_v = self.vtx_lin(vfeat)        
            
            feat_e = efeat

            # edge attention
            g.ndata['h'] = {'node': feat_v}
            g.ndata['k'] = {'node' : self.kv_lin(feat_v)}
            g.ndata['v'] = {'node' : self.vv_lin(feat_v)}
            g.ndata['q'] = {'edge' : self.qe_lin(feat_e)}
            g.apply_edges(self.attention, etype='in')
            g.update_all(self.message_func, self.reduce_func, etype='in')
            feat_e = g.ndata['h']['edge']
            
            # node attention
            g.ndata['k'] = {'edge' : self.ke_lin(feat_e)}
            g.ndata['v'] = {'edge' : self.ve_lin(feat_e)}
            g.ndata['q'] = {'node' : self.qv_lin(feat_v)}
            g.apply_edges(self.attention, etype='con')
            g.update_all(self.message_func, self.reduce_func, etype='con')
            feat_v = g.ndata['h']['node']
        
            if not last_layer :
                feat_v = F.dropout(feat_v, self.dropout)
            if last_layer:
                pred=self.cls(feat_v)
                return feat_v, feat_e, pred
            else:
                return [g, feat_v, feat_e]

    
class HyperAttn_n(nn.Module):
    def __init__(self, input_vdim, query_dim, vertex_dim, input_edim, edge_dim, n_cls, dropout = 0.5):
        super(HyperAttn_n, self).__init__()
        self.dropout = dropout
        self.query_dim = query_dim
        self.vtx_lin_1layer = torch.nn.Linear(input_vdim, vertex_dim)
        self.vtx_lin = torch.nn.Linear(vertex_dim, vertex_dim)
        self.edge_lin_1layer = torch.nn.Linear(input_edim, edge_dim)
        self.edge_lin = torch.nn.Linear(edge_dim, edge_dim)
        
        self.qv_lin = torch.nn.Linear(vertex_dim, query_dim)
        self.ke_lin = torch.nn.Linear(edge_dim, query_dim)
        self.ve_lin = torch.nn.Linear(edge_dim, vertex_dim)
        
        self.cls = nn.Linear(vertex_dim, n_cls)
        
    def attention(self, edges):
        attn_score = F.leaky_relu((edges.src['k'] * edges.dst['q']).sum(-1))
        return {'Attn': attn_score/np.sqrt(self.query_dim)}
    
    def message_func(self, edges):
        return {'v': edges.src['v'], 'Attn': edges.data['Attn']}

    def reduce_func(self, nodes):
        attention_score = F.softmax((nodes.mailbox['Attn']), dim=1)
        aggr = torch.sum(attention_score.unsqueeze(-1) * nodes.mailbox['v'], dim=1)
        return {'h': aggr}

    def forward(self, g, vfeat, efeat, first_layer, last_layer):
        with g.local_scope():
            if first_layer:
                feat_v = self.vtx_lin_1layer(vfeat)
                feat_e = self.edge_lin_1layer(efeat)
            else:
                feat_v = self.vtx_lin(vfeat)
                feat_e = self.edge_lin(efeat)

            # node attention
            g.ndata['h'] = {'node': feat_v, 'edge': feat_e}
            g.ndata['k'] = {'edge' : self.ke_lin(feat_e)}
            g.ndata['v'] = {'edge' : self.ve_lin(feat_e)}
            g.ndata['q'] = {'node' : self.qv_lin(feat_v)}
            g.apply_edges(self.attention, etype='con')
            g.update_all(self.message_func, self.reduce_func, etype='con')
            feat_v = g.ndata['h']['node']
            
            if not last_layer :
                feat_v = F.dropout(feat_v, self.dropout)
            if last_layer:
                pred=self.cls(feat_v)
                return feat_v, feat_e, pred
            else:
                return [g, feat_v, feat_e]

class Fused(nn.Module):
    def __init__(self, input_dim, query_dim, vertex_dim, edge_dim, num_class, dropout = 0.5):
        super(Fused, self).__init__()
        self.dropout = dropout
        self.query_dim = query_dim
        self.vtx_lin_1layer = torch.nn.Linear(input_dim, vertex_dim)
        self.vtx_lin = torch.nn.Linear(vertex_dim, vertex_dim)
        
        self.ve_lin = torch.nn.Linear(vertex_dim, edge_dim)
                
        self.qv_lin = torch.nn.Linear(vertex_dim, query_dim)
        self.ke_lin = torch.nn.Linear(edge_dim, query_dim)
        self.vale_lin = torch.nn.Linear(edge_dim, vertex_dim)
        self.cls = nn.Linear(vertex_dim, num_class)

    def weight_fn(self, edges):
        weight = edges.src['reg_weight']/edges.dst['reg_sum']
        return {'weight': weight}
    
    def message_func(self, edges):
        #pdb.set_trace()
        return {'Wh': edges.src['Wh'], 'weight': edges.data['weight']}

    def reduce_func(self, nodes):
        Weight = nodes.mailbox['weight']
        #pdb.set_trace()
        aggr = torch.sum(Weight * nodes.mailbox['Wh'], dim=1)
        return {'h': aggr}
    
    def attention(self, edges):
        attn_score = F.leaky_relu((edges.src['k'] * edges.dst['q']).sum(-1))
        return {'Attn': attn_score/np.sqrt(self.query_dim)}
    
    def attn_message_func(self, edges):
        return {'v': edges.src['v'], 'Attn': edges.data['Attn']}

    def attn_reduce_func(self, nodes):
        attention_score = F.softmax((nodes.mailbox['Attn']), dim=1)
        aggr = torch.sum(attention_score.unsqueeze(-1) * nodes.mailbox['v'], dim=1)
        return {'h': aggr}
    
    def forward(self, g, vfeat, efeat, v_reg_weight, v_reg_sum, e_reg_weight, e_reg_sum, first_layer, last_layer):
        with g.local_scope():
            if first_layer:
                feat_v = self.vtx_lin_1layer(vfeat)
            else:
                feat_v = self.vtx_lin(vfeat)

            feat_e = efeat
            
            g.ndata['h'] = {'node': feat_v}
            g.ndata['Wh'] = {'node' : self.ve_lin(feat_v)}
            g.ndata['reg_weight'] = {'node':v_reg_weight, 'edge':e_reg_weight}
            g.ndata['reg_sum'] = {'node':v_reg_sum, 'edge':e_reg_sum}
            
            # edge aggregation
            g.apply_edges(self.weight_fn, etype='in')
            g.update_all(self.message_func, self.reduce_func, etype='in')
            feat_e = g.ndata['h']['edge']
            
            # node attention
            g.ndata['k'] = {'edge' : self.ke_lin(feat_e)}
            g.ndata['v'] = {'edge' : self.vale_lin(feat_e)}
            g.ndata['q'] = {'node' : self.qv_lin(feat_v)}
            g.apply_edges(self.attention, etype='con')
            g.update_all(self.attn_message_func, self.attn_reduce_func, etype='con')
            
            feat_v = g.ndata['h']['node']
            
            if not last_layer :
                feat_v = F.dropout(feat_v, self.dropout)
            
            if last_layer:
                pred=self.cls(feat_v)
                return feat_v, feat_e, pred
            else:
                return [g, feat_v, feat_e, v_reg_weight, v_reg_sum, e_reg_weight, e_reg_sum]

class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout    
        self.v2e_1layer = torch.nn.Linear(in_ch, n_hid)
        self.v2e = torch.nn.Linear(n_hid, n_hid)
        self.cls = torch.nn.Linear(n_hid, n_class)
    
    def message_func(self, edges):
        return {'Wh': edges.src['Wh'], 'weight': edges.src['weight']}

    def reduce_func(self, nodes):
        Weight = nodes.mailbox['weight']
        aggr = torch.sum(Weight.unsqueeze(-1) * nodes.mailbox['Wh'], dim=1)
        return {'h': aggr}
    
    def weight_fn(self, edges):
        return {'weight_mul': edges.src['weight'] * edges.dst['weight']}

    def message_func2(self, edges):
        return {'h': edges.src['h'], 'weight_mul': edges.data['weight_mul']}

    def reduce_func2(self, nodes):
        weight = nodes.mailbox['weight_mul']
        aggr = torch.sum(weight.unsqueeze(-1) * nodes.mailbox['h'], dim=1)
        return {'h': aggr}


    def forward(self, vfeat, g, DV2, invDE, first_layer, last_layer):
        # x = fts = vfeat
        with g.local_scope():
            # vertex to edge gathering
            g.ndata['h'] = {'node': vfeat}
            g.ndata['weight'] = {'node' : DV2, 'edge' : invDE}
            if first_layer :
                g.ndata['Wh'] = {'node' : self.v2e_1layer(vfeat)}
            else :
                g.ndata['Wh'] = {'node' : self.v2e(vfeat)}
            g.update_all(self.message_func, self.reduce_func, etype='in')
            
            # edge to vertex gathering
            g.apply_edges(self.weight_fn, etype='con')
            g.update_all(self.message_func2, self.reduce_func2, etype='con')
            
            vfeat = g.ndata['h']['node']
            vfeat = F.relu(vfeat)
            
            if not last_layer :
                vfeat = F.dropout(vfeat, self.dropout)
            if last_layer:
                preds = self.cls(vfeat)
                preds = F.softmax(preds)
                return vfeat, preds
                #return preds
            else:
                return [vfeat, g, DV2, invDE]