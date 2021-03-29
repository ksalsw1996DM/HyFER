import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
import torch.optim as optim
from sklearn import metrics
import random

import utils
from utils import gen_HE_data
from data_load import gen_data, gen_DGLGraph
from data_load import add_edge, del_edge
import models
from tqdm import tqdm
import pdb
from sampler import UNSSampler, MNSSampler, SNSSampler,CNSSampler
from batch import DataLoader
from aggregator import *
import time

args = utils.parse_args()
device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
dataset_name = args.dataset_name #'citeseer' 'cora'

f_log = open("e_logs/{}_layer{}_epoch{}_dropout{}_ab{}{}_ns{}_aggr{}_model_{}.log".format(dataset_name, args.n_layers, args.epochs, args.dropout*10, args.alpha_v, args.alpha_e, args.nsampler, args.aggr, args.model), "w")
f_log.write("args: {}\n".format(args))

if args.fix_seed:
    np.random.seed(0)
    torch.manual_seed(0)

exp_num = args.exp_num
log_epoch = 1
test_epoch = 1
plot_epoch = args.epochs
print(exp_num)

acc_all = np.zeros(exp_num)

for j in tqdm(range(exp_num)):
    best_test_acc = 0
    test_acc = 0
    do_val = True
    data_dict = torch.load('/workspace/jupyter/data/splits/{}{}split{}.pt'.format(args.dataset_name, args.neg_ratio, j))
    args = gen_data(args, dataset_name, do_val=do_val)
    #ground
    ground = data_dict["ground_data"]
    #ground = [list(hedge) for hedge in ground]
    max_length = 0
    for hedge in ground:
        if len(hedge) > max_length:
            max_length = len(hedge)
    
    g = gen_DGLGraph(args, ground)
    
    # train positive & test data
    train_pos = data_dict["train_pos"]
    if args.nsampler == 'MNS' :
        train_neg = data_dict["train_mns"]
    elif args.nsampler == 'CNS' :
        train_neg = data_dict["train_cns"]
    elif args.nsampler == 'SNS' :
        train_neg = data_dict["train_sns"]
    else :
        train_neg = []
    test_pos = data_dict["test_set"]
    test_neg = data_dict["test_neg"]
    train_data = train_pos + train_neg
    test_data = test_pos + test_neg
    train_label = [1 for i in range(len(train_pos))] + [0 for i in range(len(train_neg))]
    test_label = [1 for i in range(len(test_pos))] + [0 for i in range(len(test_neg))]

    train_batchloader = DataLoader(train_data, train_label, args.bs, device, False)
    test_batchloader = DataLoader(test_data, test_label, args.bs, device, True)
    
    # model init
    if args.model == "attn_ne":
        model = models.multilayers(models.HyperAttn_ne, [args.input_dim, args.dim_query, args.dim_vertex, args.dim_edge,  args.n_cls], args.n_layers)
    elif args.model == "hnhn":
        model = models.multilayers(models.HNHN, [args.input_dim, args.dim_vertex, args.dim_edge, args.n_cls], args.n_layers)
    elif args.model == 'fused':
        model = models.multilayers(models.Fused, [args.input_dim, args.dim_query, args.dim_vertex, args.dim_edge, args.n_cls], args.n_layers)
    elif args.model == 'hgnn':
        args.dim_vertex = 128
        model = models.multilayers(models.HGNN, [args.input_dim, args.n_cls, args.dim_vertex], args.n_layers)
    model.to(device)
    Aggregator = None
    if args.aggr == 'mean':
        Aggregator = MeanAggregator(args.dim_vertex)
    elif args.aggr == 'set':
        Aggregator = SetAggregator(args.dim_vertex, args.num_heads)
    elif args.aggr == 'sagnn':
        Aggregator = SAGNNAggregator(args.dim_vertex, args.num_heads)
    elif args.aggr == 'maxmin':
        Aggregator = MaxminAggregator(args.dim_vertex)
    Aggregator.to(device)
    optim = torch.optim.Adam(list(model.parameters())+list(Aggregator.parameters()), lr=args.lr)
    loss_fn = nn.BCELoss()
    total_step = 0
    train_acc=0
    for epoch in tqdm(range(args.epochs)):
        # Training stage
        model.train()
        total_pred = []
        total_label = []
        num_data = 0
        total_loss = 0
        while True :
#             pdb.set_trace()
            hedges, labels, is_last = train_batchloader.next()
            batch_size = len(hedges)
            num_data+=batch_size
            add_edge(args, hedges, g)
            if args.model == "hnhn" or args.model == "fused":
                #pdb.set_trace()
                v_feat = args.v_feat[g.nodes('node')]
                e_feat = args.e_feat[g.nodes('edge')]
                v_reg_weight = args.v_reg_weight[g.nodes('node')]
                v_reg_sum = args.v_reg_sum[g.nodes('node')]
                e_reg_weight = args.e_reg_weight[g.nodes('edge')]
                e_reg_sum = args.e_reg_sum[g.nodes('edge')]
                v, e, pred = model([g, v_feat, e_feat, v_reg_weight, v_reg_sum, e_reg_weight, e_reg_sum], args.n_layers)
            elif args.model == 'hgnn': 
                v_feat = args.v_feat[g.nodes('node')]
                e_feat = args.e_feat[g.nodes('edge')]
                DV2 = args.DV2[g.nodes('node')]
                invDE = args.invDE[g.nodes('edge')]
                v, pred = model([v_feat, g, DV2, invDE], args.n_layers)
            else :
                v_feat = args.v_feat[g.nodes('node')]
                e_feat = args.e_feat[g.nodes('edge')]
                v, e, pred = model([g, v_feat, e_feat], args.n_layers)
            predictions = []
            for hedge in hedges :
                embeddings = v[hedge]
                pred, embed = Aggregator(embeddings)
                predictions.append(pred)
                total_pred.append(pred.detach())
            total_label.append(labels.detach())
            predictions = torch.stack(predictions)
            predictions = predictions.squeeze()
            if predictions.size() == torch.Size([1]) or predictions.size() == torch.Size([]) :
                    continue
            del_edge(args, hedges, g)
            try :
                loss = loss_fn(predictions, labels.float())
                total_loss+=loss.item()*batch_size
                optim.zero_grad()
                loss.backward()
                optim.step()
                #scheduler.step()
            except :
                pass
            torch.cuda.empty_cache()
            if is_last :
                break
        total_pred = torch.stack(total_pred)
        total_label = torch.cat(total_label, dim=0)
        total_pred = total_pred.squeeze()
        pred_cls = torch.round(total_pred)
        acc = torch.eq(pred_cls, total_label).sum().item()/len(total_label)
        loss = total_loss / num_data
        # log training and validation loss
        if epoch%log_epoch == 0:
            f_log.write("{} epoch: Training loss : {} / Training acc : {}\n".format(epoch, loss, acc))
            
        # log test loss
        if epoch%test_epoch == 0:
            model.eval()
            total_pred = []
            total_label = []
            num_data = 0
            total_loss = 0
            while True :
                hedges, labels, is_last = test_batchloader.next()
                batch_size = len(hedges)
                num_data+=batch_size
                add_edge(args, hedges, g)
                if args.model == "hnhn" or args.model == "fused":
                    v_feat = args.v_feat[g.nodes('node')]
                    e_feat = args.e_feat[g.nodes('edge')]
                    v_reg_weight = args.v_reg_weight[g.nodes('node')]
                    v_reg_sum = args.v_reg_sum[g.nodes('node')]
                    e_reg_weight = args.e_reg_weight[g.nodes('edge')]
                    e_reg_sum = args.e_reg_sum[g.nodes('edge')]
                    v, e, pred = model([g, v_feat, e_feat, v_reg_weight, v_reg_sum, e_reg_weight, e_reg_sum], args.n_layers)
                elif args.model == 'hgnn': 
                    v_feat = args.v_feat[g.nodes('node')]
                    e_feat = args.e_feat[g.nodes('edge')]
                    DV2 = args.DV2[g.nodes('node')]
                    invDE = args.invDE[g.nodes('edge')]
                    v, pred = model([v_feat, g, DV2, invDE], args.n_layers)
                else :
                    v_feat = args.v_feat[g.nodes('node')]
                    e_feat = args.e_feat[g.nodes('edge')]
                    v, e, pred = model([g, v_feat, e_feat], args.n_layers)
                predictions = []
                for hedge in hedges :
                    embeddings = v[hedge]
                    pred, embed= Aggregator(embeddings)
                    predictions.append(pred)
                    total_pred.append(pred.detach())
                total_label.append(labels.detach())
                predictions = torch.stack(predictions)
                predictions = predictions.squeeze()
                del_edge(args, hedges, g)
                try :
                    loss = loss_fn(predictions, labels.float())
                    total_loss+=loss.item()*batch_size
                except :
                    pass
                if is_last :
                    break
            total_pred = torch.stack(total_pred)
            total_label = torch.cat(total_label, dim=0)
            total_pred = total_pred.squeeze()
            pred_cls = torch.round(total_pred)
            test_acc = torch.eq(pred_cls, total_label).sum().item()/len(total_label)
            loss = total_loss / num_data

            roc_auc = metrics.roc_auc_score(total_label.cpu().numpy(), pred_cls.cpu().numpy())
            f_log.write("{} epoch: Test loss : {} / Test accuracy : {} / AUROC : {} \n".format(epoch, loss, test_acc, roc_auc))

            # save best validation loss and its test accuracy
            if best_test_acc<test_acc:
                best_test_acc = test_acc
                f_log.write("{} epoch is currently best with test acc {}\n".format(epoch, test_acc))
                f_log.flush()
                best_test_acc=test_acc
                best_roc_auc = roc_auc
                
    acc_all[j] = best_test_acc           
    print(best_test_acc, best_roc_auc)
print("test {}, max {}, min {}, aver {}, std {}".format(exp_num, acc_all.max(), acc_all.min(), acc_all.mean(), acc_all.std()))
print (args.label_percent)
f_log.write("test {}, max {}, min {}, aver {}, std {}".format(exp_num, acc_all.max(), acc_all.min(), acc_all.mean(), acc_all.std()))