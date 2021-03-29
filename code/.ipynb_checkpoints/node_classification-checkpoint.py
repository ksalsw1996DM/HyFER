import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
import torch.optim as optim

import utils
from data_load import gen_data
import models
from tqdm import tqdm
import pdb
import time

args = utils.parse_args()
dataset_name = args.dataset_name #'citeseer' 'cora'
device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'

f_log = open("n_logs/{}{}_layer{}_epoch{}_dropout{}_ab{}{}_veh{}{}{}_model_{}.log".format(dataset_name, args.label_percent, args.n_layers, args.epochs, args.dropout*10, args.alpha_v, args.alpha_e, args.dim_vertex, args.dim_edge, args.dim_query, args.model), "w")
f_log.write("args: {}\n".format(args))

exp_num = args.exp_num
log_epoch = 20
test_epoch = 20
acc_all = np.zeros(exp_num)

for j in tqdm(range(exp_num)):

    best_val_acc = 0
    test_acc = 0

    do_val=True

    g, args = gen_data(args, dataset_name, do_val=do_val)

    if args.model == "attn_ne":
        model = models.multilayers(models.HyperAttn_ne, [args.input_dim, args.dim_query, args.dim_vertex, args.dim_edge,  args.n_cls, args.dropout], args.n_layers)
    elif args.model == "attn_n":
        model = models.multilayers(models.HyperAttn_n, [args.input_dim, args.dim_query, args.dim_vertex, args.dim_edge, args.dim_edge, args.n_cls, args.dropout], args.n_layers)
    elif args.model == "hnhn":
        model = models.multilayers(models.HNHN, [args.input_dim, args.dim_vertex, args.dim_edge, args.n_cls, args.dropout], args.n_layers)
    elif args.model == 'fused':
        model = models.multilayers(models.Fused, [args.input_dim, args.dim_query, args.dim_vertex, args.dim_edge, args.n_cls, args.dropout], args.n_layers)
    elif args.model == 'hgnn':
        model = models.multilayers(models.HGNN, [args.input_dim, args.n_cls, args.dim_vertex, args.dropout], args.n_layers)
    model.to(device)
    
    optim = torch.optim.Adam(model.parameters(), lr=.004)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm(range(args.epochs)):
        # Training stage
        model.train()
        if args.model == "hnhn" or args.model == "fused":
            v, e, pred = model([g, args.v_feat, args.e_feat, args.v_reg_weight, args.v_reg_sum, args.e_reg_weight, args.e_reg_sum], args.n_layers)
        elif args.model == 'hgnn': 
            v, pred = model([args.v_feat, g, args.DV2, args.invDE], args.n_layers)
        else :
            v, e, pred = model([g, args.v_feat, args.e_feat], args.n_layers)

        #pdb.set_trace()
        pred = pred[args.label_idx]
        label = args.labels
        loss = loss_fn(pred, label)
        optim.zero_grad()
        loss.backward()
        optim.step()

        pred_cls = torch.argmax(pred, -1)
        acc = torch.eq(pred_cls, label).sum().item()/len(label)


        # log training and validation loss
        if epoch%test_epoch == 0 :
            f_log.write("{} epoch: Training loss : {} / Training acc : {}\n".format(epoch, loss, acc))
            model.eval()

            if args.model == "hnhn" or args.model == "fused":
                v, e, pred = model([g, args.v_feat, args.e_feat, args.v_reg_weight, args.v_reg_sum, args.e_reg_weight, args.e_reg_sum], args.n_layers)
            elif args.model == 'hgnn': 
                v, pred = model([args.v_feat, g, args.DV2, args.invDE], args.n_layers)
            else :
                v, e, pred = model([g, args.v_feat, args.e_feat], args.n_layers)
            pred = pred[args.val_idx]
            valid_labels = args.valid_labels
            loss = loss_fn(pred, valid_labels)
            pred_cls = torch.argmax(pred, -1)
            val_acc = torch.eq(pred_cls, label).sum().item()/len(valid_labels)
            f_log.write("{} epoch: Validation loss : {} / Validation acc : {}\n".format(epoch, loss, val_acc))

        # log test loss
        if epoch%test_epoch == 0:
            model.eval()
            if args.model == "hnhn" or args.model == "fused":
                v, e, pred = model([g, args.v_feat, args.e_feat, args.v_reg_weight, args.v_reg_sum, args.e_reg_weight, args.e_reg_sum], args.n_layers)
            elif args.model == 'hgnn': 
                v, pred = model([args.v_feat, g, args.DV2, args.invDE], args.n_layers)
            else :
                v, e, pred = model([g, args.v_feat, args.e_feat], args.n_layers)
            ones = torch.ones(len(args.all_labels))
            ones[args.label_idx] = -1
            ones[args.val_idx] = -1
            pred = pred[:len(args.all_labels)]
            test_pred = pred[ones>-1]
            test_labels = args.all_labels[ones>-1]
            test_cls = torch.argmax(test_pred, -1)
            loss=loss_fn(test_pred, test_labels)
            test_acc = torch.eq(test_cls, test_labels).sum().item()/len(test_labels)
            f_log.write("{} epoch: Test loss : {} / Test accuracy : {}\n".format(epoch, loss, test_acc))

        # save best validation loss and its test accuracy
        if best_val_acc<val_acc:
            best_val_acc = val_acc
            f_log.write("{} epoch is currently best with val acc {}\n".format(epoch, val_acc))
            best_test_acc=test_acc

    acc_all[j] = best_test_acc
print("test {}, max {}, min {}, aver {}, std {}".format(exp_num, acc_all.max(), acc_all.min(), acc_all.mean(), acc_all.std()))
f_log.write("test {}, max {}, min {}, aver {}, std {}".format(exp_num, acc_all.max(), acc_all.min(), acc_all.mean(), acc_all.std()))