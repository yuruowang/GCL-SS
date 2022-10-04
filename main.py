#两种监督

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataset import MLDataset_Homo, DBDataset_Homo,YPDataset_Homo
from parse import parse_args
from torch_geometric.utils import normalized_cut,add_self_loops,to_dense_adj, dense_to_sparse, is_undirected,to_undirected
from torch_geometric.nn.models import GCN,GAT
from util import Aug_II,Aug_UU, calc_rmse, calc_mae, trans_, calc_rmse_mae
from trainer import Trainer
from layers import MY_LSTM, Attention,BiDecoder

class MY_NET(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_relations, n_users, n_items, gcn_layers):
        super(MY_NET, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.uEmbd = nn.Embedding(n_users,input_dim)
        self.iEmbd = nn.Embedding(n_items,input_dim)
        self.lstm_uu = MY_LSTM(input_dim, hidden_dim1, n_users)
        self.lstm_ii = MY_LSTM(input_dim, hidden_dim1, n_items)
        self.gcn_uu = GCN(input_dim, hidden_dim1, num_layers=1, dropout=0.0, act=F.tanh)
        self.gcn_ii = GCN(input_dim, hidden_dim1, num_layers=1, dropout=0.0, act=F.tanh)

        self.gcn = GCN(in_channels=input_dim, hidden_channels=hidden_dim1, num_layers=gcn_layers, out_channels=hidden_dim2,  dropout=0.0, act=F.tanh)
        self.gcn1 = GCN(hidden_dim1, hidden_dim2, num_layers=1, dropout=0.0, act=F.tanh)
        # self.gcn1 = GAT(hidden_dim1, hidden_dim2, num_layers=1, dropout=0.0, act=F.tanh)
        # self.gcn2 = GCN(hidden_dim1, hidden_dim1, gcn_layers, dropout=0.0, act=F.tanh)
        # self.gat = GAT(in_channels=input_dim, hidden_channels=hidden_dim1, num_layers=gcn_layers, out_channels=hidden_dim2,  dropout=0.0, act=F.tanh)
        self.bidec = BiDecoder(hidden_dim2,num_relations)

        self.mlp_u = torch.nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim2*2),
            nn.Tanh(),
            nn.Linear(hidden_dim2*2, hidden_dim2)
        )
        self.mlp_i = torch.nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim2 * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim2 * 2, hidden_dim2)
        )
        self.mlp_uu = torch.nn.Sequential(
            nn.Linear(input_dim, hidden_dim1 * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim1 * 2, hidden_dim1)
        )
        self.mlp_ii = torch.nn.Sequential(
            nn.Linear(input_dim, hidden_dim1 * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim1 * 2, hidden_dim1)
        )
        # self.mlp_edge_predict = torch.nn.Sequential(
        #     nn.Linear(hidden_dim2*2, hidden_dim2),
        #     nn.Tanh(),
        #     nn.Linear(hidden_dim2, 1)
        # )

        # self.att_u1 =Attention(hidden_dim1)
        # self.att_i1 =Attention(hidden_dim1)
        self.att_u1 = nn.Linear(hidden_dim1*2,hidden_dim1)
        self.att_i1 = nn.Linear(hidden_dim1*2,hidden_dim1)
        self.att_u2 = Attention(hidden_dim2)
        self.att_i2 = Attention(hidden_dim2)

        self.linear = nn.Linear(hidden_dim2*2, num_relations)
        self.linear1 = nn.Linear(num_relations, 1)

    def getFeatureMat(self):
        uidx = torch.LongTensor([i for i in range(self.n_users)])
        iidx = torch.LongTensor([i for i in range(self.n_items)])
        userEmbd = self.uEmbd(uidx)
        itemEmbd = self.iEmbd(iidx)
        return userEmbd, itemEmbd

    def forward(self,edge_index_ui, S, S_dict,edge_index_uu, C, C_dict,edge_index_ii, UOCU, UOCU_dict,edge_index_uocu, IGEI, IGEI_dict,edge_index_igei):
        userEmbd, itemEmbd = self.getFeatureMat()
        # print(userEmbd)
        # 得到第二视图：uiu,iui元路径视图
        u_emb_v1 = self.lstm_uu(userEmbd, S, S_dict)
        i_emb_v1 = self.lstm_ii(itemEmbd, C, C_dict)

        # 得到第二视图：uocu,igei元路径视图
        u_emb_v2 = self.lstm_uu(userEmbd, UOCU, UOCU_dict)
        i_emb_v2 = self.lstm_ii(itemEmbd, IGEI, IGEI_dict)
        # u_emb_v1 = self.gcn_uu(userEmbd, edge_index_uu)
        # i_emb_v1 = self.gcn_ii(itemEmbd, edge_index_ii)
        #
        # u_emb_v2 = self.gcn_uu(userEmbd, edge_index_uocu)
        # i_emb_v2 = self.gcn_ii(itemEmbd, edge_index_igei)

        #换成mlp
        # u_emb_v1 = self.mlp_uu(userEmbd)
        # i_emb_v1 = self.mlp_ii(itemEmbd)
        #
        # u_emb_v2 = self.mlp_uu(userEmbd)
        # i_emb_v2 = self.mlp_ii(itemEmbd)

        # 两个uu视图和ii视图加权融合
        u_emb_merge = self.att_u1(torch.cat((u_emb_v1, u_emb_v2), dim=1))
        i_emb_merge = self.att_i1(torch.cat((i_emb_v1, i_emb_v2), dim=1))

        ui_emb_v3 = self.gcn1(torch.vstack((u_emb_merge, i_emb_merge)), edge_index_ui)
        u_emb_v3 = ui_emb_v3[:self.n_users]
        i_emb_v3 = ui_emb_v3[self.n_users:]

        # 得到第二视图的u - i特征
        ui_emb_v4 = torch.vstack((userEmbd, itemEmbd))
        ui_emb_v4 = self.gcn(ui_emb_v4, edge_index_ui)  # 做GCN
        # ui_emb_v4 = self.gcn1(ui_emb_v4, edge_index_ui)  #做GCN1
        # ui_emb_v4 = self.gat(ui_emb_v4, edge_index_ui)  #做GAT
        u_emb_v4 = ui_emb_v4[:self.n_users]
        i_emb_v4 = ui_emb_v4[self.n_users:]

        # mlp,两视图共享
        # u_emb_v1 = self.mlp_u(u_emb_merge)
        # u_emb_v2 = self.mlp_u(u_emb_v3)
        # i_emb_v1 = self.mlp_i(i_emb_merge)
        # i_emb_v2 = self.mlp_i(i_emb_v3)

        # 得到总的ui特征
        u_emb = self.att_u2([u_emb_v3, u_emb_v4])
        i_emb = self.att_i2([i_emb_v3, i_emb_v4])
        u_emb = self.mlp_u(u_emb)
        i_emb = self.mlp_i(i_emb)
        # 得到每条边的类别
        edge_user = edge_index_ui[0][:int(edge_index_ui.shape[1] / 2)]
        edge_item = edge_index_ui[1][:int(edge_index_ui.shape[1] / 2)] - self.n_users
        if args.bid == True:
            edge_emb = self.bidec(u_emb, i_emb, edge_user, edge_item)
        elif args.regression == True:
            edge_emb = self.bidec(u_emb, i_emb, edge_user, edge_item)
            edge_emb = self.linear1(edge_emb)
        else:
            edge_u = u_emb[edge_user]
            edge_i = i_emb[edge_item]
            edge_emb = self.linear(torch.cat((edge_u, edge_i), dim=1))

        return u_emb_v1, i_emb_v1, u_emb_v2, i_emb_v2, u_emb_v3, i_emb_v3, u_emb_v4, i_emb_v4, edge_emb



if __name__ == "__main__":

    args = parse_args()
    if args.dataset == 'Movielens':
        data = MLDataset_Homo(args.dataset_path, args.train_ratio)
    elif args.dataset == 'Douban_Book':
        data = DBDataset_Homo(args.dataset_path, args.train_ratio)
    else:
        data = YPDataset_Homo(args.dataset_path, args.train_ratio)
    # data = MLDataset_Homo(args.dataset_path, args.train_ratio)
    print(args.train_ratio)
    # Hete_data = MLDataset_Hete(args.dataset_path, args.train_ratio)
    # data = data.cuda()
    # 增强UU
    data.edge_index_uu, data.edge_attr_uu, data.S, data.S_dict = Aug_UU(data)
    # 增强II
    data.edge_index_ii, data.edge_attr_ii, data.C, data.C_dict = Aug_II(data)

    # data.S, data.S_dict = trans_(data.edge_index_uu, data.edge_attr_uu)
    # data.C, data.C_dict = trans_(data.edge_index_ii, data.edge_attr_ii)
    # 得到uocu以及mgem两种元路径的权重阵和邻接表
    data.UOCU, data.UOCU_dict = trans_(data.edge_index_uocu, data.edge_attr_uocu, data.num_users)
    data.IGEI, data.IGEI_dict = trans_(data.edge_index_igei, data.edge_attr_igei, data.num_items)

    # f = open("record/record.txt", 'a')
    # f.write("min_record: ")
    # f.write("\n")
    for i, alpha in enumerate([0.1,0.2,0.3]):
        # print(data)
        model = MY_NET(input_dim = args.embed_size,
                       hidden_dim1 = args.hidden_dim,
                       hidden_dim2 = args.hidden_dim,
                       num_relations = int(data.y.max()) + 1,
                       n_users = data.num_users,
                       n_items = data.num_items,
                       gcn_layers = args.gcn_layers)

        # optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr, weight_decay=args.weight_decay,
        )

        # train
        trainer = Trainer(
            model, data, calc_rmse_mae, optimizer, alpha, args.regression
        )
        best_test_rmse, best_test_mae = trainer.training(args.epochs)
        # f.write("alpha:"+ str(alpha) +'\t'+ ",Best test rmse:"+  str(best_test_rmse) + '\t' + ",Best test mae:" + str(best_test_mae) + '\n')
        # f.write("\n")
        # f.flush()
    print("Ending")
    # f.close()



    # logits = model(data,data.edge_index_ui)
