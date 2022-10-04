import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import normalized_cut,add_self_loops,to_dense_adj, dense_to_sparse, is_undirected,to_undirected

def Aug_UU(data):
    R_maxtix = torch.zeros((data.num_users, data.num_items))
    edge_index_ui = data.edge_index_ui
    edge_user = edge_index_ui[0][:int(edge_index_ui.shape[1] / 2)]
    edge_item = edge_index_ui[1][:int(edge_index_ui.shape[1] / 2)]
    index = (edge_user, (edge_item - data.num_users))
    new_value = torch.ones(edge_user.shape[0])
    R_maxtix = R_maxtix.index_put(index, new_value)

    S = to_dense_adj(data.edge_index_uu)[0]
    if S.shape[0] < data.num_users:
        S = torch.tensor(np.pad(S, ((0, data.num_users - S.shape[0]), (0, data.num_users - S.shape[0])), 'constant', constant_values=(0, 0)))
    # S = to_dense_adj(add_self_loops(data.edge_index_uu)[0])[0]  #添加自连接，防止孤立点
    S = torch.mm(R_maxtix,R_maxtix.T) * S
    edge_index_uu, edge_attr_uu = dense_to_sparse(S)
    # edge_attr_uu = normalized_cut(edge_index_uu, edge_attr_uu)
    # S_weight = to_dense_adj(edge_index=edge_index_uu, edge_attr=edge_attr_uu)[0]
    #得到uu邻接矩阵的字典
    S_dict = {i: np.nonzero(row).squeeze(1).tolist() for i, row in enumerate(S)}
    # print(S_dict)

    return edge_index_uu, edge_attr_uu, S, S_dict


def Aug_II(data):
    R_maxtix = torch.zeros((data.num_users, data.num_items))
    edge_index_ui = data.edge_index_ui
    edge_user = edge_index_ui[0][:int(edge_index_ui.shape[1] / 2)]
    edge_item = edge_index_ui[1][:int(edge_index_ui.shape[1] / 2)]
    index = (edge_user, (edge_item - data.num_users))
    new_value = torch.ones(edge_user.shape[0])
    R_maxtix = R_maxtix.index_put(index, new_value)

    C = to_dense_adj(data.edge_index_ii)[0]
    if C.shape[0] < data.num_items:
        C = torch.tensor(np.pad(C, ((0, data.num_items - C.shape[0]), (0, data.num_items - C.shape[0])), 'constant', constant_values=(0, 0)))
    # C = to_dense_adj(add_self_loops(data.edge_index_ii)[0])[0]  #添加自连接，防止孤立点
    C = torch.mm(R_maxtix.T, R_maxtix) * C
    edge_index_ii, edge_attr_ii = dense_to_sparse(C)
    # edge_attr_ii = normalized_cut(edge_index_ii,edge_attr_ii)
    # C_weight = to_dense_adj(edge_index=edge_index_ii, edge_attr=edge_attr_ii)[0]
    # 得到ii邻接矩阵的字典
    C_dict = {i: np.nonzero(row).squeeze(1).tolist() for i, row in enumerate(C)}
    # print(C_dict)

    return edge_index_ii, edge_attr_ii, C, C_dict


def trans_(edge_index, edge_attr,num):
    A = to_dense_adj(edge_index=edge_index,edge_attr=edge_attr).squeeze(0)
    if A.shape[0] < num:
        A = torch.tensor(np.pad(A, ((0, num - A.shape[0]), (0, num - A.shape[0])), 'constant', constant_values=(0, 0)))
    A_dict = {i: np.nonzero(row).squeeze(1).tolist() for i, row in enumerate(A)}

    return A, A_dict


def softmax(x):
    e_x = np.exp(x - np.max(x))  # for computation stability
    return e_x / e_x.sum()



def calc_rmse_mae(pred, gt):
    pred = F.softmax(pred, dim=1)
    expected_pred = torch.zeros(gt.shape)
    for relation in range(pred.shape[1]):
        expected_pred += pred[:, relation] * (relation + 1)

    rmse = (gt.to(torch.float) + 1) - expected_pred
    rmse = torch.pow(rmse, 2)
    rmse = torch.pow(torch.sum(rmse) / gt.shape[0], 0.5)

    mae = abs((gt.to(torch.float) + 1) - expected_pred)
    mae = torch.sum(mae) / gt.shape[0]

    return rmse, mae

def calc_rmse_mae_reg(pred, gt):
    pred = (pred.squeeze(1) - 0.01) * 4
    rmse = torch.sqrt(((pred - gt) ** 2).mean())
    mae = torch.abs((pred - gt)).mean()
    # rmse = (gt.to(torch.float) + 1) - expected_pred
    # rmse = torch.pow(rmse, 2)
    # rmse = torch.pow(torch.sum(rmse) / gt.shape[0], 0.5)
    return rmse, mae


def calc_rmse(pred, gt):
    pred = F.softmax(pred, dim=1)
    expected_pred = torch.zeros(gt.shape)
    for relation in range(pred.shape[1]):
        expected_pred += pred[:, relation] * (relation + 1)

    rmse = (gt.to(torch.float) + 1) - expected_pred
    rmse = torch.pow(rmse, 2)
    rmse = torch.pow(torch.sum(rmse) / gt.shape[0], 0.5)

    return rmse

def calc_rmse_reg(pred, gt):
    pred = (pred.squeeze(1) - 0.01) * 4
    rmse = torch.sqrt(((pred - gt) ** 2).mean())

    # rmse = (gt.to(torch.float) + 1) - expected_pred
    # rmse = torch.pow(rmse, 2)
    # rmse = torch.pow(torch.sum(rmse) / gt.shape[0], 0.5)
    return rmse



def calc_mae(pred, gt):
    pred = F.softmax(pred, dim=1)
    expected_pred = torch.zeros(gt.shape)
    for relation in range(pred.shape[1]):
        expected_pred += pred[:, relation] * (relation + 1)

    mae = abs((gt.to(torch.float) + 1) - expected_pred)
    mae = torch.sum(mae) / gt.shape[0]
    return mae

def calc_mae_reg(pred, gt):
    pred = (pred.squeeze(1) - 0.01) * 4
    mae = torch.abs((pred - gt)).mean()


    # rmse = (gt.to(torch.float) + 1) - expected_pred
    # rmse = torch.pow(rmse, 2)
    # rmse = torch.pow(torch.sum(rmse) / gt.shape[0], 0.5)
    return mae


def common_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cost = torch.mean((cov1 - cov2)**2)
    return cost


def loss_dependence(emb1, emb2, dim):
    R = torch.eye(dim) - (1/dim) * torch.ones(dim, dim)
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC


