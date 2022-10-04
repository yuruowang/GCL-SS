import numpy as np
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, HeteroData, Data, download_url, extract_zip
# import torch
from torch_geometric.utils import normalized_cut,add_self_loops,to_dense_adj, dense_to_sparse,is_undirected
#
# #
# edge_index = torch.tensor([[0, 1, 1, 4],
#                            [1, 0, 2, 2]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
#
# data = Data(x=x, edge_index=edge_index)
# adj = add_self_loops(edge_index)
#
# # print(adj)
# a = torch.ones(3,4)
# b = torch.ones(4,3)
# c = torch.tensor([[1,0,1],
#               [0,1,0],
#               [1,1,2]])
#
# c[:,1]
# aa = torch.mm(a,b)*c


# data = Data()
#
# data.edge_index = torch.LongTensor([[1,2,3,2],
#                                 [3,2,1,0]])
# data.edge_attr =  torch.tensor([1,3,4,1])
#
# aa = to_dense_adj(edge_index=data.edge_index,edge_attr=data.edge_attr)



ubfile = 'user_user.txt'
# col_names = ['user_id', 'item_id', 'relation']
col_names = ['user_id1', 'user_id2']
# col_names = ['user_id', 'user_group']   #group 2936
# col_names = ['book_id', 'year']
df = pd.read_csv(ubfile, sep='\t', names=col_names)

print(df)

