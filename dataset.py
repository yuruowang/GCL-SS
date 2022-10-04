import torch
from torch_scatter import scatter_add
from torch_geometric.data import InMemoryDataset, HeteroData, Data, download_url, extract_zip
import pandas as pd
import numpy as np
from torch_geometric.utils import is_undirected,to_dense_adj,to_undirected,normalized_cut
from parse import parse_args

def MLDataset_Homo(path, train_ratio):
    user_item_path = path + '/user_movie.txt'
    u_u_path = path + '/user_user(knn).txt'
    i_i_path = path + '/movie_movie(knn).txt'
    u_oc_u_path = path + '/metapaths/uocu_cos_w.txt'
    i_ge_i_path = path + '/metapaths/mgem_cos_w.txt'

    df, nums = create_df(user_item_path)
    #得到评分
    rating = torch.tensor(df['relation'].values)
    #得到user-item的边
    edge_user = torch.tensor(df['user_id'].values)    # user范围0-942
    edge_item = torch.tensor(df['item_id'].values) + nums['user']   # movie范围943-2624

    edge_index_ui = torch.stack((torch.cat((edge_user, edge_item), 0),    #user范围0-942，movie范围943-2624
                              torch.cat((edge_item, edge_user), 0)), 0)

    edge_index_ui = edge_index_ui.to(torch.long)
    # 节点id
    x = torch.arange(nums['node'], dtype=torch.long)
    # x_item  = torch.arange(nums['item'], dtype=torch.long)

    #得到user_user边(KNN)
    uu_df = uu_ii_df(u_u_path)
    edge_user1 = torch.tensor(uu_df['id1'].values)
    edge_user2 = torch.tensor(uu_df['id2'].values)
    edge_attr_uu = torch.tensor(uu_df['similarity'].values)
    edge_index_uu = torch.stack((edge_user1, edge_user2), 0).to(torch.long)
    edge_index_uu, edge_attr_uu = to_undirected(edge_index_uu, edge_attr_uu)
    # edge_index_uu = torch.stack((torch.cat((edge_user1, edge_user2), 0),  # user范围0-942
    #                               torch.cat((edge_user2, edge_user1), 0)), 0)
    # edge_index_uu = edge_index_uu.to(torch.long)

    # 得到元路径uocu的边
    uocu_df = uu_ii_df(u_oc_u_path)
    edge_user1 = torch.tensor(uocu_df['id1'].values)
    edge_user2 = torch.tensor(uocu_df['id2'].values)
    edge_attr_uocu = torch.tensor(uocu_df['similarity'].values)
    edge_index_uocu = torch.stack((edge_user1, edge_user2), 0).to(torch.long)
    # edge_index_uocu, edge_attr_uocu = to_undirected(edge_index_uocu, edge_attr_uocu)

    #得到item_item的边(KNN)
    ii_df = uu_ii_df(i_i_path)
    edge_item1 = torch.tensor(ii_df['id1'].values)
    edge_item2 = torch.tensor(ii_df['id2'].values)
    edge_attr_ii = torch.tensor(ii_df['similarity'].values)
    edge_index_ii = torch.stack((edge_item1, edge_item2), 0).to(torch.long)
    edge_index_ii, edge_attr_ii = to_undirected(edge_index_ii, edge_attr_ii)
    # edge_index_ii = torch.stack((torch.cat((edge_item1, edge_item2), 0),  # movie范围0-1681
    #                              torch.cat((edge_item2, edge_item1), 0)), 0)
    # edge_index_ii = edge_index_ii.to(torch.long)

    # 得到元路径mgem的边
    igei_df = uu_ii_df(i_ge_i_path)
    edge_item1 = torch.tensor(igei_df['id1'].values)
    edge_item2 = torch.tensor(igei_df['id2'].values)
    edge_attr_igei = torch.tensor(igei_df['similarity'].values)
    edge_index_igei = torch.stack((edge_item1, edge_item2), 0).to(torch.long)
    # edge_index_igei, edge_attr_igei = to_undirected(edge_index_igei, edge_attr_igei)

    #得到训练集，测试集
    train_mask, test_mask = mask(rating.shape[0], train_ratio)


    data = Data(x=x,
                edge_index_ui=edge_index_ui,
                edge_index_uu=edge_index_uu,
                edge_index_ii=edge_index_ii,
                edge_index_uocu = edge_index_uocu,
                edge_index_igei = edge_index_igei,
                # edge_attr_uu = edge_attr_uu,
                # edge_attr_ii = edge_attr_ii,
                edge_attr_uocu = edge_attr_uocu,
                edge_attr_igei = edge_attr_igei,
                y=rating)

    data.train_mask = train_mask
    data.test_mask = test_mask
    data.num_users = nums['user']
    data.num_items = nums['item']
    # data.edge_attr_ui = normalized_cut(edge_index_ui, torch.cat((rating,rating)))

    return data

# def MLDataset_Hete(path,train_ratio):
#     user_item_path = path + '/user_movie.txt'
#     u_u_path = path + '/user_user(knn).txt'
#     i_i_path = path + '/movie_movie(knn).txt'
#     df, nums = create_df(user_item_path)
#     #得到评分
#     rating = torch.tensor(df['relation'].values)
#     #得到user-item的边
#     edge_user = torch.tensor(df['user_id'].values)    # user范围0-942
#     edge_item = torch.tensor(df['item_id'].values)    # movie范围0-1681
#
#     # edge_index_ui = torch.stack((torch.cat((edge_user, edge_item), 0),    #user范围0-942，movie范围943-2624
#     #                           torch.cat((edge_item, edge_user), 0)), 0)
#     # edge_index_ui = edge_index_ui.to(torch.long)
#
#     # 节点id
#     x_user = torch.arange(nums['user'], dtype=torch.long)
#     x_item  = torch.arange(nums['item'], dtype=torch.long)
#
#     #得到user_user边(KNN)
#     uu_df = uu_ii_df(u_u_path)
#     edge_user1 = torch.tensor(uu_df['id1'].values)
#     edge_user2 = torch.tensor(uu_df['id2'].values)
#     edge_index_uu = torch.stack((edge_user1, edge_user2), 0).to(torch.long)
#     edge_index_uu = to_undirected(edge_index_uu)
#
#     #得到item_item的边
#     ii_df = uu_ii_df(i_i_path)
#     edge_item1 = torch.tensor(ii_df['id1'].values)
#     edge_item2 = torch.tensor(ii_df['id2'].values)
#     edge_index_ii = torch.stack((edge_item1, edge_item2), 0).to(torch.long)
#     edge_index_ii = to_undirected(edge_index_ii)
#
#
#
#     #得到训练集，测试集
#     train_mask, test_mask = mask(rating.shape[0], train_ratio)
#
#     data = HeteroData()
#     data['user'].x = x_user
#     data['item'].x = x_item
#     data['user', 'social', 'user'].edge_index = edge_index_uu
#     data['user', 'type1', 'item'].edge_index = torch.stack((edge_user,edge_item),0).to(torch.long)
#     data['item', 'type2', 'user'].edge_index = torch.stack((edge_item,edge_user),0).to(torch.long)
#     data['item', 'sim' , 'item'].edge_index = edge_index_ii
#     data.y = rating
#     data.train_mask = train_mask
#     data.test_mask = test_mask
#     data.num_users = nums['user']
#     data.num_items = nums['item']
#
#     return data

def DBDataset_Homo(path, train_ratio):
    user_item_path = path + '/user_book.txt'
    u_u_path = path + '/user_user.txt'
    i_i_path = path + '/book_book(knn).txt'
    u_oc_u_path = path + '/metapaths/ulu_cos.txt'
    i_ge_i_path = path + '/metapaths/bab_cos.txt'

    df, nums = create_df(user_item_path)
    #得到评分
    rating = torch.tensor(df['relation'].values)
    #得到user-item的边
    edge_user = torch.tensor(df['user_id'].values)    # user范围0-942
    edge_item = torch.tensor(df['item_id'].values) + nums['user']   # movie范围943-2624

    edge_index_ui = torch.stack((torch.cat((edge_user, edge_item), 0),    #user范围0-942，movie范围943-2624
                              torch.cat((edge_item, edge_user), 0)), 0)

    edge_index_ui = edge_index_ui.to(torch.long)
    # 节点id
    x = torch.arange(nums['node'], dtype=torch.long)
    # x_item  = torch.arange(nums['item'], dtype=torch.long)

    #得到user_user边(KNN)
    uu_df = uu_ii_df(u_u_path)
    edge_user1 = torch.tensor(uu_df['id1'].values)
    edge_user2 = torch.tensor(uu_df['id2'].values)
    # edge_attr_uu = torch.tensor(uu_df['similarity'].values)
    edge_index_uu = torch.stack((edge_user1, edge_user2), 0).to(torch.long)
    edge_index_uu = to_undirected(edge_index_uu)
    # edge_index_uu = torch.stack((torch.cat((edge_user1, edge_user2), 0),  # user范围0-942
    #                               torch.cat((edge_user2, edge_user1), 0)), 0)
    # edge_index_uu = edge_index_uu.to(torch.long)

    # 得到元路径uocu的边
    uocu_df = uu_ii_df(u_oc_u_path)
    edge_user1 = torch.tensor(uocu_df['id1'].values)
    edge_user2 = torch.tensor(uocu_df['id2'].values)
    edge_attr_uocu = torch.tensor(uocu_df['similarity'].values)
    edge_index_uocu = torch.stack((edge_user1, edge_user2), 0).to(torch.long)
    # edge_index_uocu, edge_attr_uocu = to_undirected(edge_index_uocu, edge_attr_uocu)

    #得到item_item的边(KNN)
    ii_df = uu_ii_df(i_i_path)
    edge_item1 = torch.tensor(ii_df['id1'].values)
    edge_item2 = torch.tensor(ii_df['id2'].values)
    # edge_attr_ii = torch.tensor(ii_df['similarity'].values)
    edge_index_ii = torch.stack((edge_item1, edge_item2), 0).to(torch.long)
    edge_index_ii = to_undirected(edge_index_ii)
    # edge_index_ii = torch.stack((torch.cat((edge_item1, edge_item2), 0),  # movie范围0-1681
    #                              torch.cat((edge_item2, edge_item1), 0)), 0)
    # edge_index_ii = edge_index_ii.to(torch.long)

    # 得到元路径mgem的边
    igei_df = uu_ii_df(i_ge_i_path)
    edge_item1 = torch.tensor(igei_df['id1'].values)
    edge_item2 = torch.tensor(igei_df['id2'].values)
    edge_attr_igei = torch.tensor(igei_df['similarity'].values)
    edge_index_igei = torch.stack((edge_item1, edge_item2), 0).to(torch.long)
    # edge_index_igei, edge_attr_igei = to_undirected(edge_index_igei, edge_attr_igei)

    #得到训练集，测试集
    train_mask, test_mask = mask(rating.shape[0], train_ratio)


    data = Data(x=x,
                edge_index_ui=edge_index_ui,
                edge_index_uu=edge_index_uu,
                edge_index_ii=edge_index_ii,
                edge_index_uocu = edge_index_uocu,
                edge_index_igei = edge_index_igei,
                # edge_attr_uu = edge_attr_uu,
                # edge_attr_ii = edge_attr_ii,
                edge_attr_uocu = edge_attr_uocu,
                edge_attr_igei = edge_attr_igei,
                y=rating)

    data.train_mask = train_mask
    data.test_mask = test_mask
    data.num_users = nums['user']
    data.num_items = nums['item']
    # data.edge_attr_ui = normalized_cut(edge_index_ui, torch.cat((rating,rating)))

    return data


def YPDataset_Homo(path, train_ratio):
    user_item_path = path + '/user_business.txt'
    u_u_path = path + '/user_user.txt'
    i_i_path = path + '/bus_bus(knn).txt'
    u_oc_u_path = path + '/metapaths/ucu_cos_1.txt'
    i_ge_i_path = path + '/metapaths/bcab_cos_2.txt'

    df, nums = create_df(user_item_path)
    #得到评分
    rating = torch.tensor(df['relation'].values)
    #得到user-item的边
    edge_user = torch.tensor(df['user_id'].values)    # user范围0-942
    edge_item = torch.tensor(df['item_id'].values) + nums['user']   # movie范围943-2624

    edge_index_ui = torch.stack((torch.cat((edge_user, edge_item), 0),    #user范围0-942，movie范围943-2624
                              torch.cat((edge_item, edge_user), 0)), 0)

    edge_index_ui = edge_index_ui.to(torch.long)
    # 节点id
    x = torch.arange(nums['node'], dtype=torch.long)
    # x_item  = torch.arange(nums['item'], dtype=torch.long)

    #得到user_user边(KNN)
    uu_df = uu_ii_df(u_u_path)
    edge_user1 = torch.tensor(uu_df['id1'].values)
    edge_user2 = torch.tensor(uu_df['id2'].values)
    # edge_attr_uu = torch.tensor(uu_df['similarity'].values)
    edge_index_uu = torch.stack((edge_user1, edge_user2), 0).to(torch.long)
    edge_index_uu = to_undirected(edge_index_uu)
    # edge_index_uu = torch.stack((torch.cat((edge_user1, edge_user2), 0),  # user范围0-942
    #                               torch.cat((edge_user2, edge_user1), 0)), 0)
    # edge_index_uu = edge_index_uu.to(torch.long)

    # 得到元路径uocu的边
    uocu_df = uu_ii_df(u_oc_u_path)
    edge_user1 = torch.tensor(uocu_df['id1'].values)
    edge_user2 = torch.tensor(uocu_df['id2'].values)
    edge_attr_uocu = torch.tensor(uocu_df['similarity'].values)
    edge_index_uocu = torch.stack((edge_user1, edge_user2), 0).to(torch.long)
    # edge_index_uocu, edge_attr_uocu = to_undirected(edge_index_uocu, edge_attr_uocu)

    #得到item_item的边(KNN)
    ii_df = uu_ii_df(i_i_path)
    edge_item1 = torch.tensor(ii_df['id1'].values)
    edge_item2 = torch.tensor(ii_df['id2'].values)
    # edge_attr_ii = torch.tensor(ii_df['similarity'].values)
    edge_index_ii = torch.stack((edge_item1, edge_item2), 0).to(torch.long)
    edge_index_ii = to_undirected(edge_index_ii)
    # edge_index_ii = torch.stack((torch.cat((edge_item1, edge_item2), 0),  # movie范围0-1681
    #                              torch.cat((edge_item2, edge_item1), 0)), 0)
    # edge_index_ii = edge_index_ii.to(torch.long)

    # 得到元路径mgem的边
    igei_df = uu_ii_df(i_ge_i_path)
    edge_item1 = torch.tensor(igei_df['id1'].values)
    edge_item2 = torch.tensor(igei_df['id2'].values)
    edge_attr_igei = torch.tensor(igei_df['similarity'].values)
    edge_index_igei = torch.stack((edge_item1, edge_item2), 0).to(torch.long)
    # edge_index_igei, edge_attr_igei = to_undirected(edge_index_igei, edge_attr_igei)

    #得到训练集，测试集
    train_mask, test_mask = mask(rating.shape[0], train_ratio)


    data = Data(x=x,
                edge_index_ui=edge_index_ui,
                edge_index_uu=edge_index_uu,
                edge_index_ii=edge_index_ii,
                edge_index_uocu = edge_index_uocu,
                edge_index_igei = edge_index_igei,
                # edge_attr_uu = edge_attr_uu,
                # edge_attr_ii = edge_attr_ii,
                edge_attr_uocu = edge_attr_uocu,
                edge_attr_igei = edge_attr_igei,
                y=rating)

    data.train_mask = train_mask
    data.test_mask = test_mask
    data.num_users = nums['user']
    data.num_items = nums['item']
    # data.edge_attr_ui = normalized_cut(edge_index_ui, torch.cat((rating,rating)))

    return data


def create_df(txt_path):
    col_names = ['user_id', 'item_id', 'relation', 'ts']
    df = pd.read_csv(txt_path, sep='\t', names=col_names)
    df = df.drop('ts', axis=1)
    df['user_id'] = df['user_id'] - 1
    df['item_id'] = df['item_id'] - 1
    df['relation'] = df['relation'] - 1

    nums = {'user': df.max()['user_id'] + 1,
            'item': df.max()['item_id'] + 1,
            'node': df.max()['user_id'] + df.max()['item_id'] + 2,
            'edge': len(df)}
    return df, nums

def create_df_db(txt_path):
    col_names = ['user_id', 'item_id', 'relation']
    df = pd.read_csv(txt_path, sep='\t', names=col_names)
    # df = df.drop('ts', axis=1)
    df['user_id'] = df['user_id'] - 1
    df['item_id'] = df['item_id'] - 1
    df['relation'] = df['relation'] - 1

    nums = {'user': df.max()['user_id'] + 1,
            'item': df.max()['item_id'] + 1,
            'node': df.max()['user_id'] + df.max()['item_id'] + 2,
            'edge': len(df)}
    return df, nums

def uu_ii_df(txt_path):
    col_names = ['id1', 'id2', 'similarity']
    df = pd.read_csv(txt_path, sep='\t', names=col_names)
    # df = df.drop('similarity', axis=1)
    df['id1'] = df['id1'] - 1
    df['id2'] = df['id2'] - 1

    return df

def mask(num, train_ratio):
    # n_rating = len(data)
    train_mask = torch.zeros([num])
    test_mask = torch.zeros([num])
    train_mask[:int(num * train_ratio)] = 1
    test_mask[int(num * train_ratio):] = 1

    train_mask = train_mask.type(torch.bool)
    test_mask = test_mask.type(torch.bool)
    return train_mask, test_mask


def mask1(num, train_ratio):
    # n_rating = len(data)
    train_mask = np.zeros(num)
    train_mask[:int(num * train_ratio)] = 1
    np.random.shuffle(train_mask) #打乱顺序
    train_mask = torch.tensor(train_mask)
    test_mask = 1 - train_mask
    # train_mask = torch.zeros([num])
    # test_mask = torch.zeros([num])
    # test_mask[int(num * train_ratio):] = 1
    train_mask = train_mask.type(torch.bool)
    test_mask = test_mask.type(torch.bool)
    return train_mask, test_mask

# MLDataset_Homo('.\Movielens',0.8)
# MLDataset_Hete('.\Movielens',0.8)
# DBDataset_Homo('.\Douban_Book',0.8)