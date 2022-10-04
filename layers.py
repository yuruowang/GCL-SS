import numpy as np
import torch
import torch.nn as nn
from util import softmax
from parse import parse_args
from torch.nn import functional as F

args = parse_args()


class MY_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, node_num):
        super(MY_LSTM, self).__init__()
        self.node_num = node_num
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1)

    def random_walk(self, G, path_length, alpha=0, rand=np.random, start=None):
        """ Returns a truncated random walk.
            path_length: Length of the random walk.
            alpha: probability of restarts.
            start: the start node of the random walk.
        """
        # G = self #图G
        path = [start]

        # 只要没到最大长度
        while len(path) < path_length:
            cur = path[-1]

            node_neighbor = G[cur]
            if len(node_neighbor) == 0:
                node_neighbor.append(cur)
                # relevance_probability = np.ones(1)
            # else:
            #     relevance_probability = all_score[cur][node_neighbor]
            #     # convert tensor to numpy array
            #     relevance_probability = relevance_probability.cpu().detach().numpy()
            #     relevance_probability = softmax(relevance_probability)

            if len(G[cur]) > 0:
                # 按一定概率转移,重启
                if rand.random() >= alpha:
                    path.append(rand.choice(G[cur], size=1)[0])
                else:
                    path.append(path[0])
            else:
                break
        return path

    def random_walk_1(self, G, path_length, alpha=0, rand=np.random, start=None, all_score=None):
        """ Returns a truncated random walk.
            path_length: Length of the random walk.
            alpha: probability of restarts.
            start: the start node of the random walk.
        """
        # G = self #图G
        path = [start]

        # 只要没到最大长度
        while len(path) < path_length:
            cur = path[-1]

            node_neighbor = G[cur]
            if len(node_neighbor) == 0:
                node_neighbor.append(cur)
                relevance_probability = np.ones(1)
            else:
                relevance_probability = all_score[cur][node_neighbor]
                # convert tensor to numpy array
                relevance_probability = relevance_probability.cpu().detach().numpy()
                relevance_probability = softmax(relevance_probability)

            if len(G[cur]) > 0:
                # 按一定概率转移,重启
                if rand.random() >= alpha:
                    path.append(rand.choice(G[cur], size=1, p=relevance_probability)[0])
                else:
                    path.append(path[0])
            else:
                break
        return path


    def forward(self, x, all_score, dict):

        walk = []
        for i in range(self.node_num):
            # walk.append(self.random_walk(self.A, path_length=3, start=i))
            # walk.append(self.random_walk_1(G=dict, path_length=args.sample_length, alpha = args.alpha, start=i, all_score=all_score))
            walk.append(self.random_walk(G=dict, path_length=args.sample_length, alpha = args.alpha, start=i))
        # print(walk)
        k = np.array(walk, dtype=int)

        sort_feature = torch.transpose(x[k],0,1)
        out, (h, c) = self.lstm(sort_feature)

        return F.tanh(h[-1])


class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention,self).__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = F.softmax(self.dense_weight(stacked), dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs


class BiDecoder(nn.Module):
    def __init__(self, input_dim, num_relations):
        super(BiDecoder, self).__init__()
        self.num_relations = num_relations
        # self.linear = nn.Linear(input_dim,input_dim)
        self.linear = nn.ModuleList(nn.Linear(input_dim,input_dim) for i in range(num_relations))
        self.apply_drop = args.bidec_drop
        self.dropout = nn.Dropout(args.drop_prob)



    def forward(self,u_features, i_features, edge_user, edge_item):

        if self.apply_drop:
            u_features = self.dropout(u_features)
            i_features = self.dropout(i_features)

        edge_u = u_features[edge_user]
        edge_i = i_features[edge_item]
        for i in range(self.num_relations):
            if i == 0:
                x = torch.sum(torch.mul(self.linear[i](edge_u),edge_i),dim=1).unsqueeze(1)
            else:
                x = torch.cat((x, torch.sum(torch.mul(self.linear[i](edge_u),edge_i),dim=1).unsqueeze(1)),dim=1)

        return x

