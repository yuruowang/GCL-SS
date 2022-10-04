import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Movielens', help='Movielens, Douban_Book, Yelp')
    parser.add_argument('--dataset_path', default='./Movielens', help=' ')
    parser.add_argument('--train_ratio', type=float, default=0.8, help=' ')
    parser.add_argument('--embed_size', type=int, default=256, help='the dimension of embedding')  # 256 dimension
    parser.add_argument('--hidden_dim', type=int, default=64, help='the dimension of embedding')  # 64 dimension
    parser.add_argument('--alpha', type=float, default=0.3, help='')  #随机游走重置率
    parser.add_argument('--bidec_drop', type=bool, default=False, help='')  #
    parser.add_argument('--drop_prob', type=float, default=0.7, help='')  #
    parser.add_argument('--sample_length', type=int, default=10, help='')  #采样  长度
    parser.add_argument('--gcn_layers', type=int, default=2, help='')  #GCN层数
    parser.add_argument('--epochs', type=int, default=1200, help='the number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')  # [0.01, 0.001, 0.0005, 0.0001]
    parser.add_argument('--weight_decay', type=float, default=1e-10, help='weight_decay')  # [0.0, e-10, e-9, e-8]
    parser.add_argument('--alpha_loss', type=float, default=0.2, help='alpha_loss')  #自监督损失权重
    parser.add_argument('--bid', type=bool, default=True, help='alpha_loss')  #是否使用双线性解码器
    parser.add_argument('--regression', type=bool, default=False, help='')  #是否使用回归

    args = parser.parse_args()
    return args

    #0.8: 随机游走0.1；bidec_drop:False；采样长度10；自监督权重0.2    最优 0.9049
    #0.6: 随机游走0.3；bidec_drop:True；drop_prob:0.5；采样长度10；自监督权重0.2    最优0.938  drop_prob 0.5最优
        # 随机游走0.4 ；采样长度15