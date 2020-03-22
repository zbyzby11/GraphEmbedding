"""
semi-GCN的实现，半监督的方法
"""
import time

import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import random
from torch import nn, optim
from torch.nn.parameter import Parameter

from readGraph import CreateGraph


class GraphConvolutionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, a_hat):
        """
        图卷积层的实现
        :param input_dim: 输入的维度
        :param hidden_dim: 这一层隐藏的维度
        :param a_hat: 图的处理过后的拉普拉斯矩阵
        """
        super(GraphConvolutionLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.a_hat = a_hat
        # 参数W
        self.w = Parameter(torch.rand(self.input_dim, self.hidden_dim))

    def forward(self, x):
        """
        前向传播算法
        :param x: 输入x
        :return: 该图卷积层的输出
        """
        # x = [节点数量，每个节点的特征]
        x = torch.mm(x, self.w)
        x = torch.mm(self.a_hat, x)
        # 这里与论文中不一样
        output = F.leaky_relu(x)
        return output


class _GCN(nn.Module):
    """
    两层的一个GCN网络
    """

    def __init__(self, input_dim, output_dim, a_hat, num_class, dropout_ratio):
        """
        初始化GCN的原型
        :param input_dim: 输入的节点的特征的维度
        :param output_dim: 输出的节点的特征的维度，这里可以看做是节点的向量表示
        :param a_hat: 处理的拉普拉斯矩阵
        :param num_class: 标签的节点信息
        :param dropout_ratio: dropout率
        """
        super(_GCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.a_hat = a_hat
        self.num_class = num_class
        self.layers = nn.Sequential(GraphConvolutionLayer(self.input_dim, 512, self.a_hat),
                                    GraphConvolutionLayer(512, self.output_dim, self.a_hat))
        self.fc = nn.Linear(self.output_dim, num_class)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        """
        前向传播算法
        :param x: 输入x
        :return: 损失函数值out，第i层的输出out_x（这就是产生的节点embedding）
        """
        out_x = self.layers(x)
        out = self.dropout(out_x)
        out = self.fc(out)
        return out, out_x


class GCN(object):
    """
    GCN进行半监督节点分类的实现，如果有特征文件则读入特征文件，如果没有特征文件则将节点特征矩阵初始化为I
    """

    def __init__(self, graph, epoch=200, lr=1e-3, embedding_size=128, output_file='./out.txt', feature_file=None, label_file=None,
                 node_label_ratio=0.15, dropout_ratio=0.5):
        self.device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
        self.g = graph.G
        self.epoch = epoch
        self.lr = lr
        self.emb_dim = embedding_size
        self.dropout_ratio = dropout_ratio
        self.label_file = label_file
        assert label_file, '标签文件必须给定，必须要输入图中节点的标签！'
        self.node_label_ratio = node_label_ratio
        self.label_dict, self.num_class = self.get_node_label()
        self.a_hat = self.init_adj()
        self.feature_file = feature_file
        self.output_file = output_file
        if self.feature_file is None:
            self.X = torch.eye(len(self.g.nodes())).to(self.device)
        else:
            self.X = self.get_feature()
        self.model = _GCN(self.X.shape[1], self.emb_dim, self.a_hat, self.num_class, dropout_ratio).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.creation = nn.CrossEntropyLoss()
        self.summary()

    def summary(self):
        """
        进行一些模型的参数输出
        :return: 参数信息
        """
        print('===========================================================')
        print('===========================================================')
        print('图中的节点个数：{}'.format(len(self.g.nodes())))
        print('gcn 学习率：{}'.format(self.lr))
        print('gcn 迭代次数：{}'.format(self.epoch))
        print('gcn 节点向量维度：{}'.format(self.emb_dim))
        print('gcn 是否使用GPU：{}'.format('True' if self.device.startswith('cuda') else 'False'))
        print('gcn 最后的结果保存路径：{}'.format(self.output_file))
        print('gcn 节点特征文件：{}'.format(self.feature_file))
        print('gcn 节点标签文件：{}'.format(self.label_file))
        print('gcn 使用{}%的节点进行交叉熵训练'.format(self.node_label_ratio))
        print('gcn dropout率：{}'.format(self.dropout_ratio))
        print('===========================================================')
        print('===========================================================')

    def get_feature(self):
        """
        得到节点的特征矩阵，输入格式为node1 f1 f2 f3....
                                    node2 f1 f2 f3....
        :return:
        """
        result = []
        node_list = sorted(list(map(int, list(self.g.nodes()))))
        node_feature = dict()
        f = open(self.feature_file, 'r', encoding='utf8')
        for line in f:
            line = line.strip()
            line_list = line.split()
            node_feature[int(line_list[0])] = list(map(float, line_list[1:]))
        for node in node_list:
            result.append(node_feature[int(node)])
        result = torch.FloatTensor(result).to(self.device)
        print(node_feature)
        return result

    def get_node_label(self):
        """
        获取节点的标签，文件格式为node1 label
                                node2 label
                                node3 label
        :return: dict{node: label, node: label, ...}， num_class
        """
        # 总共有多少标签集合
        label_set = set()
        # 存储一定量的选取的节点的标签
        label_dict = {}
        # 将node索引进行排序，从0-1-2-.....升序
        node_list = sorted(list(map(int, list(self.g.nodes()))))
        node_tuple = list()
        f = open(self.label_file, 'r', encoding='utf8')
        for line in f:
            line = line.strip()
            node, label = line.split()[0], line.split()[1]
            node_tuple.append((node, label))
            label_set.add(label)
        sample_node = random.sample(node_tuple, int(self.node_label_ratio * len(node_tuple)))
        for node, label in sample_node:
            label_dict[int(node)] = int(label)
        return label_dict, len(label_dict)

    def init_adj(self):
        """
        初始化邻接矩阵等图的特征
        :return: a_hat矩阵
        """
        # 将node索引进行排序，从0-1-2-.....升序
        node_list = sorted(list(map(int, list(self.g.nodes()))))
        # 提取出按node排序的邻接矩阵，用numpy稠密矩阵表示
        mat_adj = nx.adjacency_matrix(self.g, nodelist=node_list).todense()
        # 根据论文加上一个单位矩阵
        A_ = mat_adj + np.eye(mat_adj.shape[0])
        # 拉普拉斯矩阵中的D，D=对角矩阵，每个元素是这个节点A-hat矩阵中这行的总和
        D = np.zeros_like(A_)
        for i in self.g.nodes():
            D[i, i] = np.sum(A_[i])
        # D^{-1/2}
        d = np.power(D, -0.5)
        d[np.isinf(d)] = 0.
        # a_hat = D^{-1/2} * A_ * D^{-1/2}
        a_hat = d.dot(A_).dot(d)
        # 将这些转化成与设备无关的tensor
        a_hat = torch.FloatTensor(a_hat).to(self.device)
        return a_hat

    def save_embedding(self, out_vector):
        """
        保存向量
        :param out_vector: GCN输出的向量
        :return: None
        """
        fout = open(self.output_file, 'w', encoding='utf8')
        for idx, emb in enumerate(out_vector):
            fout.write("{} {}\n".format(idx,
                                        ' '.join([str(x) for x in emb[1:]])))
        fout.close()

    def train(self):
        """
        模型的训练
        :return: None
        """
        t = time.time()
        labeled_node_list = list(self.label_dict.keys())
        label_list = torch.LongTensor(list(self.label_dict.values())).to(self.device)
        out_vector = None
        for epoch in range(self.epoch):
            output, out_vector = self.model(self.X)
            pred_label_matrix = output[labeled_node_list]
            loss = self.creation(pred_label_matrix, label_list)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('epoch {} || loss is :{}'.format(epoch, loss.item()))
        # GCN是否可以将特征矩阵看做节点向量？
        with torch.no_grad():
            # 是否使用归一化的向量表示
            # out_vector = out_vector / torch.norm(out_vector, dim=1, keepdim=True)
            out_vector = out_vector.data.cpu().numpy().tolist()
        # 保存向量
        print('开始保存向量......')
        self.save_embedding(out_vector)
        print('模型训练总共消耗时间：{}秒'.format(round(time.time()-t, 3)))


if __name__ == '__main__':
    g = CreateGraph()
    g.read_edgelist('./data/wiki/Wiki_edgelist.txt')
    model = GCN(graph=g, epoch=200, lr=0.0001, embedding_size=128,
                label_file='./data/wiki/wiki_labels.txt')
    model.train()
