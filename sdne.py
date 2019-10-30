"""
SDNE的算法实现, 节点的type为int类型
"""
import time

import torch
import networkx as nx
import numpy as np
from torch import nn, optim
from readGraph import CreateGraph


class _SDNE(nn.Module):
    def __init__(self, node_size, encoder_list, alpha, beta, v, B, D, output):
        """
        模型定义
        :param node_size: 图节点大小
        :param encoder_list: auto-encoder的维度，一个自定义list
        :param alpha: 超参数，控制一阶损失
        :param beta: 超参数，控制二阶损失
        :param v: 超参数，控制正则化损失
        :param B: 矩阵B，用于计算二阶损失的
        :param D: 对角矩阵，值为邻接矩阵的每一行的和
        :param output: 输出向量的文件
        """
        super(_SDNE, self).__init__()
        # encoder_list = [数字1，数字2，....]，对应encoder的隐藏层的维度
        self.encoder_list = encoder_list
        self.alpha = alpha
        self.beta = beta
        assert self.beta >= 1, "beta值必须大于1"
        self.v = v
        self.node_size = node_size
        self.encoder = nn.Sequential(*self.get_fc()[0])
        self.decoder = nn.Sequential(*self.get_fc()[1])
        self.B = B
        self.D = D
        self.output = output
        self.vector_list = None

    def get_fc(self):
        """
        建立auto-enocoder层
        :return: encoder层和decoder层
        """
        encoder, decoder = [], []  # encoder和decoder定义的列表
        encoder_list = [self.node_size] + self.encoder_list
        decoder_list = list(reversed(self.encoder_list)) + [self.node_size]
        for num1, num2 in zip(range(len(encoder_list) - 1), range(len(decoder_list) - 1)):
            encoder.append(nn.Linear(encoder_list[num1], encoder_list[num1 + 1]))
            encoder.append(nn.LeakyReLU(inplace=True))  # 实际测试，使用LeakyRelu效果比sigmoid好
            decoder.append(nn.Linear(decoder_list[num2], decoder_list[num2 + 1]))
            decoder.append(nn.LeakyReLU(inplace=True))
        return encoder, decoder[:-1]

    def forward(self, x):
        """
        前向传播，这边可以优化为批次的数据
        :param x: 邻接矩阵
        :return: loss
        """
        # x为邻接矩阵，[n, n]
        # 经过encoder编码，[n, n] => [n, dim], 得到第k层的表示作为节点的embedding
        encoder_x = self.encoder(x)
        # 经过decoder解码, [n, dim] => [n, n],重建出邻接矩阵
        decoder_x = self.decoder(encoder_x)
        # 求出2阶损失函数的值，是一个矩阵的F范数
        l_2nd = torch.norm(torch.mul(decoder_x - x, self.B), p=2)
        # 拉普拉斯矩阵L, [n, n]
        L = self.D - x
        # 一阶损失的矩阵，[dim, n] * [n, n] * [n, dim] => [dim, dim]
        matrix_1st = encoder_x.t().mm(L).mm(encoder_x)
        # 一阶损失函数的值
        l_1st = torch.trace(matrix_1st) * 2 * self.alpha
        # 正则化损失
        l_reg = 0
        for para_name, value in self.named_parameters():
            # 仅对w参数进行正则化
            if 'weight' in para_name:
                l_reg += torch.norm(value, p=2)
        l_reg = 1 / 2 * self.v * l_reg
        loss = l_1st + l_2nd + l_reg
        with torch.no_grad():
            # 获得encoder最后一层的向量表示
            self.vector_list = encoder_x.data.cpu().numpy().tolist()
        return loss

    def save_embedding(self):
        """
        保存节点的向量
        :return:
        """
        fout = open(self.output, 'w', encoding='utf8')
        for idx, emb in enumerate(self.vector_list):
            fout.write("{} {}\n".format(idx,
                                        ' '.join([str(x) for x in emb[1:]])))
        fout.close()


class SDNE(object):
    def __init__(self, graph,
                 encoder_list=[1000,128],
                 alpha=1e-6,
                 beta=5,
                 v=0.1,
                 epoch=1000,
                 lr=1e-2,
                 output_file='./out.txt'):
        """
        SDNE算法的实现
        :param graph: 图
        :param encoder_list: auto-encoder的list，形如[1000,500, ...,128],这里的最后一个数字就是节点的embedding向量维度
        :param alpha: 模型超参数
        :param beta: 模型超参数
        :param v: 模型超参数
        :param epoch: 迭代次数
        :param lr: 学习率
        :param output_file: 输出向量的文件
        """
        # 设备无关的tensor
        self.device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
        self.g = graph.G
        self.encoder_list = encoder_list
        self.alpha = alpha
        self.beta = beta
        self.v = v
        self.epoch = epoch
        self.lr = lr
        self.output_file = output_file
        # 节点大小
        self.node_size = graph.node_size
        # 图的邻接矩阵，论文中B矩阵， D矩阵用来求出拉普拉斯矩阵
        self.adj_mat, self.B, self.D = self.init_adj()
        # 模型的初始化
        self.model = _SDNE(self.node_size, self.encoder_list, self.alpha, self.beta, self.v, self.B, self.D, self.output_file).to(
            self.device)
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
        self.summary()

    def summary(self):
        """
        进行一些模型的参数输出
        :return: 参数信息
        """
        print('===========================================================')
        print('===========================================================')
        print('图中的节点个数：{}'.format(len(self.g.nodes())))
        print('sdne 学习率：{}'.format(self.lr))
        print('sdne 迭代次数：{}'.format(self.epoch))
        print('sdne encoder列表：{}'.format(self.encoder_list))
        print('sdne 节点向量维度：{}'.format(self.encoder_list[-1]))
        print('sdne 超参数alpha：{}'.format(self.alpha))
        print('sdne 超参数beta：{}'.format(self.beta))
        print('sdne 超参数v：{}'.format(self.v))
        print('sdne 节点向量维度：{}'.format(self.encoder_list[-1]))
        print('sdne 是否使用GPU：{}'.format('True' if self.device.startswith('cuda') else 'False'))
        print('sdne 最后的结果保存路径：{}'.format(self.output_file))
        print('===========================================================')
        print('===========================================================')

    def init_adj(self):
        """
        初始化邻接矩阵等图的特征
        :return:
        """
        # 将node索引进行排序，从0-1-2-.....升序
        node_list = sorted(list(map(int, list(self.g.nodes()))))
        # 提取出按node排序的邻接矩阵，用numpy稠密矩阵表示
        mat_adj = nx.adjacency_matrix(self.g, nodelist=node_list).todense()
        # B矩阵就是论文中的B矩阵，用来定义二阶损失的
        B = np.where(mat_adj == 0, 1, self.beta)
        # 拉普拉斯矩阵中的D，L=D-adj，D=对角矩阵，每个元素是这个节点邻接矩阵这行的总和
        D = np.zeros_like(mat_adj)
        for i in self.g.nodes():
            D[i, i] = np.sum(mat_adj[i])
        # 将这些转化成与设备无关的tensor
        mat_adj = torch.FloatTensor(mat_adj).to(self.device)
        B = torch.FloatTensor(B).to(self.device)
        D = torch.FloatTensor(D).to(self.device)
        return mat_adj, B, D

    def train(self):
        """
        训练模型
        :return: None
        """
        print('开始模型的训练......')
        t = time.time()
        for epoch in range(self.epoch):
            loss = self.model(self.adj_mat)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('epoch:{} || loss is: {}'.format(epoch, round(loss.item(), 3)))
        print('模型训练结束。总时间消耗为: {}秒'.format(round(time.time() - t, 3)))
        print('开始保存节点向量')
        # 保存节点向量
        self.model.save_embedding()
        print('节点向量保存完毕！')


if __name__ == '__main__':
    g = CreateGraph()
    g.read_edgelist('./data/wiki/Wiki_edgelist.txt')
    s = SDNE(g)
    s.train()
