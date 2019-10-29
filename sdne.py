"""
SDNE的算法实现, 节点的type为int类型
"""
import torch
import networkx as nx
import numpy as np
from torch import nn, optim
from readGraph import CreateGraph


class _SDNE(nn.Module):
    def __init__(self, node_size, encoder_list, alpha, beta, v, B, D):
        super(_SDNE, self).__init__()
        # encoder_list = [数字1，数字2，....]，对应encoder的隐藏层的维度
        self.encoder_list = encoder_list
        self.alpha = alpha
        self.beta = beta
        assert self.beta >= 1, "beta值必须大于1"
        self.v = v
        self.node_size = node_size
        # self.adj_mat, self.B = self.init_adj()
        # self.embedding = nn.Embedding(self.node_size, self.encoder_list[-1])
        self.encoder = nn.Sequential(*self.get_fc()[0])
        self.decoder = nn.Sequential(*self.get_fc()[1])
        print(self.encoder)
        self.B = B
        self.D = D
        # for para in self.named_parameters():
        #     print(para)

    def get_fc(self):
        encoder, decoder = [], []  # encoder和decoder定义的列表
        encoder_list = [self.node_size] + self.encoder_list
        decoder_list = list(reversed(self.encoder_list)) + [self.node_size]
        for num1, num2 in zip(range(len(encoder_list) - 1), range(len(decoder_list) - 1)):
            encoder.append(nn.Linear(encoder_list[num1], encoder_list[num1 + 1]))
            encoder.append(nn.Sigmoid())
            decoder.append(nn.Linear(decoder_list[num2], decoder_list[num2 + 1]))
            decoder.append(nn.Sigmoid())
        return encoder, decoder[:-1]

    def forward(self, x):
        # x为邻接矩阵，[n, n]
        # 经过encoder编码，[n, n] => [n, dim], 得到第k层的表示作为节点的embedding
        encoder_x = self.encoder(x)
        # print(encoder_x)
        # 经过decoder解码, [n, dim] => [n, n],重建出邻接矩阵
        decoder_x = self.decoder(encoder_x)
        # 求出2阶损失函数的值，是一个矩阵的F范数
        l_2nd = torch.norm(torch.mul(decoder_x - x, self.B), p=2)
        # 拉普拉斯矩阵L, [n, n]
        L = self.D - x
        # print(L)
        # 一阶损失的矩阵，[dim, n] * [n, n] * [n, dim] => [dim, dim]
        # with torch.no_grad():
        matrix_1st = encoder_x.t().mm(L).mm(encoder_x)
        # print(matrix_1st)
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
        print('1st: ', l_1st)
        print('2nd: ', l_2nd)
        print('reg: ',l_reg)
        return loss


class SDNE(object):
    def __init__(self, graph, encoder_list, alpha, beta, v, epoch, lr):
        self.device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
        self.g = graph.G
        self.encoder_list = encoder_list
        self.alpha = alpha
        self.beta = beta
        self.v = v
        self.epoch = epoch
        self.lr = lr
        self.node_size = graph.node_size
        self.adj_mat, self.B, self.D = self.init_adj()
        self.model = _SDNE(self.node_size, self.encoder_list, self.alpha, self.beta, self.v, self.B, self.D).to(
            self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def init_adj(self):
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
        mat_adj = torch.FloatTensor(mat_adj).to(self.device)
        B = torch.FloatTensor(B).to(self.device)
        D = torch.FloatTensor(D).to(self.device)
        return mat_adj, B, D

    def train(self):
        for epoch in range(self.epoch):
            loss = self.model(self.adj_mat)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('loss is: ', loss.item())


if __name__ == '__main__':
    g = CreateGraph()
    g.read_edgelist('./data/wiki/Wiki_edgelist.txt')
    s = SDNE(g, [1000, 128], 0.1, 5, 1e-5, 500, 0.001)
    s.train()
