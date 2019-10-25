"""
line算法的实现，一阶相似度，二阶相似度，1+2阶相似度
pytorch 实现相关算法，不实现按照边的权重调整节点的负采样的概率分布（按照随机选择负例点），
实现边采样优化算法
"""
from __future__ import absolute_import

import time

import torch
import math
import numpy as np
import random
import torch.nn.functional as F
from torch import nn, optim
from readGraph import CreateGraph


class _Line(nn.Module):
    """
    LINE算法的原型封装，继承自nn.Module模块，参数order指示使用几阶相似度优化，其中的order=3的算法需要同时优化
    一阶相似度和二阶相似度，最后的向量表示将这两个相似度计算得出的向量进行一个拼接
    """

    def __init__(self, graph, dim=128, negative_ratio=5, batch_size=32, order=3):
        """
        LINE算法的原型封装
        :param graph: 读入的图
        :param dim: embedding维度
        :param negative_ratio: 负采样个数
        :param batch_size: 批处理的节点的个数
        :param order: 相似度的阶数
        """
        super(_Line, self).__init__()
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        self.g = graph.G
        self.node_size = graph.node_size
        self.dim = dim
        self.negative_ratio = negative_ratio
        self.batch_size = batch_size
        self.order = order
        self.alias_edge_prob = {}  # 存储边的归一化的权重概率
        self.prob_table = {}  # 存储alias原本的概率
        self.alias_table = {}  # 存储alias补充的概率的来源
        # 节点的向量表示
        self.embedding = nn.Embedding(self.node_size, self.dim)
        # 节点的作为上下文的向量表示
        self.context_embedding = nn.Embedding(self.node_size, self.dim)
        # 初始化embedding矩阵
        self.init_embedding()
        # 初始化权重矩阵
        self.alias_setup()
        # self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def init_embedding(self):
        """
        初始化embedding矩阵
        :return: None
        """
        nn.init.xavier_uniform_(self.embedding.weight.data)
        nn.init.xavier_uniform_(self.context_embedding.weight.data)

    def alias_setup(self):
        """
        得到输入的图的归一化的边的权重
        :return: None
        """
        # 存储的是边的权重的概率字典，键为图的边（一个元组），值为这条边的归一化权重的概率
        edge_weight_dict = {}
        sum_weight = 0  # 存储总的边的权重
        for edge in self.g.edges():
            w = self.g[edge[0]][edge[1]]['weight']
            sum_weight += float(w)
        for edge in self.g.edges():
            edge_weight_dict[edge] = float(self.g[edge[0]][edge[1]]['weight']) / sum_weight
        self.alias_edge_prob = edge_weight_dict
        self.prob_table, self.alias_table = self.alias_sample(self.alias_edge_prob)

    def alias_draw(self, edges_list):
        """
        alias算法的最后一步，返回一个样本
        :param edges_list: 图中的边的列表[(1,2),(2,3),......]
        :return: 经过alias采样的边
        """
        sample = random.choice(edges_list)  # 从图中随机选择一个节点对（边）
        if self.prob_table[sample] >= np.random.rand():
            return sample
        else:
            return self.alias_table[sample]

    def generate_batch(self):
        """
        生成器，用于产生指定数量的批量的数据
        :return: 一批数据[(节点1，节点2)，(节点1，节点4)，.......]
        """
        data_ = []
        edges_list = list(self.g.edges())
        # sum_edges = len(edges_list)  # 总共有多少条边
        # n_batch = math.ceil(sum_edges / self.batch_size)  # 总共有多少个batch
        # for i in range(n_batch):
        #     yield edges_list[i:i+1]  # [(节点1，节点2)，(节点1，节点4)，.......]
        for i in range(self.batch_size):
            data_.append(self.alias_draw(edges_list))  # [(节点1，节点2)，(节点1，节点4)，.......]
        # print(data_)
        yield data_
        # data_.clear()
        # data_ = []

    def negative_sample(self):
        """
        负采样，实际是对节点对进行负采样
        :return: 采样好的列表，[[源节点，上下文节点，负采样1，负采样2，......],[...],[...],......]
        """
        # 保存结果的列表，第一个元素是源节点，第二个节点是上下文节点（邻居节点），后面所有的节点均是负采样的点
        result_list = []
        # 遍历整个节点对node_pair=(节点1，节点2)
        for node_pair in self.generate_batch():
            for src_node, target_node in node_pair:
                # print(node_pair)
                # src_node = node_pair[0]  # 源节点
                # target_node = node_pair[1]  # 与源节点相邻的节点
                # print(src_node)
                # print(target_node)
                # 进行负采样
                count = 0
                negative_list = []
                # for negative_num in range(self.negative_ratio):
                while count < self.negative_ratio:
                    node = random.choice(list(self.g.nodes()))
                    if not self.g.has_edge(src_node, node) and not self.g.has_edge(node, target_node):
                        negative_list.append(node)
                        count += 1
                        # result_list.append([src_node, target_node] + negative_list)
                        break
                    else:
                        continue
                result_list.append([src_node, target_node] + negative_list)
                # print(result_list)
        # print(len(result_list[0]))
        return result_list

    def alias_sample(self, prob_dict):
        """
        alias采样方法，采样边的权值方差很大时候的一种方法，O(1)的复杂度
        :param prob_dict: 概率的字典
        :return: alias算法产生的两个数组
        """
        num = len(prob_dict)  # alias采样中的个数
        prob_table = {}  # 存储的是alias算法中的原本的概率
        alias_table = {}  # 存储的是alias算法中补充的概率的来源，-1用来表示本身就够1了
        small, large = [], []  # small存储的是乘以num后小于1的，large存储大于等于1的
        for edge_, prob in prob_dict.items():
            # 将概率值乘以总个数
            prob_table[edge_] = num * prob
            # 如果这个值小于1，就是第一类，需要补充到1的类别
            if prob_table[edge_] < 1:
                small.append(edge_)
            # 否则是不需要补充到1的类别，需要将自己概率给别人
            else:
                large.append(edge_)
        # 进行alias算法中的两个数组的创建
        while len(small) > 0 and len(large) > 0:
            # 从small和large中各弹出一个
            s = small.pop()
            l = large.pop()
            # 改变prob_table的值，将值改成原本的概率，原本的概率的改变只会发生在large中
            prob_table[l] = prob_table[s] + prob_table[l] - 1.0
            # alias_table用来记录不到1概率的补充的概率来自哪一个
            alias_table[s] = l
            # 进行判断，large给过之后是否还是大于1，如果是，需要添加到large列表中继续
            # 如果不是，添加到small列表中待补充概率
            if prob_table[l] < 1.0:
                small.append(l)
            else:
                large.append(l)
        # 当输进去的概率和不为1的时候，要有之后的过程。
        while large:
            prob_table[large.pop()] = 1.0
        while small:
            prob_table[small.pop()] = 1.0
        return prob_table, alias_table

    def forward(self, v_i, v_j, negative):
        """
        前向传播
        :param v_i: 源节点
        :param v_j: 上下文节点
        :param negative: 负采样节点
        :return: loss
        """
        # v_i_emb = [batch_size, emb_size]
        v_i_emb = self.embedding(v_i).to(self.device)
        # print(v_i_emb.shape)
        # v_j_emb = [batch_size, emb_size]
        v_j_emb = self.context_embedding(v_j).to(self.device)
        # print(v_j_emb.shape)
        # negative_emb = [batch_size, negative_num, emb_size]
        negative_emb = self.context_embedding(negative).to(self.device)
        # print(negative_emb.shape)
        # 一阶相似度单独计算
        if self.order == 1:
            loss = torch.sigmoid(torch.sum(torch.mul(v_i_emb, v_j_emb), dim=1))
            return torch.mean(loss)
        # 二阶或者三阶相似度计算
        else:
            pos_score = F.logsigmoid(torch.sum(torch.mul(v_i_emb, v_j_emb), dim=1))
            # 广播机制，将原始节点的向量插入一维，这样就可以与negative向量相乘
            neg_score = torch.sum(
                F.logsigmoid(torch.sum(torch.mul(v_i_emb.view(self.batch_size, 1, -1), -negative_emb), dim=2)), dim=1)
            loss = -(pos_score + neg_score)
            return torch.mean(loss)

    def train_one_epoch(self):
        edges = len(list(self.g.edges()))
        sum_loss = 0.0
        n_batch = math.ceil(edges // self.batch_size)
        for n in range(n_batch):
            node_list = self.negative_sample()
            node_list = np.array(node_list).astype(np.int)
            # print(node_list.shape)
            node_list = torch.LongTensor(node_list)
            # print(node_list)
            # print(node_list[:, 0])
            v_i = node_list[:, 0]
            v_j = node_list[:, 1]
            negative = node_list[:, 2:]
            # print(v_i)
            sum_loss += self.forward(v_i, v_j, negative)
            # self.optimizer.zero_grad()
            # sum_loss.backward()
            # self.optimizer.step()
            # print(sum_loss)
        return sum_loss

    def get_embedding(self):
        vector = {}
        if self.device.startswith('cuda'):
            node_embedding = self.embedding.weight.cpu().detach().numpy().tolist()
        else:
            node_embedding = self.embedding.weight.detach().numpy().tolist()
        for idx, emb in enumerate(node_embedding):
            vector[idx] = node_embedding[idx]
        return vector


class Line(object):

    def __init__(self, graph, rep_size=128, batch_size=128, epoch=10, negative_ratio=5, lr=0.01, order=3,
                 output='./out.txt'):
        self.g = graph.G
        self.dim = rep_size
        self.bs = batch_size
        self.epoch = epoch
        self.negative_ratio = negative_ratio
        self.lr = lr
        self.order = order
        self.vector = {}
        self.output = output
        t = time.time()
        print('开始训练模型！')
        if order == 3:
            self.model1 = _Line(graph=graph, dim=rep_size // 2, negative_ratio=negative_ratio, batch_size=batch_size, order=1)
            self.model2 = _Line(graph=graph, dim=rep_size // 2, negative_ratio=negative_ratio, batch_size=batch_size, order=2)
            self.device = self.model2.device
            self.summary()
            print(self.device)
            for i in range(epoch):
                loss1 = self.model1.train_one_epoch()
                loss2 = self.model2.train_one_epoch()
                loss = loss1 + loss2
                optimizer = optim.Adam(self.model2.parameters(),
                                       lr=lr)
                # self.model1.zero_grad()
                self.model2.zero_grad()
                loss.backward()
                optimizer.step()
                print('epoch:{} || loss 为：{}'.format(i, loss.item()))
            vector1 = self.model1.get_embedding()
            vector2 = self.model2.get_embedding()
            print('模型训练完毕！保存节点的向量.....')
            for node in vector1.keys():
                self.vector[node] = np.append(vector1[node], vector2[node])
            self.save_embedding()
            print("节点向量保存完毕！")
        else:
            self.model = _Line(graph=graph, dim=rep_size, negative_ratio=negative_ratio, batch_size=batch_size, order=order)
            self.device = self.model.device
            self.summary()
            for i in range(epoch):
                loss = self.model.train_one_epoch()
                optimizer = optim.Adam(self.model.parameters(), lr=lr)
                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                print('epoch:{} || loss 为：{}'.format(i, loss.item()))
            self.vector = self.model.get_embedding()
            print('模型训练完毕！保存节点的向量.....')
            self.save_embedding()
            print("节点向量保存完毕！")
        print('模型总时间消耗为：{}秒'.format(round(time.time() - t, 3)))

    def summary(self):
        """
        进行一些模型的参数输出
        :return: 参数信息
        """
        print('===========================================================')
        print('===========================================================')
        print('图中的节点个数：{}'.format(len(self.g.nodes())))
        print('line 学习率：{}'.format(self.lr))
        print('line 迭代次数：{}'.format(self.epoch))
        print('line 每批数据的大小：{}'.format(self.bs))
        print('line 向量维度：{}'.format(self.dim))
        print('line 负采样个数：{}'.format(self.negative_ratio))
        print('line 是否使用GPU：{}'.format('True' if self.device.startswith('cuda') else 'False'))
        print('line 使用的相似度计算：{}'.format(self.order))
        print('line 最后的结果保存路径：{}'.format(self.output))
        print('===========================================================')
        print('===========================================================')

    def save_embedding(self):
        """
        保存词向量
        :return: None
        """
        fout = open(self.output, 'w')
        node_num = len(self.vector.keys())
        fout.write("{} {}\n".format(node_num, self.dim))
        for node, vec in self.vector.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()


if __name__ == '__main__':
    g = CreateGraph()
    g.read_edgelist('./data/wiki/Wiki_edgelist.txt')
    line = Line(g, epoch=200, batch_size=1000, lr=0.05, negative_ratio=5, order=2)