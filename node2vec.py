"""
node2vec的算法实现
"""
from __future__ import absolute_import
import multiprocessing
import random
import time
import numpy as np
from readGraph import CreateGraph
from gensim.models import Word2Vec


class Node2Vec(object):
    def __init__(self, graph, **kwargs):
        """
        初始化函数，用来传入构建好的图和一些node2vec算法的参数
        :param graph: 建立好的一个图
        :param kwargs: 其余的参数，node2vec中有p、q、walk_length等参数
        """
        self.g = graph.G
        self.alias_edges = {}
        self.alias_nodes = {}
        self.vectors = dict()
        self.p = kwargs.get('p', 0.25)
        self.q = kwargs.get('q', 0.25)
        self.workers = kwargs.get('workers', 8)
        self.walk_length = kwargs.get('walk_length', 80)
        self.num_walks = kwargs.get('num_walks', 10)
        self.window_size = kwargs.get('window_size', 10)
        self.dim = kwargs.get('dim', 128)
        self.output = kwargs.get('output', './output.txt')
        self.node_size = graph.node_size
        self.summary()
        self.train()
        self.save_embedding()

    def summary(self):
        """
        进行一些模型的参数输出
        :return: 参数信息
        """
        print('===========================================================')
        print('===========================================================')
        print('图中的节点个数：{}'.format(self.node_size))
        print('node2vec 的p参数的值：{}'.format(self.p))
        print('node2vec 的q参数的值：{}'.format(self.q))
        print('模型训练并行数量：{}'.format(self.workers))
        print('node2vec 的序列游走长度：{}'.format(self.walk_length))
        print('node2vec 的游走的次数：{}'.format(self.num_walks))
        print('模型节点表示的维度：{}'.format(self.dim))
        print('skip-gram训练模型的窗口大小：{}'.format(self.window_size))
        print('node2vec 最后的结果保存路径：{}'.format(self.output))
        print('===========================================================')
        print('===========================================================')

    @staticmethod
    def alias_setup(probes):
        """
        alias sampling算法，直接copy过来用，声明成静态方法
        :param probes: 计算好的归一化的概率的矩阵（边的权重）
        :return:alias sampling的返回的两个矩阵
        """
        K = len(probes)
        q = np.zeros(K, dtype=np.float32)
        J = np.zeros(K, dtype=np.int32)
        smaller = []
        larger = []
        for kk, prob in enumerate(probes):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        return J, q

    @staticmethod
    def alias_draw(J, q):
        """
        静态方法，直接copy过来，alias算法的采样
        :param J: alias矩阵1
        :param q: alias矩阵2
        :return: 当前节点下一个游走的节点
        """
        K = len(J)
        kk = int(np.floor(np.random.rand() * K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]

    def init_weight(self):
        """
        node2vec算法中的概率矩阵的初始化，属于所有方法的一个前置方法
        :return:
        """
        G = self.g
        alias_nodes = {}
        for node in G.nodes():
            # 未归一化的概率矩阵，取出权重
            unnormlized_prob = [G[node][ner]['weight'] for ner in G.neighbors(node)]
            # 节点到节点的权重的总和
            number = sum(unnormlized_prob)
            # 使用每个节点到节点的权重除以权重总和得到归一化的权重矩阵，使从node出发到其邻居节点的概率为一
            normlized_prob = [float(prob_i) / number for prob_i in unnormlized_prob]
            alias_nodes[node] = self.alias_setup(normlized_prob)
        # 对边也进行这样的处理
        alias_edges = {}
        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

    def get_alias_edge(self, src, dst):
        """
        这个方法是node2vec算法的核心概率计算部分，计算node2vec的搜索概率
        :param src: 源节点
        :param dst: 目标节点
        :return: alias 采样的两个矩阵
        """
        G = self.g
        p = self.p
        q = self.q
        unnormalized_probs = []
        for dst_nbr in G.neighbors(dst):
            # 如果回到了源节点，即为1/p的概率回到源节点，对应论文中d_{tx} = 0
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
            # 如果为和源节点和当前节点都有边的节点，概率为1，对应论文中d_{tx} = 1
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            # 如果节点更远了，概率为1/q，对应论文中d_{tx} = 2
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
        # 将游走概率归一化
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        return self.alias_setup(normalized_probs)

    def node2vec_walk(self, walk_length, start_node):
        """
        node2vec算法的游走算法，对于单个节点的游走，对于大规模图可以并行优化
        :param walk_length: 游走的长度
        :param start_node: 开始的节点
        :return: 游走的序列
        """
        G = self.g
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                # 只有一个节点时，用node到node的概率判断下一个游走节点
                if len(walk) == 1:
                    walk.append(cur_nbrs[self.alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                # 如果游走序列大于1个，运用node2vec算法进行游走
                else:
                    prev = walk[-2]
                    pos = (prev, cur)
                    next = cur_nbrs[self.alias_draw(alias_edges[pos][0], alias_edges[pos][1])]
                    walk.append(next)
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        """
        node2vec的采样算法
        :param num_walks: 对所有节点采样多少次
        :param walk_length: 每个节点的采样序列长度
        :return: 所有节点的游走序列
        """
        G = self.g
        walks = []
        nodes = list(G.nodes())
        for walk_iter in range(num_walks):
            print('当前游走次数：{}'.format(walk_iter + 1))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(
                    walk_length=walk_length, start_node=node))
        return walks

    def train(self):
        """
        训练整个算法
        :return: None
        """
        t = time.time()
        print('开始初始化概率矩阵......')
        self.init_weight()
        print('初始化概率矩阵完成！开始节点的游走......')
        sentences = self.simulate_walks(self.num_walks, self.walk_length)
        corpus = []
        # 需要将sentence转换为str类型
        for idx, each in enumerate(sentences):
            corpus.append(list(map(str, sentences[idx])))
        print('节点游走完成！开始训练skip-gram模型......')
        word2vec = Word2Vec(sentences=corpus,
                            size=self.dim,
                            window=self.window_size,
                            min_count=0,
                            workers=self.workers,
                            sg=1
                            )
        print('模型训练完成！算法总时间消耗：{} seconds'.format(round(time.time() - t, 2)))
        for node in self.g.nodes():
            self.vectors[node] = word2vec.wv[str(node)]

    def save_embedding(self):
        """
        保存embedding向量
        :return: None
        """
        print('开始保存结果.....')
        fout = open(self.output, 'w')
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()
        print('结果保存完毕！')


if __name__ == '__main__':
    g = CreateGraph()
    g.read_edgelist('./data/wiki/Wiki_edgelist.txt')
    d = Node2Vec(g)
