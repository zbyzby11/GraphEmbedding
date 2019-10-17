"""
deepwalk算法的实现
"""
from __future__ import absolute_import
import multiprocessing
import random
import time
from typing import List
from readGraph import CreateGraph
from gensim.models import Word2Vec


class DeepWalk(object):
    def __init__(self, graph, **kwargs):
        """
        初始化函数，传入的参数为一个建立好的图
        :param graph: 建立好的图
        :param kwargs: workers:进程个数，walk_length：随机游走的长度，num_walks：每个节点游走序列的个数
                        window_size:word2vec模型的窗口大小，dim；向量维度, output：输出向量结果的文件
        """
        self.g = graph.G
        self.vectors = dict()
        self.workers = kwargs.get('workers', 8)
        self.walk_length = kwargs.get('walk_length', 80)
        self.num_walks = kwargs.get('num_walks', 10)
        self.window_size = kwargs.get('window_size', 10)
        self.dim = kwargs.get('dim', 128)
        self.output = kwargs.get('output', './output.txt')
        self.train()
        self.save_embedding()
        # self.node_size = self.g.node_size

    def get_sequence(self, start_node) -> List[int]:
        """
        对单个节点进行随机游走算法
        :param start_node: 开始节点
        :return: 一条游走序列
        """
        walk = [start_node]  # 初始化的walk序列
        while len(walk) < self.walk_length:
            current_node = walk[-1]  # 当前的节点
            neighbors = list(self.g.neighbors(current_node))  # 取出当前游走节点的邻居节点
            if len(neighbors) > 0:
                walk.append(random.choice(neighbors))
            else:
                break
        return walk

    def random_walk(self) -> List[List[int]]:
        """
        随机游走算法，采集图中点的序列，超参数walk_length，windows_size
        :return: 二维list,随机游走生成的序列
        """
        walks = []  # 存储整个的游走序列
        result = []
        p = multiprocessing.Pool(processes=self.workers)
        for walk_iter in range(self.num_walks):
            node_list = list(self.g.nodes())
            random.shuffle(node_list)
            for node in node_list:
                result.append(p.apply_async(self.get_sequence, (node,)))
        p.close()
        p.join()
        for res in result:
            temp_walk = res.get()
            walks.append(temp_walk)
        return walks

    def train(self):
        print('开始进行随机游走！')
        sentence = self.random_walk()
        print('随机游走结束，开始进行模型训练！')
        word2vec = Word2Vec(sentences=sentence,
                            size=self.dim,
                            window=self.window_size,
                            min_count=0,
                            workers=self.workers,
                            sg=1
                            )
        for node in self.g.nodes():
            self.vectors[node] = word2vec.wv[node]
        print('模型训练完成！')

    def save_embedding(self):
        print('开始保存结果.....')
        fout = open(self.output, 'w')
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()
        print('结果保存完毕！')


if __name__ == '__main__':
    g = CreateGraph()
    g.read_edgelist('./data/blogCatalog/bc_edgelist.txt')
    d = DeepWalk(g)
    t = time.time()
    x = d.random_walk()
    print(list(x))
    print(time.time() - t)
