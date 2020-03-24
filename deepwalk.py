"""
deepwalk算法的实现
"""
from __future__ import absolute_import
import random
import time
from typing import List
from readGraph import CreateGraph
from gensim.models import Word2Vec
from multiprocessing.dummy import Pool


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
        self.node_size = graph.node_size
        self.output = kwargs.get('output', './out.txt')
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
        print('模型训练并行数量：{}'.format(self.workers))
        print('deepwalk 的序列游走长度：{}'.format(self.walk_length))
        print('deepwalk 的游走的次数：{}'.format(self.num_walks))
        print('模型节点表示的维度：{}'.format(self.dim))
        print('skip-gram训练模型的窗口大小：{}'.format(self.window_size))
        print('deepwalk 最后的结果保存路径：{}'.format(self.output))
        print('===========================================================')
        print('===========================================================')

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
        随机游走算法，采集图中点的序列，超参数walk_length，windows_size，大规模图可以并行优化
        :return: 二维list,随机游走生成的序列
        """
        walks = []  # 存储整个的游走序列
        # p = Pool(processes=self.workers)
        for walk_iter in range(self.num_walks):
            print('当前游走次数：{}'.format(walk_iter + 1))
            node_list = list(self.g.nodes())
            random.shuffle(node_list)
            for node in node_list:
                walks.append(self.get_sequence(node))
        return walks

    def train(self):
        """
        训练模型
        :return: None
        """
        t = time.time()
        print('开始进行随机游走！')
        sentence = self.random_walk()
        corpus = []
        # 需要将sentence转换为str类型
        for idx, each in enumerate(sentence):
            corpus.append(list(map(str, sentence[idx])))
        print('随机游走结束，开始进行模型训练！')
        word2vec = Word2Vec(sentences=corpus,
                            size=self.dim,
                            window=self.window_size,
                            min_count=0,
                            workers=self.workers,
                            sg=1
                            )
        for node in self.g.nodes():
            self.vectors[node] = word2vec.wv[str(node)]
        print('模型训练完成！算法总时间消耗：{} 秒'.format(round(time.time() - t, 3)))

    def save_embedding(self):
        """
        保存最后训练的节点向量
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
    d = DeepWalk(g)
