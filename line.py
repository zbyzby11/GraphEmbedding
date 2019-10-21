"""
line算法的实现，一阶相似度，二阶相似度，1+2阶相似度
pytorch 实现相关算法
"""
import torch
from torch import nn


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
        self.g = graph.G
        self.node_size = graph.node_size
        self.dim = dim
        self.negative_ratio = negative_ratio
        self.batch_size = batch_size
        self.order = order
        # 节点的向量表示
        self.embedding = nn.Embedding(self.node_size, self.dim)
        # 节点的作为上下文的向量表示
        self.context_embedding = nn.Embedding(self.node_size, self.dim)
        # 初始化embedding矩阵
        self.init_embedding()

    def init_embedding(self):
        """
        初始化embeddign矩阵
        :return: None
        """
        nn.init.xavier_uniform_(self.embedding.weight.data)
        nn.init.xavier_uniform_(self.context_embedding.weight.data)

    def generate_batch(self):
        """生成器，用于产生指定数量的批量的数据"""
        yield None

    def negative_sample(self, batch_iter):
        '''负采样'''

