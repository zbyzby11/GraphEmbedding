"""
从文件中或者用户手动读取一个图并且建立好一个图
"""
import networkx as nx


class CreateGraph(object):
    def __init__(self):
        self.G = None
        self.node_size = 0

    def read_edgelist(self, filename, is_weight=False):
        """
        从一个文件中读取出一个图并且建立,需要判断是否有边的权重,默认权重为False，即为1.0
        :param filename: 读取的格式化的文件
        :param is_weight: bool类型参数，表示是否有边的权重
        :return: 整个建立好的图
        """
        # 如果有边的权重
        if is_weight:
            fin = open(filename, 'r', encoding='utf8')
            self.G = nx.read_edgelist(filename)
            for line in fin:
                line = line.strip()
                src, tar, weight = line.split(' ')[0], line.split(' ')[1], line.split(' ')[2]
                self.G[src][tar]['weight'] = float(weight)
        # 如果边没有权重，则默认为1.0
        else:
            self.G = nx.read_edgelist(filename)
            for i, j in self.G.edges():
                self.G[i][j]['weight'] = 1.0


if __name__ == '__main__':
    g = CreateGraph()
    g.read_edgelist('./data/cora/cora_edgelist.txt')
