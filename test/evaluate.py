"""
评估模型的效果
"""
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

random.seed(12)


class EvaluateModel(object):

    def __init__(self, embedding_file, label_file, split_ratio):
        """
        初始化
        :param embedding_file: 保存的embedding向量文件
        :param label_file: 保存的每个节点的标签文件
        :param split_ratio: 划分测试集的比例
        """
        self.embedding_file = embedding_file
        self.label_file = label_file
        self.split_ratio = split_ratio

    def get_label(self):
        """
        得到输入的标签文件
        :return: 字典，键为node，值为该node的标签
        """
        f = open(self.label_file, 'r', encoding='utf8')
        label_dict = {}
        for line in f:
            line = line.strip()
            node, real_label = int(line.split()[0]), int(line.split()[1])
            label_dict[node] = real_label
        return label_dict

    def get_embedding(self):
        """
        得到node的embedding
        :return: 字典，键为ndoe，值为list，表示该node的向量
        """
        f = open(self.embedding_file, 'r', encoding='utf8')
        embedding_dict = {}
        for line in f:
            cur_list = []
            line = line.strip()
            emb = line.split()
            cur_list.extend(list(map(float, emb[1:])))
            embedding_dict[int(emb[0])] = cur_list
        return embedding_dict

    def split_train_test(self):
        """
        划分测试集
        :return: 划分好的四个集合
        """
        label_dict = self.get_label()
        embedding_dict = self.get_embedding()
        node_list = sorted(list(label_dict))
        x_list = []  # 存储每个节点向量信息的列表
        y_list = []  # 标签信息的列表
        for node in node_list:
            x_list.append(embedding_dict[node])
            y_list.append(label_dict[node])

        x_train, x_test, y_train, y_test = train_test_split(x_list, y_list, test_size=self.split_ratio)
        return x_train, x_test, y_train, y_test

    def train(self, average='micro'):
        """
        训练并且评估模型
        :param average: 使用f1的那种评价指标评价
        :return: None
        """
        x_train, x_test, y_train, y_test = self.split_train_test()
        clf = LogisticRegression()
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        f1 = f1_score(y_test, y_predict, average=average)
        print('模型的f1值为:{},使用average={}'.format(f1, average))


if __name__ == '__main__':
    l = EvaluateModel('../out.txt', '../data/wiki/wiki_labels.txt', 0.5)
    l.train(average='micro')
