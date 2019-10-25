"""
主函数，用来运行一些算法
"""
import time
import argparse
from readGraph import CreateGraph
from deepwalk import DeepWalk


def main():
    # 获取一些命令行参数
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--input', required=True, help='输入的图文件')
    parser.add_argument('--output', required=True, help='节点向量保存文件')
    parser.add_argument('--number-walks', default=10, type=int,
                        help='游走的次数，适用于node2vec、deepwalk')
    parser.add_argument('--walk-length', default=80, type=int,
                        help='每个节点游走长度，适用于node2vec、deepwalk')
    parser.add_argument('--workers', default=8, type=int,
                        help='并行处理的数量')
    parser.add_argument('--dim', default=128, type=int,
                        help='节点的向量维度')
    parser.add_argument('--window-size', default=10, type=int,
                        help='skip-gram模型的窗口大小，适用于node2vec、deepwalk')
    parser.add_argument('--epochs', default=5, type=int,
                        help='epoch的次数')
    parser.add_argument('--p', default=1.0, type=float)
    parser.add_argument('--q', default=1.0, type=float)
    parser.add_argument('--method', required=True, choices=[
        'node2vec',
        'deepwalk',
        'line',
    ], help='图表示学习算法')
    # graph = CreateGraph()
    # graph.read_edgelist('./data/blogCatalog/bc_edgelist.txt')
    # t = time.time()
    # DeepWalk(graph)
    # print(time.time() - t)


if __name__ == '__main__':
    main()
