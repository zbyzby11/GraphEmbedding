"""
主函数，用来运行一些算法
"""
import time

from readGraph import CreateGraph
from deepwalk import DeepWalk


def main():
    graph = CreateGraph()
    graph.read_edgelist('./data/blogCatalog/bc_edgelist.txt')
    t = time.time()
    DeepWalk(graph)
    print(time.time() - t)


if __name__ == '__main__':
    main()
