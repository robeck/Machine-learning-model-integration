import pandas as pd
import numpy as np
import networkx as nx
import random
import time
import matplotlib.pyplot as plt


def network_gen():
    graph = nx.watts_strogatz_graph(30, 3, 0.1)
    return graph


'''
传染学模型！
1）首先需要设置节点：suspect（易感染者），infect（感染者），recover（恢复者）
2）根据网络的连接结构（即相关人员，病例的关联情况）设置感染概率beta 和回复概率 alpha
3）网络关联结构为我们提供了良好的传染途径分析和结果展示

简单构建一个传染过程
1）首先感染者被救治  => 2）没有被治好的感染者会去传染易感染者  => 3)易感染者变成感染者
上述结构形成循环

:args
    beta: 治愈概率
    alpha：恢复概率
'''



'''
单传染模型：感染者只向邻居中的其中一个人进行传染

1）增加了传染过程的路径表标识
'''
def SIR_one(graph, beta=0.6, alpha=0.7, random_seed=10000):
    """单接触SIR模型
    传染率beta；恢复率alpha"""
    nodes = list(graph.nodes)
    # 生成感染者，易感染者，恢复者
    I_nodes = list(np.random.choice(nodes, 5))  # graph.nodes是图的节点，5是生成的节点的规模
    S_nodes = [node for node in nodes if node not in I_nodes]
    C_nodes = []

    t = 0

    while len(I_nodes) > 0:  # 如果还有感染者，将持续进行
        I_nodescopy = I_nodes.copy()

        for i in I_nodescopy:  # 循环感染者
            # I -> R
            print(f'感染者是{i}')
            rand = random.random()
            if rand < alpha:  # 感染者被治愈
                print(f'感染者被治愈了！')
                I_nodes.remove(i)  # 移除这个感染者
                C_nodes.append(i)  # 增加恢复者
            else:
                print('很不幸，感染者未被治愈！')

            print(f'感染者序列 {I_nodes}。 易感染者 {S_nodes}。 恢复者序列 {C_nodes}') #展示下哪些是感染者，哪些是恢复者

            time.sleep(1)

            # S -> I
            # 首先要寻找的就是节点的邻居
            temp = list(graph.neighbors(i))
            s = random.choice(temp)  # 随机选择感染者的邻居 (这里就是感染一个人）
            print(f'感染者的邻居：{s}')
            rand = random.random()
            if rand < beta:  # 易感染者感染的概率
                print('（易感染者）邻居会被感染')
                for s in S_nodes:  # 如果感染者的邻居是易感染者
                    print(f'向易感染者传染了 {s}')
                    S_nodes.remove(s)
                    I_nodes.append(s)

                    time.sleep(1)
            else:
                print('感染者未进行传染！')

            print(f'感染者序列 {I_nodes}。 易感染者 {S_nodes}。 恢复者序列 {C_nodes}') #展示下哪些是感染者，哪些是恢复者

            time.sleep(1)
        t += 1

    print(f'循环的周期为 {t} , 传染的整个过程完成了')
    return len(C_nodes)

'''
多传染模型：向所有邻居进行传染

'''
def sir_more(graph, beta=0.6, alpha=0.7):
    """全接触SIR模型
    传染率beta；恢复率alpha"""
    nodes = list(graph.nodes)
    # 生成感染者，易感染者，恢复者
    I_nodes = list(np.random.choice(nodes, 5))  # graph.nodes是图的节点，5是生成的节点的规模
    S_nodes = [node for node in nodes if node not in I_nodes]
    C_nodes = []

    t = 0

    while len(I_nodes) > 0:
        Ibase = I_nodes.copy()
        for i in Ibase:
            # I --> R
            if random.random() < alpha:
                I_nodes.remove(i)
                C_nodes.append(i)

            # S --> I
            tmp = list(graph.neighbors(i))  # 这里是找到所有邻居
            for s in tmp: # 感染所有邻居中的易感染者
                if random.random() < beta:
                    if s in S_nodes:
                        S_nodes.remove(s)
                        I_nodes.append(s)

        t += 1

    return len(C_nodes)


if __name__ == '__main__':
    graph = network_gen()
    print(graph.nodes)
    print(SIR_one(graph))
    # print(sir_more(graph))