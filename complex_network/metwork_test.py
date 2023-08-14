import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

print('this is for test')
print(np.log(10))


g = nx.Graph()

nodes = [1,2,3,4,5]
g.add_nodes_from(nodes)
edges = [(1,2),(2,3),(3,4)]
g.add_edges_from(edges)

print(g.nodes)
print(g.edges)

nx.draw(g)
plt.show()



