import networkx as nx
import numpy as np 
import matplotlib.pyplot as plt
index = np.arange(1,10,1)
index_2 = 10 - np.arange(1,10,1)
G = nx.DiGraph()
G.add_node("Root")
N = 10
node_name = 'Root'
node_name_10 = 'Root'
for index_i,index_j in zip(index,index_2):

    previous = node_name 
    previous_10 = node_name_10
    node_name = index_i + index_i*10
    node_name_10 = index_j + index_i*10
    G.add_node(node_name, price = index)
    G.add_node(node_name_10, prince  = index_2)
    G.add_edge(previous,node_name,probability=1.0/N)
    G.add_edge(previous_10,node_name_10,probability=1.0/N)
# model = ScenarioTreeModelFromNetworkX(G)
plt.figure(figsize=(10,10))
nx.draw(G, with_labels=True)

plt.show()
