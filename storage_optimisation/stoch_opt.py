import numpy as np 
import pandas as ps 
import matplotlib.pyplot as plt 
import networkx as nx
def distribution(initial_spot)

def stochastic_approximation(price, bushiness):
    
def tree_generation(T,bushiness, initial_price):
    G = nx.DiGraph()
    node_name = 'Root'
    open_node = []
    G.add_node(node_name, stage = 0, probability = 1, price = initial_price)
    open_node.append(node_name)
    n = 1
    print(G.nodes['Root']['price'])
    for stage in range(1,T):
        new_node = []
        while open_node :
            active_node = open_node.pop()
            print(active_node)
            successors, succ_prob = stochastic_approximation(G.nodes[active_node]['price'], bushiness)
            for succ in successors:
                new_node.append(n)
                G.add_node(node_name, stage = stage, probability = succ_prob, price = succ)
                G.add_edge(active_node,n)
                n+=1
    return G


G = tree_generation(2,2,1)
plt.figure(figsize=(10,10))
nx.draw(G, with_labels=True)

plt.show()
