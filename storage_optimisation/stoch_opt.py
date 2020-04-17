import numpy as np 
import pandas as ps 
import matplotlib.pyplot as plt 
import networkx as nx
from Diffusion import DiffusionSpot
from itertools import product


def clustering(price,s,N):
    # Algorithm 4.1
    # Attention on utilise des unions : Que faire si on commence à avoir des valeurs qui se répètent
    diff = DiffusionSpot('../Web_Scraping/spot_€_MWh_CEGH_VTP.csv','../Web_Scraping/forward_€_MWh_CZ_VTP.csv' )
    samples = large_sampler(price,N)
    clusters = [{alea} for alea in samples]
    while len(clusters) > s:
        min_glob = np.inf
        for i,j in product(range(len(clusters)),repeat = 2):
            if i == j :
                pass
            else :
                C_i = clusters[i]
                C_j = clusters[j] 
                maxCiCj = max([abs(etha_i-etha_j) for etha_i, etha_j in product(C_i,C_j)])
                if maxCiCj < min_glob:
                    i_min  = i
                    j_min = j
                    min_glob = maxCiCj
        C_1 = clusters.pop(i_min)
        C_2 = clusters.pop(j_min-1)

        C = C_1|C_2
        clusters.append(C)
    return [np.median(list(i)) for i in clusters]

def large_sampler(price,N=10):
    large_simul=[]
    diff = DiffusionSpot('../Web_Scraping/spot_€_MWh_CEGH_VTP.csv','../Web_Scraping/forward_€_MWh_CZ_VTP.csv' )
    for i in range (N):
        large_simul.append(diff.one_iteration(price))
    #plt.hist(large_simul)
    #plt.show()
    return large_simul

def harmon():
    n=1
    S = 0
    while True:
        n+=1
        S+=1/n
        yield S

def index_min(value,Z):
    min = abs(Z[0]-value)
    index = 0
    for i, val in enumerate(Z):
        if abs(value-val) < min:
            min = abs(value-val)
            index = i 
    return index


def sign( value):
    if value >= 0 :
        return 1 
    else:
        return-1

def stochastic_approximation(price, bushiness, epsilon = 10e-8, maxiter = 1000):
    # 4.5
    N = 1000
    diff = DiffusionSpot('../Web_Scraping/spot_€_MWh_CEGH_VTP.csv','../Web_Scraping/forward_€_MWh_CZ_VTP.csv' )
    a_k = harmon()
    norm = np.inf
    Z = clustering(price,bushiness,10)
    for it in range(maxiter):
        if norm < epsilon:
            old_Z = Z
            prediction = diff.one_iteration(price)
            index = index_min(prediction,Z)
            Z[index] = Z[index] - next(a_k)*abs(Z[index]-value)*sign(Z[index]-value)
            norm = np.linalg.norm(Z-old_Z)
        else:
            break
    print(Z)
    samples  = large_sampler(price,100)
    len_samples = len(samples)
    probabilities = [0 for z in range(len(Z))]
    for sample in samples:
        i = index_min(sample,Z)
        probabilities[i] +=1/len_samples
    
    return Z,probabilities


    
def tree_generation(T,bushiness, initial_price):
    G = nx.DiGraph()
    node_name = 'Root'
    open_node = []
    G.add_node(node_name, stage = 0, probability = 1, price = initial_price)
    open_node.append(node_name)
    n = 1
    print('yip')
    print(G.nodes['Root']['price'])
    for stage in range(1,T):
        print('stage',stage)
        new_node = []
        while open_node :
            active_node = open_node.pop()
            print(active_node)
            successors, succ_prob = stochastic_approximation(G.nodes[active_node]['price'], bushiness)
            for succ, prob in zip(successors,succ_prob):
                new_node.append(n)
                print(succ, prob)
                G.add_node(n, stage = stage, probability = succ_prob, price = succ)
                G.add_edge(active_node,n)
                n+=1
        open_node = new_node
    return G


G = tree_generation(4,2,2)
plt.figure(figsize=(20,20))
nx.draw_shell(G, with_labels=True)
plt.show()

# cluster = clustering(1,3,10)
# print(cluster)
