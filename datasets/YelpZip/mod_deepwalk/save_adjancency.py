import networkx as nx
import pickle
import pandas as pd
import numpy as np

f = open('../graph.txt', 'rb')
G = pickle.load(f)
f.close()

i = 0
mapping = {}
for n in G.nodes():
    mapping[i] = n
    i = i + 1

G = nx.relabel_nodes(G, mapping)

A = nx.adjacency_matrix(G)
A = pd.DataFrame(A.todense())
A.to_csv('adj_list', sep = ' ', header=None)


M = pd.DataFrame.from_dict(mapping, orient="index")
M.to_csv('node_label_mapping')
