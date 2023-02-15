import numpy as np
import pandas as pd
import os
from node2vec import Node2Vec
import networkx as nx
import pickle


dir_name = ""

if not os.path.isfile('embeddings'):
    f = open('../graph.txt', 'rb')
    G = pickle.load(f)
    f.close()

    embeds = Node2Vec(G, dimensions=1000, walk_length=10, num_walks=10, workers=1)
    model = embeds.fit(window=10, min_count=1, batch_words=4)
    model.wv.save_word2vec_format('embeddings')

f = open('../graph.txt', 'rb')
G = pickle.load(f)
f.close()

embeds = pd.read_csv('embeddings', sep=' ', skiprows=[0], header=None)
embeds = embeds.rename(columns={0: "key"})

rev_embeds = pd.DataFrame()

'''# add node attributes to the embeddings
node_attrs = ['review_len', 'rev_is_single', 'caps_percent', 'upper_percent']
for na in node_attrs:
    attr = nx.get_node_attributes(G, na)
    g = pd.DataFrame.from_dict(attr, orient='index')
    g = g.reset_index()
    g.columns = ['key', na]

    if len(rev_embeds) == 0:
        rev_embeds = pd.merge(embeds, g, on=['key'])
    else:
        rev_embeds = pd.merge(rev_embeds, g, on=['key'])'''

# add the labels to the embeddings to make it a dataset
rev_labels = pd.read_csv(dir_name + "review_labels", header=None)
rev_labels.columns = ["key", "label"]
rev_embeds = pd.merge(embeds, rev_labels, on=['key'])

rev_embeds = rev_embeds.reset_index(drop=True)

# split the dataset and save it to csv files
r = int(rev_embeds.shape[0] / 20)

for i in range(0, 15):
    rev_embeds.iloc[i*r:(i+1)*r].to_csv(f'final_embeddings{i}.csv', index=False)

rev_embeds.iloc[15*r:].to_csv(f'test_final_embeddings.csv', index=False)
