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

    embeds = Node2Vec(G, dimensions=7999, walk_length=30, num_walks=20, workers=2)
    model = embeds.fit(window=10, min_count=1, batch_words=4)
    model.wv.save_word2vec_format('embeddings')

f = open('../graph.txt', 'rb')
G = pickle.load(f)
f.close()

embeds = pd.read_csv('embeddings', sep=' ', skiprows=[0], header=None)
embeds = embeds.rename(columns={0: "key"})

rev_labels = pd.read_csv(dir_name + "review_labels", header=None)
rev_labels.columns = ["key", "label"]

rev_embeds = pd.DataFrame()
node_attrs = nx.get_node_attributes(G, "review_len")

count = 0
for i, row in embeds.iterrows():
    if 'rev' in row['key']:
        k = row['key']
        r = row
        r['review_len'] = node_attrs[k]
        lbl = rev_labels.query(f'key == \'{k}\'').reset_index(drop=True)['label'][0]
        if lbl == -1:
            r['label'] = 1
        elif lbl == 1:
            r['label'] = 0 
        rev_embeds = rev_embeds.append(r)

rev_embeds = rev_embeds.reset_index(drop=True)

r = int(rev_embeds.shape[0] / 2)

rev_embeds.iloc[0:r].to_csv('final_embeddings.csv', index=False)
rev_embeds.iloc[r:].to_csv('test_final_embeddings.csv', index=False)
