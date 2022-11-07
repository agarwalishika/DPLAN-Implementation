import pickle
import networkx as nx
import numpy as np
import pandas as pd
import os

dir_name = "../medium_raw/"

f = open('../graph.txt', 'rb')
G = pickle.load(f)
f.close()

node_attrs = nx.get_node_attributes(G, "review_len")

embeds = pd.read_csv('embeddings', sep=' ', header=None, skiprows=[0])
embeds = embeds.rename(columns={0: "num"})

mapping = pd.read_csv('node_label_mapping', header=None, skiprows=[0])
mapping.columns = ["num", "key"]

rev_labels = pd.read_csv(dir_name + "review_labels", header=None)
rev_labels.columns = ["key", "label"]

rev_embeds = pd.DataFrame()

count = 0
for i, row in mapping.iterrows():
    if 'rev' in row['key']:
        k = row['num']
        r = embeds.query(f'num == {k}').reset_index(drop=True)
        k = row['key']
        r['review_len'] = node_attrs[k]
        lbl = rev_labels.query(f'key == \'{k}\'').reset_index(drop=True)['label'][0]
        if lbl == -1:
            r['label'] = 1
        elif lbl == 1:
            r['label'] = 0
        rev_embeds = pd.concat([rev_embeds, r])

rev_embeds = rev_embeds.reset_index(drop=True)

r = int(rev_embeds.shape[0] / 2)
rev_embeds.iloc[0:r].to_csv('final_embeddings.csv', index=False)
rev_embeds.iloc[r:].to_csv('test_final_embeddings.csv', index=False)

