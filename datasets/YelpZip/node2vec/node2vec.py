import numpy as np
import pandas as pd
import os
from node2vec import Node2Vec
import networkx 
import pickle


dir_name = "medium_raw"

if not os.path.isfile('embeddings'):
    f = open('graph.txt', 'rb')
    G = pickle.load(f)
    f.close()

    embeds = Node2Vec(G, dimension=64, walk_length=30, num_walks=200, workers=4)
    model = embeds.fit(window=10, mind_count=1, batch_words=4)
    model.wv.save_word2vec_format('embeddings')

embeds = pd.read_csv('embeddings', sep=' ', skiprows=[0], header=None)

rev_embeds = embeds.loc[embeds[0].str.contains("rev")].reset_index(drop=True)
rev_embeds = rev_embeds.rename(columns = {0: 'key'})

rev_labels = pd.read_csv(dir_name + '/review_labels', header=None)
rev_labels.columns = ['key', 'label']

rev_embeds['label'] = -1
for i in range(rev_embeds.shape[0]):
    k = rev_embeds.at[i, 'key']
    lbl = rev_labels.query(f'key == \'{k}\'').reset_index(drop=True)['label'][0]
    if lbl == -1:
        rev_embeds.at[i, 'label'] = 1
    elif lbl == 1:
        rev_embeds.at[i, 'label'] = 0

r = int(rev_embeds.shape[0] / 2)

rev_embeds.iloc[0:r].to_csv(dir_name + '/final_embeddings.csv', index=False)
rev_embeds.iloc[r:].to_csv(dir_name + 'test_final_embeddings.csv', index=False)
