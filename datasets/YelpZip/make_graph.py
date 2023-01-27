import os, sys
import numpy as np
import pandas as pd
from calc_features import *
import networkx as nx
import pickle

dir_name = "gdn"


def create_graph():
    # create one big dataframe for all the feature data
    
    meta = pd.read_csv(dir_name + "/metadata0", sep='\t', header=None)
    meta.columns = ['user_id', 'prod_id', 'rating', 'label', 'date']

    #review = pd.read_csv(dir_name + "/reviewContent", sep='\t', header=None, quoting=csv.QUOTE_NONE, error_bad_lines=False)
    review = pd.read_csv(dir_name + "/reviewContent0", sep='\t', header=None)
    review.columns = ['user_id', 'prod_id', 'date', 'review']

    for i in range(1,60):
        m = pd.read_csv(dir_name + f'/metadata{i}', sep='\t', header=None)
        m.columns = ['user_id', 'prod_id', 'rating', 'label', 'date']

        r = pd.read_csv(dir_name + f'/reviewContent{i}', sep='\t', header=None)
        r.columns = ['user_id', 'prod_id', 'date', 'review']

        meta = pd.concat([meta, m])
        review = pd.concat([review, r])
    
    meta = pd.merge(meta, review, on=['user_id', 'prod_id', 'date'], how='left')
    meta = meta.reset_index()
    meta = meta.rename(columns={'review':'reviewContent'})
    meta['reviewContent'] = meta['reviewContent'].fillna("")

    # user node features
    user_ids = get_set_of(meta, 'user_id')
    user_pos_reviews = ratio_positive_reviews(meta, 'user_id')
    user_neg_reviews = ratio_negative_reviews(meta, 'user_id')

    # create the user nodes
    user_nodes = []
    for u in user_ids:
        name = "user_" + str(u)
        feat = {}
        feat["user_pos_reviews"] = user_pos_reviews[u]
        feat["user_neg_reviews"] = user_neg_reviews[u]
        user_nodes.append((name, feat))

    # product node features
    prod_ids = get_set_of(meta, 'prod_id')
    prod_pos_reviews = ratio_positive_reviews(meta, 'prod_id')
    prod_neg_reviews = ratio_negative_reviews(meta, 'prod_id')

    # create the product nodes
    prod_nodes = []
    for p in prod_ids:
        name = "prod_" + str(p)
        feat = {}
        feat["prod_pos_reviews"] = prod_pos_reviews[p]
        feat["prod_neg_reviews"] = prod_neg_reviews[p]
        prod_nodes.append((name, feat))


    # review content node features
    review_lens = review_length(meta)

    # create the review nodes
    rev_nodes = []
    for key in review_lens:
        name = "rev_" + str(key)
        rev_nodes.append((name, {'review_len' : review_lens[key]}))

        #store the review key and label for later
        f = open(dir_name + '/review_labels', 'a')
        f.write(name + "," + str(meta.loc[key]['label']) + "\n")
        f.close()

    # create user_id to review edges
    user_to_rev_edges = edges_to_review(meta, 'user_id', "user_")

    # create prod_id to review edges
    prod_to_rev_edges = edges_to_review(meta, 'prod_id', "prod_")


    # create the graph
    G = nx.Graph()

    # add nodes to the graph
    G.add_nodes_from(user_nodes)
    G.add_nodes_from(prod_nodes)
    G.add_nodes_from(rev_nodes)

    # add edges to the graph
    G.add_edges_from(user_to_rev_edges)
    G.add_edges_from(prod_to_rev_edges)

    # draw the graph
    #pos = nx.spring_layout(G, k = 0.35)
    #nx.draw(G, pos, with_labels=True)
    #plt.savefig("graph.png", format="PNG")

    f = open('graph.txt', 'wb')
    pickle.dump(G, f)
    f.close()

if not os.path.isfile('graph.txt'):
    create_graph()
