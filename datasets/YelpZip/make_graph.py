import os, sys
import numpy as np
import pandas as pd
from calc_features import *
import networkx as nx
import pickle

dir_name = "chopped"


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
    u_ratio_pos_rev = ratio_positive_reviews(meta, 'user_id')
    u_num_pos_rev = num_positive_reviews(meta, 'user_id')
    u_ratio_neg_rev = ratio_negative_reviews(meta, 'user_id')
    u_num_neg_rev = num_negative_reviews(meta, 'user_id')

    # create the user nodes
    user_nodes = []
    for u in user_ids:
        name = "user_" + str(u)
        feat = {}
        feat["u_ratio_pos_rev"] = u_ratio_pos_rev[u]
        feat["u_num_pos_rev"] = u_num_pos_rev[u]
        feat["u_ratio_neg_rev"] = u_ratio_neg_rev[u]
        feat["u_num_neg_rev"] = u_num_neg_rev[u]
        user_nodes.append((name, feat))

    # product node features
    prod_ids = get_set_of(meta, 'prod_id')
    p_ratio_pos_rev = ratio_positive_reviews(meta, 'prod_id')
    p_num_pos_rev = num_positive_reviews(meta, 'prod_id')
    p_ratio_neg_rev = ratio_negative_reviews(meta, 'prod_id')
    p_num_neg_rev = num_negative_reviews(meta, 'prod_id')

    # create the product nodes
    prod_nodes = []
    for p in prod_ids:
        name = "prod_" + str(p)
        feat = {}
        feat["p_ratio_pos_rev"] = p_ratio_pos_rev[p]
        feat["p_num_pos_rev"] = p_num_pos_rev[p]
        feat["p_ratio_neg_rev"] = p_ratio_neg_rev[p]
        feat["p_num_neg_rev"] = p_num_neg_rev[p]
        prod_nodes.append((name, feat))


    # review content node features
    review_lens = review_length(meta)
    rev_is_single = is_singleton(meta)
    caps_percent = all_caps_percentage(meta)
    upper_percent = uppercase_percent(meta)

    # create the review nodes
    rev_nodes = []
    for key in review_lens:
        name = "rev_" + str(key)
        rev_nodes.append((name, {'review_len' : review_lens[key]}))
        rev_nodes.append((name, {'rev_is_single' : rev_is_single[key]}))
        rev_nodes.append((name, {'caps_percent' : caps_percent[key]}))
        rev_nodes.append((name, {'upper_percent' : caps_percent[key]}))

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
