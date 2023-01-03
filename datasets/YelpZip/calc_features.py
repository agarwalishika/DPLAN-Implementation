import numpy as np
import pandas as pd

'''
returns a set (represented as list) of 'key's
'''
def get_set_of(data, key):
    feat = []
    group = data.groupby(key)
    for g, d in group:
        feat.append(g)
    return feat

'''
returns a dictionary with the key as 'key' and 
the value as the ratio of positive ratings. positive
ratings are ratings with 4-5 stars
'''
def ratio_positive_reviews(data, key):
    feat = {}
    group = data.groupby(key)
    for g,d in group:
        ratings = d['rating'].reset_index(drop=True)
        pos_rates = np.where(ratings >= 4.0)
        feat[g] = pos_rates[0].size/ratings.size
    return feat

'''
returns a dictionary with the key as a user_id and 
the value as the ratio of negative ratings. negative
ratings are ratings with 1-2 stars
'''
def ratio_negative_reviews(data, key):
    feat = {}
    group = data.groupby(key)
    for g,d in group:
        ratings = d['rating'].reset_index(drop=True)
        neg_rates = np.where(ratings <= 2.0)
        feat[g] = neg_rates[0].size/ratings.size
    return feat

'''
returns a dictionary with the key as the review number
and the value as the length of the review
'''
def review_length(data):
    feat = pd.Series(data['reviewContent'])
    feat = feat.str.len()
    return feat.to_dict()
'''
returns a list of edges (represented by tuples)
between 'key' and reviews
'''
def edges_to_review(data, key, key_pref):
    edges = []
    for i,row in data.iterrows():
        from_name = key_pref + str(row[key])
        to_name = "rev_" + str(row['index'])
        edges.append((from_name, to_name))
    return edges




