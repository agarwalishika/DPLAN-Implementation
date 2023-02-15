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
returns two dictionaries: one with the key as 'key' and 
the value as the number of positive ratings and the second
with the number of total reviews. positive ratings are 
ratings with 4-5 stars
'''
def process_positive_reviews(data, key):
    feat = {}
    size = {}
    group = data.groupby(key)
    for g,d in group:
        ratings = d['rating'].reset_index(drop=True)
        pos_rates = np.where(ratings >= 4.0)
        feat[g] = pos_rates[0].size
        size[g] = ratings.size
    return feat, size

'''
returns a dictionary with the key as 'key' and 
the value as the ratio of positive ratings. positive
ratings are ratings with 4-5 stars
'''
def ratio_positive_reviews(data, key):
    feat, size = process_positive_reviews(data, key)
    for k in feat:
        feat[k] = feat[k] / size[k]
    return feat

'''
returns a dictionary with the key as 'key' and 
the value as the number of positive ratings. positive
ratings are ratings with 4-5 stars
'''
def num_positive_reviews(data, key):
    feat, _ = process_positive_reviews(data, key)
    return feat

'''
returns two dictionaries: one with the key as 'key' and 
the value as the number of negative ratings and the second
with the number of total reviews. negative
ratings are ratings with 1-2 stars
'''
def process_negative_reviews(data, key):
    feat = {}
    size = {}
    group = data.groupby(key)
    for g,d in group:
        ratings = d['rating'].reset_index(drop=True)
        neg_rates = np.where(ratings <= 2.0)
        feat[g] = neg_rates[0].size
        size[g] = ratings.size
    return feat, size

'''
returns a dictionary with the key as 'key' and 
the value as the ratio of negative ratings. negative
ratings are ratings with 4-5 stars
'''
def ratio_negative_reviews(data, key):
    feat, size = process_negative_reviews(data, key)
    for k in feat:
        feat[k] = feat[k] / size[k]
    return feat

'''
returns a dictionary with the key as 'key' and 
the value as the number of positive ratings. positive
ratings are ratings with 4-5 stars
'''
def num_negative_reviews(data, key):
    feat, _ = process_negative_reviews(data, key)
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
returns a dictionary with the key as the review number
and the value as whether or not it is a singleton review
(whether the corresponding user has only 1 review)
'''
def is_singleton(data):
    feat = {}
    group = data.groupby('user_id')
    for g,d in group:
        if len(d) == 1:
            key = 1
        else:
            key = 0
        
        d['key'] = key
        d = pd.DataFrame(d.key.values, index=d.index).to_dict()[0]
        feat = {**feat, **d}
    return feat

'''
returns a dictionary where the key is the review number
and the value is the percentage of capital letters in
the review
'''
def uppercase_percent(data):
    feat = pd.Series(data['reviewContent'])
    upper = feat.str.count(r'[A-Z]')
    lower = feat.str.count(r'[a-z]')
    feat = upper / lower
    return feat.to_dict()

'''
auxiliary function that is used in all_caps_percentage()
'''
def count_caps(row):
    return sum([word.isupper() for word in row.split(' ')])
    
'''
returns a dictionary where the key is the review number
and the value is the percentage of all caps words in 
the review
'''
def all_caps_percentage(data):
    feat = pd.Series(data['reviewContent'])
    feat = feat.apply(count_caps)
    return feat

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




