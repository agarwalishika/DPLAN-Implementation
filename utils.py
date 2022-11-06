from tensorflow.keras.models import Model
import os
import numpy as np
import tensorflow.keras.backend as K
import argparse
import torch
from time import perf_counter
import networkx as nx
import random
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve
import scipy.sparse as sp


def penulti_output(x: np.ndarray, DQN: Model):
    inp = DQN.input
    penulti_func = K.function([inp], [DQN.layers[-2].output])
    latent_x = penulti_func(x)[0]

    return latent_x

def writeResults(name,rocs,prs,train_times, test_times, file_path):
    roc_mean=np.mean(rocs)
    roc_std=np.std(rocs)
    pr_mean=np.mean(prs)
    pr_std=np.std(prs)
    train_mean=np.mean(train_times)
    train_std=np.std(train_times)
    test_mean=np.mean(test_times)
    test_std=np.std(test_times)

    header=True
    if not os.path.exists(file_path):
        header=False

    with open(file_path,'a') as f:
        if not header:
            f.write("{}, {}, {}, {}, {}\n".format("Name",
                                        "AUC-ROC(mean/std)",
                                        "AUC-PR(mean/std)",
                                        "Train time/s",
                                        "Test time/s"))

        f.write("{}, {}/{}, {}/{}, {}/{}, {}/{}\n".format(name,
                                                          roc_mean, roc_std,
                                                          pr_mean, pr_std,
                                                          train_mean, train_std,
                                                          test_mean, test_std))


def aucPerformance(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    roc_auc = roc_auc_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auc_pr = auc(recall, precision)
    ap = average_precision_score(y_true, y_pred)
    return roc_auc, auc_pr, ap


def sgc_precompute(features, adj, degree):
    # compute S^K
    for i in range(degree):
        features = torch.spmm(adj, features)
    return features


def normalize_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def normalize_feature(feature):
    # Row-wise normalization of sparse feature matrix
    rowsum = np.array(feature.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(feature)
    return mx

