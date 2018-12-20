import surprise as spr
import scipy.sparse as sp
import numpy as np

## Helper functions to use surprise library

def pandas_to_surprise(data, pred=False):
    """Function that converts pandas dataframe into surprise data
    or directly the testset if the submission sample is given and pred set to True"""
    reader = spr.Reader(rating_scale=(1, 5))
    data_spr = spr.Dataset.load_from_df(data[['User', 'Movie', 'Rating']], reader)
    if pred:
        data_spr = data_spr.build_full_trainset().build_testset()
        data_spr = sorted(data_spr, key=lambda x: (x[1], x[0]))
    return data_spr

def get_testset_indices(test_spr):
    """Returns the users, and movies indices and their label of a testset"""
    users_idx, movies_idx, labels = list(zip(*test_spr))
    return users_idx, movies_idx, labels

def testset_to_sparse_matrix(test_spr):
    """Convert a testset to a scipy sparse matrix"""
    matrix = sp.lil_matrix((10000, 1000))
    users_idx, movies_idx, labels = get_testset_indices(test_spr)
    matrix[users_idx, movies_idx] = labels
    return matrix

def spr_estimate_to_vect(predict):
    "Convert the estimations of a surprise prediction to a vector"
    vect = np.zeros(len(predict))
    for i, pred in enumerate(predict):
        vect[i] = pred.est
    return vect

## Function running the suprise's models

def surprise_SVD(trainset, finalset):
    "SVD model"
    algo = spr.SVD(n_factors=40, n_epochs=20, lr_all=0.001)

    algo.fit(trainset)
    predictions_final = algo.test(finalset)

    return spr_estimate_to_vect(predictions_final)

def surprise_basicKNN(trainset, finalset):
    "Basic K Nearest Neighbours model"
    algo = spr.KNNBasic()

    algo.fit(trainset)
    predictions_final = algo.test(finalset)

    return spr_estimate_to_vect(predictions_final)

def surprise_baselineKNN(trainset, finalset):
    "Item based baseline K-Nearest neighbours model"
    options = {'name': 'pearson_baseline',
               'user_based': False}

    algo = spr.KNNBaseline(k=40, sim_options=options)

    algo.fit(trainset)
    predictions_final = algo.test(finalset)

    return spr_estimate_to_vect(predictions_final)

def surprise_slopeOne(trainset, finalset):
    "Slope One model"
    algo = spr.SlopeOne()

    algo.fit(trainset)
    predictions_final = algo.test(finalset)

    return spr_estimate_to_vect(predictions_final)

def surprise_SVDpp(trainset, finalset):
    "SVD++ model"
    algo = spr.SVDpp(n_factors=40, n_epochs=20, lr_all=0.001)

    algo.fit(trainset)
    predictions_final = algo.test(finalset)

    return spr_estimate_to_vect(predictions_final)

def surprise_baseline(trainset, finalset):
    "baseline model"
    
    algo = spr.BaselineOnly()

    algo.fit(trainset)
    predictions_final = algo.test(finalset)

    return spr_estimate_to_vect(predictions_final)
