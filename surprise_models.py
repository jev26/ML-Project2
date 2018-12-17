import surprise as spr
import scipy.sparse as sp
import numpy as np


def pandas_to_surprise(data, pred=False):
    reader = spr.Reader(rating_scale=(1, 5))
    data_spr = spr.Dataset.load_from_df(data[['User', 'Movie', 'Rating']], reader)
    if pred:
        data_spr = data_spr.build_full_trainset().build_testset()
        data_spr = np.sorted(data_spr, key=lambda x: (x[1], x[0]))
    return data_spr

def get_testset_indices(test_spr):
    users_idx, movies_idx, labels = list(zip(*test_spr))
    return users_idx, movies_idx, labels

def testset_to_sparse_matrix(test_spr, pred=False):
    matrix = sp.lil_matrix((10000, 1000))
    users_idx, movies_idx, labels = get_testset_indices(test_spr)
    matrix[users_idx, movies_idx] = labels
    return matrix


def spr_estimate_to_vect(predict):
    vect = np.zeros(len(predict))
    for i, pred in enumerate(predict):
        vect[i] = pred.est

    return vect


def surprise_SVD(trainset, testset, finalset):
    algo = spr.SVD(n_factors=80, n_epochs=40, lr_bu=0.01, lr_bi=0.01, lr_pu=0.1, lr_qi=0.1, reg_bu=0.05, reg_bi=0.05, reg_pu=0.09, reg_qi=0.1)

    algo.fit(trainset)
    predictions_test = algo.test(testset)
    predictions_final = algo.test(finalset)
    # Then compute RMSE
    rmse = spr.accuracy.rmse(predictions_test)

    return spr_estimate_to_vect(predictions_test), spr_estimate_to_vect(predictions_final), rmse


def surprise_basicKNN(trainset, testset, finalset):
    algo = spr.KNNBasic()

    algo.fit(trainset)
    predictions_test = algo.test(testset)
    predictions_final = algo.test(finalset)
    # Then compute RMSE
    rmse = spr.accuracy.rmse(predictions_test)

    return spr_estimate_to_vect(predictions_test), spr_estimate_to_vect(predictions_final), rmse


def surprise_slopeOne(trainset, testset, finalset):
    algo = spr.SlopeOne()

    algo.fit(trainset)
    predictions_test = algo.test(testset)
    predictions_final = algo.test(finalset)
    # Then compute RMSE
    rmse = spr.accuracy.rmse(predictions_test)

    return spr_estimate_to_vect(predictions_test), spr_estimate_to_vect(predictions_final), rmse





