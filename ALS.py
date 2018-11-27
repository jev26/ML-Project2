import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
from helper import *
import pandas as pd
import time
from itertools import groupby

from helpers import init_MF


#taken from course
def build_index_groups(train):
    """build groups for nnz rows and cols."""
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    sorted_nz_train_byrow = sorted(nz_train, key=lambda x: x[0])
    grouped_nz_train_byrow = groupby(sorted_nz_train_byrow, lambda x: x[0])
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    sorted_nz_train_bycol = sorted(nz_train, key=lambda x: x[1])
    grouped_nz_train_bycol = groupby(sorted_nz_train_bycol, lambda x: x[1])
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices


def update_user_feature(
        train, item_features, lambda_user,
        nonzero_items_per_user, nonzero_user_itemindices):
    """update user feature matrix."""
    user_count = nonzero_items_per_user.shape[0]
    feature_count = item_features.shape[0]
    lambda_I = lambda_user * sp.eye(feature_count)
    updated_user_features = np.zeros((feature_count, user_count))

    for user, items in nonzero_user_itemindices:
        # extract the columns corresponding to the prediction for given item
        M = item_features[:, items]

        # update column row of user features
        V = M @ train[items, user]
        A = M @ M.T + nonzero_items_per_user[user] * lambda_I
        X = np.linalg.solve(A, V)
        updated_user_features[:, user] = np.copy(X.T)
    return updated_user_features


def update_item_feature(
        train, user_features, lambda_item,
        nonzero_users_per_item, nonzero_item_userindices):
    """update item feature matrix."""
    item_count = nonzero_users_per_item.shape[0]
    feature_count = user_features.shape[0]
    lambda_I = lambda_item * sp.eye(feature_count)
    updated_item_features = np.zeros((feature_count, item_count))

    for item, users in nonzero_item_userindices:
        # extract the columns corresponding to the prediction for given user
        M = user_features[:, users]
        V = M @ train[item, users].T
        A = M @ M.T + nonzero_users_per_item[item] * lambda_I
        X = np.linalg.solve(A, V)
        updated_item_features[:, item] = np.copy(X.T)
    return updated_item_features


def ALS(ratings):
    """Alternating least squares (ALS)"""
    #define parameters

    num_features = 20
    lambda_user = 0.1
    lambda_item = 0.7
    stop_criterion = 1e-4
    change = 1
    error_list = [0, 0]

    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(ratings, num_features)

    #group indices by row or column
    nonzero_ratings, nonzero_item_userindices, nonzero_user_itemindices = build_index_groups(ratings)

    #get the number of users per item and the number of items per user
    nonzero_users_per_item_count = [len(users) for items, users in nonzero_item_userindices]
    nonzero_items_per_user_count = [len(items) for users, items in nonzero_user_itemindices]


    #run ALS
    print('Start ALS learning...')

    #stop if the error difference is smaller than the stop criterion
    while(np.abs(error_list[-2]-error_list[-1]) > stop_criterion):

        # update user_features and item_features
        user_features = update_user_feature(ratings, item_features, lambda_user, nonzero_items_per_user_count, nonzero_item_userindices)
        item_features = update_item_feature(ratings, user_features, lambda_item, nonzero_users_per_item_count, nonzero_user_itemindices)

        # compute error (RMSE)
        rmse = compute_error(ratings, user_features, item_features, nonzero_ratings)
        print("RMSE: {}.".format(rmse))
        error_list.append(rmse)


    #remove initial values
    error_list.pop(0)
    error_list.pop(0)

    #return preduction
    return prediction(user_features, item_features), error_list



def ALS_CV(train, test, num_features, lambda_user, lambda_item, stop_criterion):
    """Alternating least squares (ALS)"""
    #define parameters

    error_list = [0, 0]

    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(train, num_features)

    #group indices by row or column
    nonzero_ratings_tr, nonzero_item_userindices_tr, nonzero_user_itemindices_tr = build_index_groups(train)
    nonzero_ratings_te = list(zip(test.nonzero())).reshape((-1,2))

    #get the number of users per item and the number of items per user
    nonzero_users_per_item_count = [len(users) for items, users in nonzero_item_userindices_tr]
    nonzero_items_per_user_count = [len(items) for users, items in nonzero_user_itemindices_tr]


    #run ALS
    print('Start ALS learning...')

    #stop if the error difference is smaller than the stop criterion
    while(np.abs(error_list[-2]-error_list[-1]) > stop_criterion):
        # shuffle the training rating indices
        np.random.shuffle(nonzero_ratings_tr)

        # update user_features and item_features
        user_features = update_user_feature(train, item_features, lambda_user, nonzero_items_per_user_count, nonzero_item_userindices_tr)
        item_features = update_item_feature(train, user_features, lambda_item, nonzero_users_per_item_count, nonzero_user_itemindices_tr)

        # compute error (RMSE)
        rmse = compute_error(train, user_features, item_features, np.array(nonzero_ratings_tr).reshape(-1, 2))
        print("RMSE: {}.".format(rmse))
        error_list.append(rmse)


    #remove initial values
    error_list.pop(0)
    error_list.pop(0)

    #return preduction
    return prediction(user_features, item_features), error_list





