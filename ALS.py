import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
from helper import *
import pandas as pd
import time
from itertools import groupby

from SGD import init_MF


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
    grouped_nz_train_bycol = groupby(sorted_nz_train_byrow, lambda x: x[1])
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices


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

    # find the non-zero ratings indices
    nonzero_row, nonzero_col = ratings.nonzero()
    nonzero_ratings = list(zip(nonzero_row, nonzero_col))

    nonzero_ratings, nonzero_item_userindices, nonzero_user_itemindices = build_index_groups(ratings)
    nonzero_users_per_item = [len(array) for user, array in nonzero_item_userindices]
    nonzero_items_per_user = [len(array) for user, array in nonzero_user_itemindices]


    #run ALS
    while(np.abs(error_list[-2]-error_list[-1]) > stop_criterion):


        # update user_features and item_features

        # TEST RMSE

        pass

    #return preduction
    return 0



