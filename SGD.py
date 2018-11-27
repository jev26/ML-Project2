import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
from helper import *
import pandas as pd
import time



# ALS and SGD

def matrix_factorization_SGD(ratings):
    """matrix factorization by SGD."""
    # define parameters
    gamma = 0.01
    num_features = 40  # K
    lambda_user = 0.01
    lambda_item = 0.01
    errors = [4, 3]
    error = [0]
    stop_criterion = 1e-3
    iter = 0;

    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(ratings, num_features)

    # find the non-zero ratings indices
    nonzero_row, nonzero_col = ratings.nonzero()
    nonzero_ratings = list(zip(nonzero_row, nonzero_col))

    t0 = time.clock()
    print("learn the matrix factorization using SGD...")
    while ((errors[-2] - errors[-1]) > stop_criterion):
        iter += 1 ;
        # shuffle the training rating indices
        #np.random.shuffle(nonzero_ratings)

        # decrease step size
        gamma /= 1.2

        for d, n in nonzero_ratings:
            # calculate the error, update of the user_ and item_ features matrices
            error = ratings[d, n] - prediction(user_features[:, n], item_features[:, d])
            # Update latent user feature matrix
            user_features[:, n] += gamma * (error * item_features[:, d] - lambda_user * user_features[:, n])
            item_features[:, d] += gamma * (error * user_features[:, n] - lambda_item * item_features[:, d])

        rmse = compute_error(ratings, user_features, item_features, nonzero_ratings)
        print("iter: {}, RMSE :{}.".format(iter, rmse))
        print("current time to compute this SGD was : {} ".format(time.clock()))
        errors.append(rmse)

    # remove the initializations
    print("Iteration stopped, as iteration criterion {} was reached. RMSE = {}".format(stop_criterion, errors[-1]))
    errors.remove(4)
    errors.remove(3)
    return prediction(user_features, item_features), errors

# Cross-validation algorithms

def matrix_factorization_SGD_CV(train, test, num_features, lambda_user, lambda_item):
    # define parameters

    errors = [5, 4]
    stop_criterion = 1e-4
    e = [0]
    gamma = 0.01

    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(train, num_features)

    # find the non-zero ratings indices
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    print("learn the matrix factorization using SGD...")
    while ((errors[-2] - errors[-1]) > stop_criterion):
        # shuffle the training rating indices
        np.random.shuffle(nz_train)

        # decrease step size
        gamma = gamma * 1 / 2

        for d, n in nz_train:
            # do matrix factorization.
            e = train[d, n] - prediction(user_features[:, n], item_features[:, d])
            user_features[:, n] += gamma * (e * item_features[:, d] - lambda_user * user_features[:, n])  # Update latent user feature matrix
            item_features[:, d] += gamma * (e * user_features[:, n] - lambda_item * item_features[:, d])

        nz_train2 = np.array(nz_train).reshape((-1, 2))
        rmse = compute_error(train, user_features, item_features, nz_train2)

        errors.append(rmse)

    nz_test2 = np.array(nz_test).reshape((-1, 2))
    rmse = compute_error(test, user_features, item_features, nz_test2)
    print("RMSE on test data: {}.".format(rmse))

    return prediction(user_features, item_features), rmse

