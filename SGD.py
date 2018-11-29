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
    nb_features = 40  # K
    lambda_user = 0.01
    lambda_film = 0.01
    errors = [4, 3]
    error = [0]
    stop_criterion = 1e-3
    iter = 0;

    # set seed
    np.random.seed(988)

    # init matrix
    user_features, film_features = init_MF(ratings, nb_features)

    # find the non-zero ratings indices
    nonzero_row, nonzero_col = ratings.nonzero()
    nonzero_ratings = list(zip(nonzero_row, nonzero_col))

    t0 = time.clock()
    print("learn the matrix factorization using SGD...")
    while (errors[-2] - errors[-1]) > stop_criterion:
        iter += 1 ;
        # shuffle the training rating indices
        #np.random.shuffle(nonzero_ratings)

        # decrease step size
        gamma /= 1.2

        for d, n in nonzero_ratings:
            # calculate the error, update of the user_ and item_ features matrices
            error = ratings[d, n] - prediction(user_features[:, n], film_features[:, d])
            # Update latent user feature matrix
            user_features[:, n] += gamma * (error * film_features[:, d] - lambda_user * user_features[:, n])
            film_features[:, d] += gamma * (error * user_features[:, n] - lambda_film * film_features[:, d])

        rmse = compute_error(ratings, user_features, film_features, nonzero_ratings)
        print("iter: {}, RMSE :{}.".format(iter, rmse))
        print("current time to compute this SGD was : {} ".format(time.clock()))
        errors.append(rmse)

    # remove the initializations
    print("Iteration stopped, as iteration criterion {} was reached. RMSE = {}".format(stop_criterion, errors[-1]))
    errors.remove(4)
    errors.remove(3)
    return prediction(user_features, film_features), errors

# Cross-validation algorithms


def matrix_factorization_SGD_CV(train, test, num_features, lambda_user, lambda_film, stop_criterion):
    # define parameters

    errors = [5, 4]
    error = [0]
    gamma = 0.01

    # set seed
    np.random.seed(988)

    # init matrix
    user_features, film_features = init_MF(train, num_features)

    # find the non-zero ratings indices
    nonzero_row, nonzero_col = train.nonzero()
    nonzero_train = list(zip(nonzero_row, nonzero_col))
    nonzero_row, nonzero_col = test.nonzero()
    nonzero_test = list(zip(nonzero_row, nonzero_col))

    print("learn the matrix factorization using SGD...")
    while (errors[-2] - errors[-1]) > stop_criterion:
        # shuffle the training rating indices
        np.random.shuffle(nonzero_train)

        # decrease step size
        gamma = gamma * 1 / 2

        for d, n in nonzero_train:
            # do matrix factorization.
            error = train[d, n] - prediction(user_features[:, n], film_features[:, d])
            user_features[:, n] += gamma * (error * film_features[:, d] - lambda_user * user_features[:, n])  # Update latent user feature matrix
            film_features[:, d] += gamma * (error * user_features[:, n] - lambda_film * film_features[:, d])

        rmse = compute_error(train, user_features, film_features, np.array(nonzero_train).reshape((-1, 2)))

        errors.append(rmse)
    errors.remove(5)
    errors.remove(4)
    rmse = compute_error(test, user_features, film_features, np.array(nonzero_test).reshape((-1, 2)))
    print("RMSE on test data: {}.".format(rmse))

    return prediction(user_features, film_features), rmse

