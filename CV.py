import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from SGD import *
from ALS import ALS_CV

def cross_validation(SGDModel,ratings, k_fold ,nb_features ,lambdas ,min_nb_ratings ,p_test, stop_criterion):


    train, test = split_data_for_CV(ratings, min_nb_ratings, p_test)
    k_indices = build_k_indices(train, k_fold)

    errors = np.zeros((nb_features.size,lambdas.size))
    #errors = np.zeros((len(nb_features), lambdas.size))

    # Cross-validation
    #for nb_feature in nb_features[:]:
    for counter_feature, nb_feature in enumerate(nb_features):
        #for lambda_ in lambdas[:]:
        for counter_lambda, lambda_ in enumerate(lambdas):
            errors_tmp = []
            for k in range(k_fold):
                train_indices = k_indices[(np.arange(len(k_indices)) != k)].ravel()
                if SGDModel :
                    pred, error = matrix_factorization_SGD_CV(train[train_indices], test[k_indices[k]], nb_feature, lambda_, lambda_, stop_criterion)
                    errors_tmp.append(error)
                else:
                    pred, error = ALS_CV(train[train_indices], test[k_indices[k]], nb_feature, lambda_, lambda_, stop_criterion)
                    errors_tmp.append(error)
            print('Mean errors = %s, num_features = %s, lambda_user = %s, lambda_item = %s, std = %s' % (
            np.mean(errors_tmp), nb_feature, lambda_, lambda_, np.std(errors_tmp)))
            errors[counter_feature, counter_lambda] = np.mean(errors_tmp)

    return errors
    # put plot
