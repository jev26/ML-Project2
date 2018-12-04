import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from SGD import *

def cross_validation(SGDModel,ratings, k_fold ,nb_features ,lambdas ,min_nb_ratings ,p_test, stop_criterion):


    train, test = split_data_for_CV(ratings, min_nb_ratings, p_test)
    k_indices = build_k_indices(train, k_fold)

    # Cross-validation
    for nb_feature in nb_features[:]:
        for lambda_ in lambdas[:]:
            errors = []
            for k in range(k_fold):
                train_indices = k_indices[(np.arange(len(k_indices)) != k)].ravel();
                if SGDModel :
                    pred, error = matrix_factorization_SGD_CV(train[train_indices], test[k_indices[k]], nb_feature, lambda_, lambda_, stop_criterion);
                    errors.append(error);
                else:
                    break
            print('Mean errors = %s, num_features = %s, lambda_user = %s, lambda_item = %s, std = %s' % (
            np.mean(errors), nb_feature, lambda_, lambda_, np.std(errors)))

    # put plot