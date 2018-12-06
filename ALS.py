from helper import *
import time
from itertools import groupby


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
        nnz_items_per_user, nz_user_itemindices):
    """update user feature matrix."""

    # update and return user feature.
    [num_features, num_items] = item_features.shape
    [num_items, num_users] = train.shape
    user_features = np.eye(num_features, num_users)

    # The update is based on the derived ALS formula Z = (WWT + λIK)-1WX, the regularized term λIK is also adjusted based on the number of ratings of each user.
    for user, item in nz_user_itemindices:
        W = item_features[:, item]
        X = train[item, user]
        I = np.eye(num_features, num_features)
        M = W @ W.T + nnz_items_per_user[user] * lambda_user * I
        update = np.linalg.solve(M, W @ X)
        update = update.reshape(num_features)
        user_features[:, user] = update

    return user_features


def update_item_feature(
        train, user_features, lambda_item,
        nnz_users_per_item, nz_item_userindices):
    """update item feature matrix."""
    # update and return item feature.
    [num_features, num_users] = user_features.shape
    [num_items, num_users] = train.shape
    item_features = np.eye(num_features, num_items)

    # The update is based on the derived ALS formula W = (ZZT + λIK)-1ZXT, the regularized term λIK is also adjusted based on the number of ratings of each item.
    for item, user in nz_item_userindices:
        Z = user_features[:, user]
        X = train[item, user]
        I = np.eye(num_features, num_features)
        M = Z @ Z.T + nnz_users_per_item[item] * lambda_item * I
        update = np.linalg.solve(M, Z @ X.T)
        update = update.reshape(num_features)
        item_features[:, item] = update

    return item_features


def ALS(ratings):
    """Alternating Least Squares (ALS) algorithm."""
    # define parameters
    num_features = 40  # K in the lecture notes
    lambda_user = 0.1
    lambda_film = 0.1
    stop_criterion = 1e-4
    errors = [5, 4]  # record the rmse for each step
    iter = 0

    # set seed
    np.random.seed(988)

    # init ALS
    user_features, item_features = init_MF(ratings, num_features)

    nz_ratings, nz_item_userindices, nz_user_itemindices = build_index_groups(ratings)
    nnz_users_per_item = [len(array) for user, array in nz_item_userindices]
    nnz_items_per_user = [len(array) for user, array in nz_user_itemindices]
    nz_ratings2 = np.array(nz_ratings).reshape((-1, 2))

    # start of the ALS-WR algorithm.
    print("learn the matrix factorization using ALS...")
    while ((errors[-2] - errors[-1]) > stop_criterion):
        iter += 1

        user_features = update_user_feature(ratings, item_features, lambda_user, nnz_items_per_user,
                                            nz_user_itemindices)
        item_features = update_item_feature(ratings, user_features, lambda_film, nnz_users_per_item,
                                            nz_item_userindices)

        # RMSE
        rmse = compute_error(ratings, user_features, item_features, nz_ratings2)
        print("RMSE: {}.".format(rmse))

        errors.append(rmse)
    print("Iteration stopped, as iteration criterion {} was reached. RMSE = {}".format(stop_criterion, errors[-1]))
    errors.remove(5)
    errors.remove(4)
    return prediction(user_features, item_features), errors



def ALS_CV(train, test, num_features, lambda_user, lambda_film, stop_criterion):
    """Alternating Least Squares (ALS) algorithm."""
    # define parameters
    errors = [5, 4]  # record the rmse for each step
    iter = 0

    # set seed
    np.random.seed(988)

    # init ALS
    user_features, item_features = init_MF(train, num_features)

    nz_ratings, nz_item_userindices, nz_user_itemindices = build_index_groups(train)
    nnz_users_per_item = [len(array) for user, array in nz_item_userindices]
    nnz_items_per_user = [len(array) for user, array in nz_user_itemindices]
    nz_ratings2 = np.array(nz_ratings).reshape((-1, 2))

    nz_ratings_te, nz_item_userindices_te, nz_user_itemindices_te = build_index_groups(test)

    # start of the ALS-WR algorithm.
    print("learn the matrix factorization using ALS...")
    while ((errors[-2] - errors[-1]) > stop_criterion):
        iter += 1

        user_features = update_user_feature(train, item_features, lambda_user, nnz_items_per_user,
                                            nz_user_itemindices)
        item_features = update_item_feature(train, user_features, lambda_film, nnz_users_per_item,
                                            nz_item_userindices)

        # RMSE
        rmse = compute_error(train, user_features, item_features, nz_ratings2)
        #print("RMSE: {}.".format(rmse))

        errors.append(rmse)
    #print("Iteration stopped, as iteration criterion {} was reached. RMSE = {}".format(stop_criterion, errors[-1]))
    errors.remove(5)
    errors.remove(4)

    rmse = compute_error(test, user_features, item_features, np.array(nz_ratings_te).reshape((-1, 2)))
    print("RMSE on test data: {}.".format(rmse))
    return prediction(user_features, item_features), rmse





