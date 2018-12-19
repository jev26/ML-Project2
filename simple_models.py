import numpy as np
from surprise_models import testset_to_sparse_matrix, get_testset_indices

def global_mean(trainset, finalpredset):

    train = testset_to_sparse_matrix(trainset.build_testset())

    # find the non zero ratings in the train
    nonzero_train = train[train.nonzero()]

    # calculate the global mean
    global_mean_train = nonzero_train.mean()

    pred = np.full(train.shape, global_mean_train)

    finalpred_usr_idx, finalpred_movies_idx, _ = get_testset_indices(finalpredset)
    return pred[finalpred_usr_idx, finalpred_movies_idx]


def user_mean(trainset, finalpredset):
    """use the user means as the prediction."""

    train = testset_to_sparse_matrix(trainset.build_testset())

    num_items, num_users = train.shape
    pred = np.zeros(train.shape)

    for user_index in range(num_users):
        # find the non-zero ratings for each user in the training dataset
        train_ratings = train[:, user_index]
        nonzeros_train_ratings = train_ratings[train_ratings.nonzero()]

        # calculate the mean if the number of elements is not 0
        if nonzeros_train_ratings.shape[0] != 0:
            user_train_mean = nonzeros_train_ratings.mean()
            pred[:, user_index] = user_train_mean

    finalpred_usr_idx, finalpred_movies_idx, _ = get_testset_indices(finalpredset)
    return pred[finalpred_usr_idx, finalpred_movies_idx]


def item_mean(trainset, finalpredset):
    """baseline method: use item means as the prediction."""

    train = testset_to_sparse_matrix(trainset.build_testset())

    num_items, num_users = train.shape
    pred = np.zeros(train.shape)

    for item_index in range(num_items):
        # find the non-zero ratings for each item in the training dataset
        train_ratings = train[item_index, :]
        nonzeros_train_ratings = train_ratings[train_ratings.nonzero()]

        # calculate the mean if the number of elements is not 0
        if nonzeros_train_ratings.shape[0] != 0:
            item_train_mean = nonzeros_train_ratings.mean()
            pred[item_index, :] = item_train_mean

    finalpred_usr_idx, finalpred_movies_idx, _ = get_testset_indices(finalpredset)
    return pred[finalpred_usr_idx, finalpred_movies_idx]


def global_median(trainset, finalpredset):

    train = testset_to_sparse_matrix(trainset.build_testset())

    # find the non zero ratings in the train
    nonzero_train = train[train.nonzero()]

    # calculate the global mean
    global_median_train = np.median(nonzero_train.toarray())

    pred = np.full(train.shape, global_median_train)

    finalpred_usr_idx, finalpred_movies_idx, _ = get_testset_indices(finalpredset)
    return pred[finalpred_usr_idx, finalpred_movies_idx]


def user_median(trainset, finalpredset):
    """use the user means as the prediction."""

    train = testset_to_sparse_matrix(trainset.build_testset())

    num_items, num_users = train.shape
    pred = np.zeros(train.shape)

    for user_index in range(num_users):
        # find the non-zero ratings for each user in the training dataset
        train_ratings = train[:, user_index]
        nonzeros_train_ratings = train_ratings[train_ratings.nonzero()]

        # calculate the mean if the number of elements is not 0
        if nonzeros_train_ratings.shape[0] != 0:
            user_train_median = np.median(nonzeros_train_ratings.toarray())
            pred[:, user_index] = user_train_median

    finalpred_usr_idx, finalpred_movies_idx, _ = get_testset_indices(finalpredset)
    return pred[finalpred_usr_idx, finalpred_movies_idx]


def item_median(trainset, finalpredset):
    """baseline method: use item means as the prediction."""

    train = testset_to_sparse_matrix(trainset.build_testset())

    num_items, num_users = train.shape
    pred = np.zeros(train.shape)

    for item_index in range(num_items):
        # find the non-zero ratings for each item in the training dataset
        train_ratings = train[item_index, :]
        nonzeros_train_ratings = train_ratings[train_ratings.nonzero()]

        # calculate the mean if the number of elements is not 0
        if nonzeros_train_ratings.shape[0] != 0:
            item_train_median = np.median(nonzeros_train_ratings.toarray())
            pred[item_index, :] = item_train_median

    finalpred_usr_idx, finalpred_movies_idx, _ = get_testset_indices(finalpredset)
    return pred[finalpred_usr_idx, finalpred_movies_idx]