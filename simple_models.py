import numpy as np

def global_mean(train, test):
    """baseline method: use the global mean."""
    # find the non zero ratings in the train
    nonzero_train = train[train.nonzero()]

    # calculate the global mean
    global_mean_train = nonzero_train.mean()

    predictions = np.copy(train)
    predictions[predictions != 0.0] = global_mean_train

    return predictions




def user_mean(train, test):
    """baseline method: use the user means as the prediction."""
    num_items, num_users = train.shape
    predictions = np.zeros(train.shape)

    for user_index in range(num_users):
        # find the non-zero ratings for each user in the training dataset
        train_ratings = train[:, user_index]
        nonzeros_train_ratings = train_ratings[train_ratings.nonzero()]

        # calculate the mean if the number of elements is not 0
        if nonzeros_train_ratings.shape[0] != 0:
            user_train_mean = nonzeros_train_ratings.mean()

        for item_index in range(num_items):
            rating = train[item_index, user_index]
            if rating:
                predictions[item_index, user_index] = rating
            else:
                predictions[item_index, user_index] = user_train_mean

    return predictions

def baseline_item_mean(train, test):
    """baseline method: use item means as the prediction."""
    num_items, num_users = train.shape
    predictions = np.zeros(train.shape)

    for item_index in range(num_items):
        # find the non-zero ratings for each item in the training dataset
        train_ratings = train[item_index, :]
        nonzeros_train_ratings = train_ratings[train_ratings.nonzero()]

        # calculate the mean if the number of elements is not 0
        if nonzeros_train_ratings.shape[0] != 0:
            item_train_mean = nonzeros_train_ratings.mean()

        for user_index in range(num_users):
            rating = train[item_index, user_index]
            if rating:
                predictions[item_index, user_index] = rating
            else:
                predictions[item_index, user_index] = item_train_mean

    return predictions