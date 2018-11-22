def split_data(ratings, num_items_per_user, num_users_per_item,min_num_ratings, p_test=0.1):

   # WORK IN PROGRESS
    """split the ratings to training data and test data.
    Args:
        min_num_ratings:
            all users and items we keep must have at least min_num_ratings per user and per item.
    """
    # set seed
    np.random.seed(988)

    # select user and item based on the condition.
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    valid_ratings = ratings[valid_items, :][:, valid_users]

    # ***************************************************
    # init
    num_rows, num_cols = valid_ratings.shape
    train = sp.lil_matrix((num_rows, num_cols))
    test = sp.lil_matrix((num_rows, num_cols))

    print("the shape of original ratings. (# of row, # of col): {}".format(
        ratings.shape))
    print("the shape of valid ratings. (# of row, # of col): {}".format(
        (num_rows, num_cols)))

    nonzero_items, nonzero_users = valid_ratings.nonzero()

    for userN in set(nonzero_users):
        rows, columns = valid_ratings[:, userN].nonzero()
        selected = np.random.choice(rows, size=int((p_test) * len(rows)))
        rest = list(set(rows) - set(selected))
        train[rest, userN] = valid_ratings[rest, userN]
        test[selected, userN] = valid_ratings[selected, userN]
    # split the data and return train and test data. TODO
    # NOTE: we only consider users and movies that have more
    # than 10 ratings
    # ***************************************************

    print("Total number of nonzero elements in origial data:{v}".format(v=ratings.nnz))
    print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
    print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    return valid_ratings, train, test