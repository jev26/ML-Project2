from helper import *
from surprise_models import *

# Cross-validation algorithm

def matrix_factorization_SGD_CV(trainset, finalpredset, num_features, lambda_user, lambda_film, stop_criterion):
    train = testset_to_sparse_matrix(trainset.build_testset())

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
        print("RMSE: {}.".format(rmse))
        errors.append(rmse)

    errors.remove(5)
    errors.remove(4)

    pred = prediction(user_features, film_features)

    finalpred_usr_idx, finalpred_movies_idx, _ = get_testset_indices(finalpredset)
    return pred[finalpred_usr_idx, finalpred_movies_idx]

def SGD_test_error_calculation(test, user_features, film_features):

    # find the non-zero ratings indices
    nonzero_row, nonzero_col = test.nonzero()
    nonzero_test = list(zip(nonzero_row, nonzero_col))

    rmse = compute_error(test, user_features, film_features, np.array(nonzero_test).reshape((-1, 2)))
    print("RMSE on test data: {}.".format(rmse))

    return rmse
