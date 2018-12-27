""" Basic functions to use input and produce output format"""
import scipy as sp
import itertools # for feature expansion
from scipy import sparse
from data_exploration import *
import pandas as pd

def calculate_rmse(real_labels, predictions):
    """Calculate RMSE."""
    return np.linalg.norm(real_labels - predictions) / np.sqrt(len(real_labels))

def prediction(user_features,item_features):
    """ compute the inner product of the 2 matrices"""
    return item_features.T.dot(user_features)

def compute_error(data, user_features, item_features, nonzero):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    mse = 0
    for row, col in nonzero:
        Wd = item_features[:, row]
        Zn = user_features[:, col]
        mse += (data[row, col] - Zn.T.dot(Wd)) ** 2
    return np.sqrt(1.0 * mse / len(nonzero))

def split_data_for_CV(ratings, min_num_ratings, p_test=0.1):
    """split the ratings to training data and test data.
    Args:
        min_num_ratings:
            all users and items we keep must have at least min_num_ratings per user and per item.
    """
    #number of film for each user, number of user for each film
    num_film_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
    num_users_per_film = np.array((ratings != 0).sum(axis=1).T).flatten()

    # set seed
    np.random.seed(3)

    # select user and item based on the condition.
    valid_users = np.where(num_film_per_user >= min_num_ratings)[0]
    valid_films = np.where(num_users_per_film >= min_num_ratings)[0]
    valid_ratings = ratings[valid_films, :][:, valid_users]

    num_rows, num_cols = valid_ratings.shape
    train = sp.sparse.lil_matrix((num_rows, num_cols))
    test = sp.sparse.lil_matrix((num_rows, num_cols))

    print("the shape of original ratings. (# of row, # of col): {}".format(ratings.shape))
    print("the shape of valid ratings. (# of row, # of col): {}".format((num_rows, num_cols)))

    nonzero_items, nonzero_users = valid_ratings.nonzero()

    for userN in set(nonzero_users):
        rows, columns = valid_ratings[:, userN].nonzero()
        selected = np.random.choice(rows, size=int((p_test) * len(rows)))
        rest = list(set(rows) - set(selected))
        train[rest, userN] = valid_ratings[rest, userN]
        test[selected, userN] = valid_ratings[selected, userN]

    print("Total number of nonzero elements in original data:{v}".format(v=ratings.nnz))
    print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
    print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    return train, test


def init_MF(train, num_features):
    """init the parameter for matrix factorization."""

    #     user_features: shape = num_features, num_user
    #     item_features: shape = num_features, num_item

    num_items, num_users = train.shape

    # initialize the user features to an identity matrix
    #user_features = np.eye(num_features, num_users)  # Z
    user_features = np.random.rand(num_features, num_users)

    # Assignment of the average rating for that movie as the 1rst row, and small random numbers for the remaining entries
    r, c, v = sp.sparse.find(train[:, 1])
    mean = np.mean(v)
    item_features = np.random.rand(num_features, num_items)
    item_features[1, :] = mean

    return user_features, item_features


def build_k_indices(train, k_fold):
    """build k indices for k-fold."""
    np.random.seed(5)
    nb_row = train.shape[0]
    separation = int(nb_row / k_fold)
    indices = np.random.permutation(nb_row)
    print(indices)

    k_indices = [indices[k * separation: (k + 1) * separation] for k in range(k_fold)]

    return np.array(k_indices)

def polynomial_features(X, degree):
    """polynomial feature function that create a new features matrix with all combinations
    of features with degree less than or equal to the degree"""
    #get the number of samples and features from the X matrix
    nb_samples, nb_features = X.shape

    #create an iterator that lets us iterate over all combinations of numbers from 0 to nb_features-1
    combi = itertools.chain.from_iterable(
        itertools.combinations_with_replacement(range(nb_features), i) for i in range(degree + 1))

    #use that iterator to get the total number of features of the output
    nb_output = sum(1 for _ in combi)

    #initiate an empty array for the output
    PF = np.empty([nb_samples, nb_output])

    #instantiate the iterator again
    combi = itertools.chain.from_iterable(
        itertools.combinations_with_replacement(range(nb_features), i) for i in range(degree + 1))

    #create the polynomial features by iterating and multipliying the columns
    for a, b in enumerate(combi):
        PF[:, a] = X[:, b].prod(1)

    return PF

def load_data_sparse(path_dataset, exploration = True):
    """
    read the file corresponding to path_dataset
    if exploration is True, then make histograms
    return a sparse matrix with the ratings
    plus a pandas dataframe vector containing Rating, User Id and Movie Id
    """
    # read the file
    data = pd.read_csv(path_dataset)

    # break the string to obtain both the row and the column index
    data['r'] = data.Id.str.split('_').str.get(0).str[1:]
    data['c'] = data.Id.str.split('_').str.get(1).str[1:]

    if exploration: data_exploration(data)

    # in the file the index start at 1
    # make it start at 0
    data['r'] = data['r'].apply(lambda l: int(l) - 1)
    data['c'] = data['c'].apply(lambda l: int(l) - 1)

    # create
    # row indices
    row_ind = np.array(data['r'], dtype=int)
    # column indices
    col_ind = np.array(data['c'], dtype=int)
    # data to be stored in COO sparse matrix
    ratings = np.array(data['Prediction'], dtype=int)

    NbrMovie = int(data['r'].max()) + 1
    NbrUser = int(data['c'].max()) + 1

    # create COO sparse matrix from three arrays
    mat_coo = sparse.coo_matrix((ratings, (row_ind, col_ind)), shape=(NbrMovie,NbrUser))
    mat_lil = mat_coo.tolil()

    data = data.rename(index=str, columns={"Prediction": "Rating", "r": "User", "c": "Movie"})
    data = data.drop(['Id'], axis = 1)

    return mat_lil, data

def create_submission_from_prediction(prediction, output_name):
    "Create a csv submission file given the prediction and the name of the file"
    def round(x):
        if (x < 1):
            return 1
        elif (x > 5):
            return 5
        else:
            return x

    DATA_TEST_PATH = 'data/sampleSubmission.csv'
    data = pd.read_csv(DATA_TEST_PATH)

    data['Prediction'] = prediction
    data['Prediction'] = data['Prediction'].apply(lambda l: np.rint(round(l)))
    data.to_csv(output_name, sep=",")
