""" Basic functions to use input and produce output format"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import csv


def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()


def load_csv_data(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)

def deal_line(line):
    pos, rating = line.split(',')
    row, col = pos.split("_")
    row = row.replace("r", "")
    col = col.replace("c", "")
    return int(row), int(col), float(rating)

def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    print("number of items: {}, number of users: {}".format(max_row, max_col))

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings
"""
def asfptype(self):
    

    fp_types = ['f', 'd', 'F', 'D']

    if self.dtype.char in fp_types:
        return self
    else:
        for fp_type in fp_types:
            if self.dtype <= np.dtype(fp_type):
                return self.astype(fp_type)

        raise TypeError('cannot upcast [%s] to a floating '
                        'point format' % self.dtype.name)
"""
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

def create_submission_from_prediction(prediction, output_name):

    def prediction_transformed(prediction, ids_process):
        """ return the prediction transformed for the submission """
        y = []
        for i in range(len(ids_process)):
            row = ids_process[i][0]
            col = ids_process[i][1]
            y.append(prediction[row - 1, col - 1])
        return y

    def create_csv_submission(ids, y_pred, name):
        """
        Creates an output file in csv format for submission to kaggle
        Arguments: ids (event ids associated with each prediction)
                   y_pred (predicted class labels)
                   name (string name of .csv output file to be created)
        """
        with open(name, 'w') as csvfile:
            fieldnames = ['Id', 'Prediction']
            writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
            writer.writeheader()
            for r1, r2 in zip(ids, y_pred):
                writer.writerow({'Id': str(r1), 'Prediction': float(r2)})

    def round(x):
        if (x < 1):
            return 1
        elif (x > 5):
            return 5
        else:
            return x

    def transform(ids_txt):
        """ split the text index"""

        def deal_line(line):
            pos, rating = line.split(',')
            return str(pos)

        ids = [deal_line(line) for line in ids_txt]
        return ids

    prediction = np.vectorize(round)(prediction)

    DATA_TEST_PATH = 'data/sampleSubmission.csv'

    ids_txt = read_txt(DATA_TEST_PATH)[1:]
    ids_process = [deal_line(line) for line in ids_txt]

    # prediction under the right format
    y = prediction_transformed(prediction, ids_process)

    y = np.rint(y)

    ids = transform(ids_txt)
    OUTPUT_PATH = output_name
    create_csv_submission(ids, y, OUTPUT_PATH)