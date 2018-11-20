""" Basic functions to use input and produce output format"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp


def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()


def load_csv_data(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)


def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""

    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

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