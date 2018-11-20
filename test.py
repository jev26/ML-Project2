import pandas as pd
from scipy import sparse
import numpy as np

def load_data_sparse(path_dataset):

    data = pd.read_csv(path_dataset)
    #print(data.head(10))

    data['r'] = data.Id.str.split('_').str.get(0).str[1:]

    data['c'] = data.Id.str.split('_').str.get(1).str[1:]
    print(data.head(10))

    # row indices
    row_ind = np.array(data['r'], dtype=int)
    # column indices
    col_ind = np.array(data['c'], dtype=int)
    # data to be stored in COO sparse matrix
    data = np.array(data['Prediction'], dtype=int)

    # create COO sparse matrix from three arrays
    mat_coo = sparse.coo_matrix((data, (row_ind, col_ind)))
    print(mat_coo.shape)

    mat_lil = mat_coo.tolil()
    print(mat_lil.shape)

    return mat_lil
