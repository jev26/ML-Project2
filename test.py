import pandas as pd
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt

def load_data_sparse(path_dataset, exploration = True):

    data = pd.read_csv(path_dataset)
    #print(data.head(10))

    data['r'] = data.Id.str.split('_').str.get(0).str[1:]

    data['c'] = data.Id.str.split('_').str.get(1).str[1:]
    print(data.head(10))

    if exploration : data_exploration(data)

    # row indices
    row_ind = np.array(data['r'], dtype=int)
    # column indices
    col_ind = np.array(data['c'], dtype=int)
    # data to be stored in COO sparse matrix
    data = np.array(data['Prediction'], dtype=int)

    # create COO sparse matrix from three arrays
    mat_coo = sparse.coo_matrix((data, (row_ind, col_ind)))
    #print(mat_coo.shape)

    mat_lil = mat_coo.tolil()
    #print(mat_lil.shape)

    return mat_lil

def data_exploration(data):
    # 10'000 users and 1'000 films

    #print(data.shape) # (1176952, 4)

    # Nbr Rate Per User
    NbrRatePerUser = data['r'].value_counts()
    print('Nbr Rate Per User')
    print('min: ' + str(np.min(NbrRatePerUser)))
    print('max: ' + str(np.max(NbrRatePerUser)))
    print('mean: ' + str(np.mean(NbrRatePerUser)))
    print('================')

    plt.figure()
    plt.hist(NbrRatePerUser, bins=1000)
    plt.xlabel("Nbr Rate Per User")
    plt.ylabel("Freqeuncy")
    plt.show()

    # Nbr Rate Per Film
    NbrRatePerFilm = data['c'].value_counts()
    #print(NbrRatePerFilm)
    print('Nbr Rate Per Film')
    print('min: ' + str(np.min(NbrRatePerFilm)))
    print('max: ' + str(np.max(NbrRatePerFilm)))
    print('mean: ' + str(np.mean(NbrRatePerFilm)))
    print('================')

    plt.figure()
    plt.hist(NbrRatePerFilm, bins=500)
    plt.xlabel("Ndr Rate Per Film")
    plt.ylabel("Freqeuncy")
    plt.show()

    # Mean Rate Per Film
    MeanPerFilm = data.groupby('c').mean()

    """plt.figure()
    plt.plot(MeanPerFilm)
    plt.xlabel("Film")
    plt.ylabel("Mean")
    plt.show()"""

    print(MeanPerFilm.shape)
    print(type(MeanPerFilm))

    plt.figure()
    plt.hist(np.transpose(MeanPerFilm),bins=30)#,range=[1,5])#bins='auto',range=[1,5])
    plt.xlabel("Mean Per Film")
    plt.ylabel("Freqeuncy")
    plt.show()

    # Variance in Rating Per Film
    StdPerFilm = data.groupby('c').std()

    """plt.figure()
    plt.plot(StdPerFilm)
    plt.xlabel("Film")
    plt.ylabel("Std")
    plt.show()"""

    plt.figure()
    plt.hist(np.transpose(StdPerFilm), bins=30)
    plt.xlabel("Variance Per Film")
    plt.ylabel("Freqeuncy")
    plt.show()