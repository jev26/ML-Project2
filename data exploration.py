import numpy as np
import matplotlib.pyplot as plt
from helper import *
import scipy.sparse as sp
from test import load_data_sparse
from SGD import *

data_name = "data/47b05e70-6076-44e8-96da-2530dc2187de_data_train.csv"

ratings_sparse, ratings_pandas = load_data_sparse(data_name, False)
#rating = load_csv_data(data_name)

print(ratings_sparse.shape)
NbrUsers = ratings_sparse.shape[1]
NbrFilm = ratings_sparse.shape[1]

#print(ratings[0,0])

#sp.csr_matrix.count_nonzero(ratings)

#import matplotlib.pylab as plt
#import scipy.sparse as sps
#A = sps.rand(10000,10000, density=0.00001)
#M = sps.csr_matrix(ratings)
#M = sps.csr_matrix(A)
#plt.spy(M)
#plt.show()

# Nbr rating per film
#print(np.count_nonzero(ratings, axis=0))

#mean_rating_user = np.zeros(rating.shape[1])
#mean_rating_movie = np.zeros(rating.shape[0])
#print(mean_rating_movie.shape)
#print(mean_rating_user.shape)


"""""
mean_rating_user = np.mean(ratings,axis =0)

mean_rating_movie = np.mean(ratings,axis =1)

plt.figure()
plt.plot(mean_rating_movie)
plt.title('mean_rating_movie')
plt.show()
plt.figure()
plt.plot(mean_rating_user.T)
plt.title('mean_rating_user')
plt.show()
"""

#matrix_factorization_SGD(ratings)