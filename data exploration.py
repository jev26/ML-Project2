import numpy as np
import matplotlib.pyplot as plt
from helper import *
import scipy.sparse as sp
from test import load_data_sparse

data_name = "data/47b05e70-6076-44e8-96da-2530dc2187de_data_train.csv"

rating = load_data_sparse(data_name)
#rating = load_csv_data(data_name)

print(rating.shape)
x = rating.shape[1]
print(type(x))
#mean_rating_user = np.zeros(rating.shape[1])
#mean_rating_movie = np.zeros(rating.shape[0])
#print(mean_rating_movie.shape)
#print(mean_rating_user.shape)

mean_rating_user = np.mean(rating,axis =0)

mean_rating_movie = np.mean(rating,axis =1)

plt.figure()
plt.plot(mean_rating_movie)
plt.title('mean_rating_movie')
plt.show()
plt.figure()
plt.plot(mean_rating_user.T)
plt.title('mean_rating_user')
plt.show()