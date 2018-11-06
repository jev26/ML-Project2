import numpy as np
import matplotlib.pyplot as plt
from helper import *
import scipy.sparse as sp


data_name = "data/47b05e70-6076-44e8-96da-2530dc2187de_data_train.csv"
rating = load_csv_data(data_name)
print(rating.shape)
x = rating.shape[1]
print(type(x))
mean_rating_user = np.zeros(rating.shape[1])
mean_rating_movie = np.zeros(rating.shape[0])
print(mean_rating_movie.shape)
print(mean_rating_user.shape)
for i in range(len(mean_rating_user)):
    print(i)
    mean_rating_user[i] = np.mean(rating[:,i])
for j in range(len(mean_rating_movie)):
    print(j)
    mean_rating_movie[j] = np.mean(rating[j,:])
plt.figure()
plt.plot(mean_rating_movie)
plt.title('mean_rating_movie')
plt.figure()
plt.plot(mean_rating_user)
plt.title('mean_rating_user')