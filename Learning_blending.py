import numpy as np
import matplotlib.pyplot as plt
from helper import *
import scipy.sparse as sp
from test import load_data_sparse
from SGD import *
from ALS import *
from Visualization import cv_result
from sklearn.linear_model import LinearRegression
from crossval import *

data_name = "data/47b05e70-6076-44e8-96da-2530dc2187de_data_train.csv"

ratings, pandas = load_data_sparse(data_name, False)

print(pandas.shape) # (1176952, 3)

stop_criterion = 1e-2
prediction = []

## Learning
# in Learning_methods.py ==> aim to find the best parameter for each model
# then the parameters are specified here to learn the best model for the final prediction

## Launch Final Models
# SGD
nb_feature = 20
lambda_ = 0.004
pred, _, _ = matrix_factorization_SGD_CV(ratings, nb_feature, lambda_, lambda_, stop_criterion)

#print(pred.shape) # (10000, 1000)
#print(type(pred)) # <class 'numpy.ndarray'>

# keep only prediction for tests
nonzero_row, nonzero_col = ratings.nonzero()
nonzero_train = list(zip(nonzero_row, nonzero_col))

pred_tmp = []
for i in nonzero_train:
    pred_tmp.append(pred.item(i))

print(len(pred_tmp)) # (1176952)
print(type(pred_tmp))

prediction = np.asarray(pred_tmp)
print(type(prediction))
print(prediction.shape)

# ALS
nb_feature = 8
lambda_ = 0.081
pred, _, _ = ALS_CV(ratings, nb_feature, lambda_, lambda_, stop_criterion)

pred_tmp = []
for i in nonzero_train:
    pred_tmp.append(pred.item(i))

print(len(pred_tmp))  # (1176952)

prediction = np.vstack((prediction,pred_tmp))

print(type(prediction))
print(prediction.shape)

## Blending function
# test without CV
print("learn blending wieghts")

y = pandas['Rating']
X = np.transpose(prediction)

clf = LinearRegression()
reg = clf.fit(X, y)

print('Weights of the different models:', clf.coef_)

""""# faire aussi un learning process pour avoir les poids
seed = 1
k_fold = 4

# split data in k fold
k_indices = build_k_indices(y, k_fold, seed)

# vector with true rating
y = pandas['Ratings']

for k in range(k_fold):
    loss_tr, loss_te, score = cross_validation(y, prediction, k_indices, k)

    rmse_te_tmp.append(loss_te)
    rmse_tr_tmp.append(loss_tr)
    accuracy_tmp.append(score)

#clf = LinearRegression()
#clf.fit(X.T, y)

# two types of recommender systems
# --> collaborative filtering"""

