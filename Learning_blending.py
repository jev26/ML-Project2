import numpy as np
import matplotlib.pyplot as plt
from helper import *
import scipy.sparse as sp
from test import load_data_sparse
from SGD import *
from ALS import *
from Visualization import cv_result
#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from crossval import *
from simple_models import *

data_name = "data/47b05e70-6076-44e8-96da-2530dc2187de_data_train.csv"

ratings, pandas = load_data_sparse(data_name, False)

print(pandas.shape) # (1176952, 3)

stop_criterion = 1e-2
pred_lst = []

## Learning
# in Learning_methods.py ==> aim to find the best parameter for each model
# then the parameters are specified here to learn the best model for the final prediction

## Launch Final Models

# SGD
nb_feature = 20
lambda_ = 0.004
pred_SGD, _, _ = matrix_factorization_SGD_CV(ratings, nb_feature, lambda_, lambda_, stop_criterion)
pred_lst.append(pred_SGD)

# ALS
nb_feature = 8
lambda_ = 0.081
pred_ALS, _, _ = ALS_CV(ratings, nb_feature, lambda_, lambda_, stop_criterion)
pred_lst.append(pred_ALS)

# simple_models
print("learn global mean")
pred_globalmean =  global_mean(ratings, 0)
print(pred_globalmean.shape)
pred_lst.append(pred_globalmean)

print("learn user mean")
pred_usermean = user_mean(ratings, 0)
print(pred_usermean.shape)
pred_lst.append(pred_usermean)

print("learn item mean")
pred_itemmean = baseline_item_mean(ratings, 0)
print(pred_itemmean.shape)
pred_lst.append(pred_itemmean)

# put all prediction into one matrix
print("prediction matrix")
prediction = prepareBlending(ratings, pred_lst)
print(len(prediction))

## Blending function
# test without CV
print("learn blending weights")

y = pandas['Rating']
X = np.transpose(prediction)

linreg = Ridge(alpha=0.1, fit_intercept=False)
linreg.fit(X, y)

print('Weights of the different models:', linreg.coef_)
print('Final score:', linreg.score(X, y))

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



