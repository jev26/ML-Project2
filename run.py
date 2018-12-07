import numpy as np
import matplotlib.pyplot as plt
from helper import *
import scipy.sparse as sp
from test import load_data_sparse
#from CV import cross_validation
#from SGD.py import SGD_final_prediction
from SGD import *
from ALS import *
from Visualization import cv_result
#from sklearn.linear_model import LinearRegression

data_name = "data/47b05e70-6076-44e8-96da-2530dc2187de_data_train.csv"

ratings, _ = load_data_sparse(data_name, False)

mean_rating_user = np.mean(ratings,axis = 0)
mean_rating_movie = np.mean(ratings,axis = 1)

# Define parameters
SGDModel = True
k_fold = 5
#nb_features = np.arange(10,110,10)
#nb_features = [10, 20, 40, 80]
nb_features = np.arange(10,110,2)
lambdas = np.logspace(-5, -0.3, num=2)
#lambdas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
min_nb_ratings = 3 # min rate par user 3. user qui fait le plus c'est 522. min rate par film 8, max 4590
p_test = 0.1
stop_criterion = 1e-2

prediction = []

## Learning
# in Learning.py ==> aim to find the best parameter for each model
# then the parameters are specified here to learn the best model for the final prediction

## Launch Algorithm
# SGD
nb_feature = 20
lambda_ = 0.004
#pred, error = matrix_factorization_SGD_CV(train, test, nb_feature, lambda_, lambda_, stop_criterion)
#pred = SGD_final_prediction(ratings, nb_feature, lambda_, lambda_, stop_criterion)
pred, _, _ = matrix_factorization_SGD_CV(ratings, nb_feature, lambda_, lambda_, stop_criterion)
prediction.append(pred)

# ALS
nb_feature = 8
lambda_ = 0.081
pred, _, _ = ALS_CV(ratings, nb_feature, lambda_, lambda_, stop_criterion)
prediction.append(pred)

print(prediction)

## Blending function
# faire aussi un learning process pour avoir les poids ?
#clf = LinearRegression()
#clf.fit(X.T, y)

# two types of recommender systems
# --> collaborative filtering

