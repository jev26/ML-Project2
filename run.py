import numpy as np
import matplotlib.pyplot as plt
from helper import *
import scipy.sparse as sp
from test import load_data_sparse
from CV import cross_validation
from Visualization import cv_result

data_name = "data/47b05e70-6076-44e8-96da-2530dc2187de_data_train.csv"

ratings, _ = load_data_sparse(data_name, False)

mean_rating_user = np.mean(ratings,axis = 0)
mean_rating_movie = np.mean(ratings,axis = 1)

# Define parameters
SGDModel = True
k_fold = 5
#nb_features = np.arange(10,110,10)
nb_features = [10, 20, 40, 80]
lambdas = np.logspace(-5, -0.3, num=2)
#lambdas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
min_nb_ratings = 3 # min rate par user 3. user qui fait le plus c'est 522. min rate par film 8, max 4590
p_test = 0.1
stop_criterion = 1e-2

# Launch Algorithm
errors = cross_validation(SGDModel, ratings, k_fold, nb_features, lambdas, min_nb_ratings, p_test, stop_criterion)

# Visualization
cv_result(errors, nb_features, lambdas)

