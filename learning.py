import pickle
from SGD import *
from ALS import *
from surprise_models import *
from simple_models import *

def learning(trainset, testset, labels, PATHSAVE):
    """Function to generate all models, train them and save predictions based on the testset"""
    baseline = surprise_baseline(trainset, testset)
    pickle.dump(baseline, open(PATHSAVE + 'baseline.pkl', 'wb'))

    svd = surprise_SVD(trainset, testset)
    pickle.dump(svd, open(PATHSAVE + 'svd.pkl', 'wb'))

    so = surprise_slopeOne(trainset, testset)
    pickle.dump(so, open(PATHSAVE + 'so.pkl', 'wb'))

    bsknn = surprise_baselineKNN(trainset, testset)
    pickle.dump(bsknn, open(PATHSAVE + 'bsknn.pkl', 'wb'))

    num_features = 40  # K in the lecture notes
    lambda_user = 0.1
    lambda_film = 0.1
    stop_criterion = 1e-4
    als = ALS_CV(trainset, testset, num_features, lambda_user, lambda_film, stop_criterion)
    pickle.dump(als, open(PATHSAVE + 'als.pkl', 'wb'))

    sgd = matrix_factorization_SGD_CV(trainset, testset, num_features, lambda_user, lambda_film, stop_criterion)
    pickle.dump(sgd, open(PATHSAVE + 'sgd.pkl', 'wb'))

    svdpp = surprise_SVDpp(trainset, testset)
    pickle.dump(svdpp, open(PATHSAVE + 'svdpp.pkl', 'wb'))

    globalmean = global_mean(trainset, testset)
    pickle.dump(globalmean, open(PATHSAVE + 'globalmean.pkl', 'wb'))

    usermean = user_mean(trainset, testset)
    pickle.dump(usermean, open(PATHSAVE + 'usermean.pkl', 'wb'))

    itemmean = item_mean(trainset, testset)
    pickle.dump(itemmean, open(PATHSAVE + 'itemmean.pkl', 'wb'))

    basicknn = surprise_basicKNN(trainset, testset)
    pickle.dump(basicknn, open(PATHSAVE + 'basicknn.pkl', 'wb'))

    globalmedian = global_median(trainset, testset)
    usermedian = user_median(trainset, testset)
    itemmedian = item_median(trainset, testset)
    pickle.dump(globalmedian, open(PATHSAVE + 'globalmedian.pkl', 'wb'))
    pickle.dump(usermedian, open(PATHSAVE + 'usermedian.pkl', 'wb'))
    pickle.dump(itemmedian, open(PATHSAVE + 'itemmedian.pkl', 'wb'))