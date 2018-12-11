import pandas as pd
import surprise as spr


def pandas_to_surprise(data):
    reader = spr.Reader(rating_scale=(1, 5))
    data_spr = spr.Dataset.load_from_df(data[['User', 'Movie', 'Rating']], reader)
    return data_spr


def surprise_to_matrix(data_spr):
    #todo
    return

def surprise_SVD(trainset, testset):

    algo = spr.SVD()

    algo.fit(trainset)
    predictions = algo.test(testset)
    # Then compute RMSE
    rmse = spr.accuracy.rmse(predictions)

    return predictions, rmse





