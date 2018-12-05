import numpy as np
import matplotlib.pyplot as plt

def cv_result(errors, nb_features, lambdas):

    #Nbr_features = nb_features.size
    #Nbr_lambdas = lambdas.size

    plt.figure()

    #for lambda_ in lambdas:
    #    plt.plot(errors[:, lambda_])
    plt.plot(errors)
    plt.legend(labels=lambdas)

    plt.show()