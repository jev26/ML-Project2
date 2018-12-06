import numpy as np
import matplotlib.pyplot as plt

def cv_result(errors, nb_features, lambdas):

    plt.figure()

    plt.plot(errors)
    plt.legend(labels=lambdas)

    plt.show()