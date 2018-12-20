import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def data_exploration(data):
    # 10'000 users and 1'000 films

    # Nbr Rate Per User
    NbrRatePerUser = data['r'].value_counts()
    print('Nbr Rate Per User')
    print('min: ' + str(np.min(NbrRatePerUser)))
    print('max: ' + str(np.max(NbrRatePerUser)))
    print('mean: ' + str(np.mean(NbrRatePerUser)))
    print('================')

    f1 = plt.figure()
    plt.hist(NbrRatePerUser, bins=100)
    plt.xlabel("Number of Ratings")
    plt.ylabel("Number of Users")

    # Nbr Rate Per Film
    NbrRatePerFilm = data['c'].value_counts()
    #print(NbrRatePerFilm)
    print('Nbr Rate Per Film')
    print('min: ' + str(np.min(NbrRatePerFilm)))
    print('max: ' + str(np.max(NbrRatePerFilm)))
    print('mean: ' + str(np.mean(NbrRatePerFilm)))
    print('================')

    f2 = plt.figure()
    plt.hist(NbrRatePerFilm, bins=50)
    plt.xlabel("Number of Ratings")
    plt.ylabel("Number of Films")

    # Mean Rate Per Film
    MeanPerFilm = data.groupby('c').mean()

    print(MeanPerFilm.shape)
    print(type(MeanPerFilm))

    f3 = plt.figure(3)
    plt.hist(np.transpose(MeanPerFilm),bins=30)#,range=[1,5])#bins='auto',range=[1,5])
    plt.xlabel("Mean Rating over all Users")
    plt.ylabel("Number of Films")

    # Variance in Rating Per Film
    StdPerFilm = data.groupby('c').std()

    f4 = plt.figure(4)
    plt.hist(np.transpose(StdPerFilm), bins=30)
    plt.xlabel("Variance over all Users")
    plt.ylabel("Number of Films")

    # Mean Rate Per Film
    MeanPerUser = data.groupby('r').mean()

    f5 = plt.figure(5)
    plt.hist(np.transpose(MeanPerUser), bins=30)  # ,range=[1,5])#bins='auto',range=[1,5])
    plt.xlabel("Mean Rating over all Films")
    plt.ylabel("Number of Users")

    # Variance in Rating Per Film
    StdPerUser = data.groupby('r').std()

    f6 = plt.figure(6)
    plt.hist(np.transpose(StdPerUser), bins=30)
    plt.xlabel("Variance over all Films")
    plt.ylabel("Number of Users")

    plt.show()

    #if we want to save figures
    """
    f1.savefig('Figures/NbrRatePerUser.png')
    f2.savefig('Figures/NbrRatePerFilm.png')
    f3.savefig('Figures/FilmMean.png')
    f4.savefig('Figures/FilmVariance.png')
    f5.savefig('Figures/UserMean.png')
    f6.savefig('Figures/UserVariance.png')
    """