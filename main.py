import numpy as np
import decision_tree as dt
import neural_net as nn
import preprocessing as prep
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
Authors: Lorenzo Limon & Steven Kempers
"""

"""
Import the glass database
"""
glass = np.genfromtxt("data/glass.data", delimiter=",")
X = glass[:, 1:10]  # remove index row and class label
y = glass[:, 10]  # class labels


def run():
    dt_avg = 0
    nn_avg = 0
    iterations = 100
    results = ""

    """ Here we run both algorithms 100 times wich each variation of preprocessing to account for
     the fluctuations in accuracy when running the algorithms """

    """ kfold """
    for i in range(0, iterations):
        X_train, X_test, y_train, y_test = prep.kfold_split(X, y, splits=5)
        dt_avg += dt.trees(X_train, X_test, y_train, y_test)  # running decision tree algorithm
        nn_avg += nn.neural(X_train, X_test, y_train, y_test)  # running neural network algorithm
    results += "Kfold: \n" + "\tDtree:\t" + str(dt_avg / iterations) + "\n\tNN: \t" + str(nn_avg / iterations) + "\n"
    dt_avg, nn_avg = 0, 0

    """ Scaled kfold """
    for i in range(0, iterations):
        X_train, X_test, y_train, y_test = prep.scaled_kfold_split(X, y, splits=5)
        dt_avg += dt.trees(X_train, X_test, y_train, y_test)
        nn_avg += nn.neural(X_train, X_test, y_train, y_test)
    results += "Scaled Kfold: \n" + "\tDtree:\t" + str(dt_avg / iterations) + "\n\tNN: \t" + str(nn_avg / iterations) + "\n"
    dt_avg, nn_avg = 0, 0

    """ Scaled """
    for i in range(0, iterations):
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.20, random_state=42)
        dt_avg += dt.trees(X_train, X_test, y_train, y_test)
        nn_avg += nn.neural(X_train, X_test, y_train, y_test)
    results += "Scaled No Prep: \n" + "\tDtree:\t" + str(dt_avg / iterations) + "\n\tNN: \t" + str(
        nn_avg / iterations) + "\n"
    dt_avg, nn_avg = 0, 0

    """ No Preprocessing """
    for i in range(0, iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        dt_avg += dt.trees(X_train, X_test, y_train, y_test)
        nn_avg += nn.neural(X_train, X_test, y_train, y_test)
    results += " No Prep: \n" + "\tDtree:\t" + str(dt_avg / iterations) + "\n\tNN: \t" + str(nn_avg / iterations) + "\n"
    dt_avg, nn_avg = 0, 0

    return results


print(run())
