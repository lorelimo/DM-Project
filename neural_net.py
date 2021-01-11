from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


def neural(X_train, X_test, y_train, y_test, make_plot: bool = False):
    clf = MLPClassifier(solver='sgd', alpha=1e-6, hidden_layer_sizes=(9, 9), random_state=1, max_iter=90000)
    clf.fit(X_train, y_train)
    loss_values = clf.loss_curve_

    if make_plot:
        plot_loss(loss_values)

    """ Printing accuracy to console"""
    # print("[Neural Network] \t Accuracy:",
    #       clf.score(X_test, y_test))  # uncomment this print if you want to see accuracy during runtime
    return clf.score(X_test, y_test)


def plot_loss(loss_values):
    """ Plotting the loss function """
    plt.plot(loss_values, color="RED")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
