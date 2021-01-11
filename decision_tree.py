from sklearn import tree
import matplotlib.pyplot as plt


def trees(X_train, X_test, y_train, y_test, plot_results: bool = False):
    clf = tree.DecisionTreeClassifier(min_samples_split=20, criterion="gini")
    clf.fit(X_train, y_train)

    if plot_results:
        plot_tree(clf)

    """ Calculating and printing accuracy to console"""
    count = 0
    for i in range(len(X_test)):
        c = clf.predict([X_test[i]])
        if c == [y_test[i]]:
            count += 1
    accuracy = count / len(X_test)
    # print("[Decision Tree] \t Accuracy:", accuracy) # uncomment this print if you want to see accuracy during runtime

    return accuracy


def plot_tree(clf):
    """ Plotting the decision tree """
    fig, ax = plt.subplots(figsize=(32, 32))
    fig.suptitle("Glass Type decision tree", fontsize=50, y=0.92)
    tree.plot_tree(clf, fontsize=25)
    plt.show()
