from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

""" Implementation of the KFold cross validation """


def kfold_split(X, y, splits: int = 5):
    kf = KFold(n_splits=splits, shuffle=True)
    return do_the_split(X, kf, y)


def scaled_kfold_split(X, y, splits: int = 5):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    kf = KFold(n_splits=splits, shuffle=True)
    return do_the_split(scaled, kf, y)


def do_the_split(X, kf: KFold, y):
    X_train, X_test, y_train, y_test = [], [], [], []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test
