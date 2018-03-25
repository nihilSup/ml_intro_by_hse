"""It will be too easy to do this task in python arrays.
So I try to use numpy as many as I can"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


def sigmoid(z):
    """z is a vector with length of number of features"""
    return 1./(1. + np.exp(-z))


def calc_z(w, x, y):
    """w is weights vector
    x is feature matrix for some object
    y is target vector"""
    return y * np.dot(X, w)


class GD(object):
    def __init__(self, k=0.1, c=0.0, eps=1.0e-5):
        """k is step, default=0.1
        C is regularization coef, default=10"""
        self.k = k
        self.c = c
        self.eps = eps
        self.w = None

    def __repr__(self):
        return f"GD(k={self.k}, c={self.c}, eps={self.eps})"

    def update_weights(self, w, X, y):
        for i, _ in enumerate(w):
            v = y*X.iloc[:, i]*(1 - sigmoid(calc_z(w, X, y)))
            w[i] = w[i] + self.k * v.mean() - self.k * self.c * w[i]

    def fit(self, X, y, init_weights=None):
        if not init_weights:
            init_weights = np.zeros(X.shape[1])
        elif X.shape[1] != init_weights.shape[0]:
            raise Exception(f'Bad init weight shape {init_weights.shape[0]}. '
                            f'Expected {X.shape[1]}')
        w = init_weights
        num_iter = 1
        while True:
            old_w = w.copy()
            self.update_weights(w, X, y)
            # check euclidean distance for convergence
            cur_eps = np.linalg.norm(old_w - w)
            if cur_eps <= self.eps:
                print(f'Converged in {num_iter} iterations')
                break
            if num_iter < 10000:
                num_iter += 1
            else:
                raise Exception('Convergence failed')
        # store result
        self.w = w

    def predict(self, X):
        return X.apply(lambda X_row: sigmoid(np.dot(self.w, X_row)), axis=1)


ml_folder = 'D:\\work\\python\\ml\\data\\'
data = pd.read_csv(ml_folder+'data-logistic.csv', header=None)
X = data[[1, 2]]
y = data[0]
# train classifier without regularization term
clf = GD()
clf.fit(X, y)
# check classifier
y_clf = clf.predict(X)
clf_score = roc_auc_score(y, y_clf)
# train classifier with regularization term, C = 10
clf_reg = GD(c=10.0)
clf_reg.fit(X, y)
# check classifier with regularization
y_clf_reg = clf_reg.predict(X)
clf_reg_score = roc_auc_score(y, y_clf_reg)
