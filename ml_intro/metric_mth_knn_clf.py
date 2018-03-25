import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

# Data upload and target/feature split
ml_folder = 'D:\\work\\python\\ml\\data\\'
wine_data = pd.read_csv(ml_folder+'wine_data.csv', header=None)
classes = wine_data.iloc[:, 0]
features = wine_data.iloc[:, 1:]
# Prepare Cross-validation folds generator
kf = KFold(n_splits=5, shuffle=True, random_state=42)


def knn_cross_validate(x, y, cv, k_num):
    res = pd.DataFrame(
        [(k, cross_val_score(KNeighborsClassifier(n_neighbors=k),
                             x, y, cv=cv, scoring='accuracy'))
         for k in range(1, k_num + 1)],
        columns=['k', 'accuracy']
    )
    res.set_index(['k'], inplace=True)
    res['mean'] = res['accuracy'].map(lambda s: np.mean(s))
    # I want both - raw data and best case, for data analysis purposes
    return res.sort_values(by='mean', ascending=False)


def cross_validate_est(est_cls, x, y, cv, par_values, par_key,
                       scoring='accuracy', est_params=None, agg_f=np.mean):
    """Variates parameter par_key of estimator created by est_cls
    (est_params is kwargs for constructor) by taking them from
    sequence par_vlaues and cross-validates results. Cross-valdation
    is performed on x,y by strategy cv.
    agg_f - is used to aggregate array of cross validation results for
    each parameter value. Default: np.mean"""
    if not est_params:
        est_params = {}
    res = pd.DataFrame(
        [(key, cross_val_score(est_cls(**{**est_params, par_key: key}),
                               x, y, cv=cv, scoring=scoring))
         for key in par_values],
        columns=[par_key, scoring]
    )
    res.set_index([par_key], inplace=True)
    res['mean'] = res[scoring].map(lambda s: agg_f(s))

    return res.sort_values(by='mean', ascending=False)


k_values = range(1, 51)
# Basic case
accuracy = knn_cross_validate(features, classes, kf, 50)

# Scaled features case
f_scaled = preprocessing.scale(features)
acc_scaled = knn_cross_validate(f_scaled, classes, kf, 50)
