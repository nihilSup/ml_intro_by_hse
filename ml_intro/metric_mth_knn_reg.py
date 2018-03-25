import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from ml_intro.custutils import cross_validate_est

bunch = datasets.load_boston()
features_scaled = preprocessing.scale(bunch['data'])
y = bunch['target']
p_values = np.linspace(1, 10, num=200)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
res = pd.DataFrame(
    [[p_val, cross_val_score(KNeighborsRegressor(n_neighbors=5, weights='distance', p=p_val),
                             features_scaled, y, cv=kf, scoring='neg_mean_squared_error')]
     for p_val in p_values],
    columns=['p', 'scores']
)
res.set_index('p', inplace=True)
res['mean'] = res['scores'].map(lambda s: s.max())
res.sort_values(by='mean', ascending=False, inplace=True)

res2 = cross_validate_est(KNeighborsRegressor, features_scaled, y, kf,
                          p_values, 'p', scoring='neg_mean_squared_error',
                          est_params={'weights': 'distance'}, agg_f=np.max)
res2 = res2.sort_values(by='amax', ascending=False)
