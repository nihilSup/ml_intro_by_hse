import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from ml_intro.custutils import to_path, cross_validate_est

# Random forest regression
# Predicting age of sea shell by natural observed parameters
# data upload
data = pd.read_csv(to_path('abalone.csv'))
# transform sex feature from text to number: F -> -1, I -> 0, M -> 1
data['Sex'] = data['Sex'].map(
    lambda s: -1 if s == 'F' else (0 if s == 'I' else '1'))
# Separate y and X
y = data['Rings']
X = data.iloc[:, :-1]
# Cross validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=1)
# Create Random Forest estimator
n_estimators_range = range(1, 51)
results = cross_validate_est(RandomForestRegressor, X, y, cv=kf, scoring='r2',
                             est_params={'random_state': 1},
                             par_key='n_estimators',
                             par_values=n_estimators_range)
# number of trees which provides score greater then 0.52
tree_num = (results['mean'] > 0.52).idxmax()
print(f'{tree_num}')
# Some practice (not in task)
exmpl_est = RandomForestRegressor(n_estimators=50)
exmpl_est.fit(X.iloc[:3500, :], y.iloc[:3500])
y_pred = pd.DataFrame(exmpl_est.predict(X.iloc[3500:, :]))
y_pred['true_value'] = y.iloc[3500:]
