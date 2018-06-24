import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import itertools

from ml_intro.custutils import cross_validate_est, to_path

# data upload
features = pd.read_csv(to_path('final_statement\\features.csv'),
                       index_col='match_id')
# remove match results features
res_features = ['duration', 'radiant_win', 'tower_status_radiant',
                'tower_status_dire', 'barracks_status_radiant',
                'barracks_status_dire']
feat_wo_res = features.drop(columns=res_features)
# find features with skipped values
col_with_skips = feat_wo_res.isnull().sum()
col_with_skips = col_with_skips[col_with_skips > 0]
print(f"Features with missed values:\n{list(col_with_skips.index)}")
# Короткое объяснение - все эти признаки описывают события, которые могли не
# не произойти за первые 5 минут игры
# Все множество признаков с пропусками можно разбить на три группы и дать
# объяснение для каждой группы:
# 1) признаки связанные с событием first blood - событие могло не произойти в
#    заданный отрезок времени (первые 5 минут игры).
#    а) Видно, что признак first_blood_player2 отсутсвует куда больше чем
#       остальные из этой категории. Если предположить, что это игрок который
#       совершил убийство, то для случаев, когда другого игрока убил NPC, этот
#       признак будет пустым. Если это второй игрок, который помогал первому
#       кого-то убивать - в таком случае возможны ситуации, когда первый игрок
#       проделал всю работу в одиночку.
# 2) признаки связанные с покупкой предмета - предмет могли не купить в заданный
#    отрезок времени
# 3) признаки связанные с первой установкой ward_observer - событие могло не
#    произойти в заданный отрезок времени
# null values handling
feat_wo_res.fillna(0.0, inplace=True)
# TO-DO: check other ways for GBoost - fill with mean, large value, small value
# Looks like it doesn't matter
# target value is 'radiant_win'
y = features['radiant_win']
# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=241)
# Gradient Boosting
n_estim_range = [10]  # 20, 30, 50, 100, 250]
results_gb = cross_validate_est(GradientBoostingClassifier, feat_wo_res, y, cv=kf,
                                scoring='roc_auc',
                                est_params={'random_state': 241},
                                par_key='n_estimators',
                                par_values=n_estim_range,
                                timed=True)
# plot gradient boost results
plt.plot(results_gb.index, results_gb['mean'])
plt.xlabel('n_estimators')
plt.ylabel('roc auc score')
plt.show()
# print summary
best_n_est = results_gb['mean'].idxmax()
exec_time = results_gb.loc[best_n_est, 'exec time (s)']
print(f'Best estimator number: {best_n_est}, exec time: {exec_time}')
# Logistic regression


def find_best_c(results):
    # plot logistic regression results
    plt.plot(np.log(results.index), results['mean'])
    plt.xlabel('log(C)')
    plt.ylabel('roc auc score')
    plt.show()
    # Best C
    best_C = results['mean'].idxmax()
    best_C_time = results.loc[best_C, 'exec time (s)']
    return best_C, results.loc[best_C, 'mean'], best_C_time


# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(feat_wo_res)
# Cross-validate setup
C_range = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01, 0.001, 0.0001]
cv_reg_def = {'cv': kf, 'scoring': 'roc_auc',
              'est_params': {'random_state': 241},
              'par_key': 'C', 'par_values': C_range,
              'timed': True}
# Logistic regression cross-validate
results = cross_validate_est(LogisticRegression, X, y,
                             **cv_reg_def)
find_best_c(results)
# remove category features
cat_features = ['lobby_type'] + \
               [f'{prod[0]}{prod[1]}_hero'
                for prod in itertools.product(['r', 'd'], range(1, 6))]
feat_wo_cat = feat_wo_res.drop(columns=cat_features)
X_wo_cat = scaler.fit_transform(feat_wo_cat)
# Logistic regression cross-validate on data w/o categorical features
res_wo_cat = cross_validate_est(LogisticRegression, X_wo_cat, y,
                                **cv_reg_def)
find_best_c(res_wo_cat)
# Get number of unique heroes.
# Match based approach. I can use cat_featuers[1:6] and [6:11] but
# prefer more explicit (also independent) way:
r_heroes = [f'r{i}_hero' for i in range(1, 6)]
d_heroes = [f'd{i}_hero' for i in range(1, 6)]
# Dictionary based approach (looks like some heroes have never been chosen):
heroes_df = pd.read_csv(to_path('final_statement\\dictionaries\\heroes.csv'),
                        index_col='id')
n_heroes = len(heroes_df)
print(f'Number of heroes according to dict: {n_heroes}')
num_rad_her = len(np.unique(feat_wo_res[r_heroes].values))
print(f'Number of unique radiant heroes: {num_rad_her}')
num_dire_her = len(np.unique(feat_wo_res[d_heroes].values))
print(f'Number of unique dire heroes: {num_dire_her}')
# Bag of words approach for picked heroes:
#   for every match (row) we create array with shape (1, number of heroes)
#   each element of array corresponds to specific hero
#   if hero played for radiant we mark it as 1, if played as dire -1, if didn't 0


def create_bag(X, n):
    X_bag = np.zeros((X.shape[0], n))
    for i, match_id in enumerate(X.index):
        for p in range(5):
            X_bag[i, X.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
            X_bag[i, X.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1
    return X_bag


X_pick = create_bag(feat_wo_res, n_heroes)
# glue together
X_full = np.hstack([X_wo_cat, X_pick])
# logistic regression cross-validate of full features with bag of words cats
res_full_lr = cross_validate_est(LogisticRegression, X_full, y, **cv_reg_def)
find_best_c(res_full_lr)
# Predict for Kaggle


def pre_process(df, feat_to_drop, n):
    df.fillna(0.0, inplace=True)
    df_wo = df.drop(columns=feat_to_drop)
    X = StandardScaler().fit_transform(df_wo)
    X_bagged = create_bag(df, n)
    return np.hstack([X, X_bagged])


clf = LogisticRegression(C=0.1)
clf.fit(X_full, y)
# get test data and pre process it
features_test = pd.read_csv(to_path('final_statement\\features_test.csv'),
                            index_col='match_id')
X_kaggle = pre_process(features_test, cat_features, n_heroes)
# get prediction for 1-st class
y_test = clf.predict_proba(X_kaggle)[:, 1]
# get max and min values
print(f'Test max: {y_test.max()}, min: {y_test.min()}')
# save results to file
y_test_df = pd.DataFrame({'radiant_win': y_test}, index=features_test.index)
y_test_df.index.name = 'match_id'
y_test_df.to_csv(f'./kaggle_pred_1.csv')
