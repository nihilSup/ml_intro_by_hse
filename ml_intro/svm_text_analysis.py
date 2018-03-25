import numpy as np
from scipy.sparse import find
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import KFold, GridSearchCV

# Data upload
newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
    )
# Feature preprocessing - convert text data to numeric by TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroups.data)
# X shape is (1786, 28382). Where
# 1786 is number of objects and
# 28382 is number of features
# Every object is text, features is TF-IFD index of word, every columns is
# index for word in union of all words in all texts. So it is very sparse matrix
# TF represents frequency of word in current text. IFD is some measure of
# uniqueness of word among all texts, i.e. if word is belong only to current
# text it will be high
y = newsgroups.target
# y is digit classes
# Searching best 'C' param, using exhaustive grid search
grid = {'C': np.power(10.0, np.arange(-5, 6))}
kf = KFold(n_splits=5, shuffle=True, random_state=241)
clf = svm.SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=kf)
# next line is heavy
gs.fit(X, y)
# Results of grid search
# old school: gs.grid_scores_
# print some info:
for sc_mean, sc_std, sc_par in zip(gs.cv_results_['mean_test_score'],
                                   gs.cv_results_['std_test_score'],
                                   gs.cv_results_['params']):
    print(f"For C={sc_par}: mean is {sc_mean}, std is {sc_std}")
# New way:
print(f'Best score is {gs.best_score_}, best params is {gs.best_params_}')
best_C = gs.best_params_['C']
# Use best C to train classifier
clf_best = svm.SVC(kernel='linear', random_state=241, C=best_C)
clf_best.fit(X, y)
# Top ten words by coef. Also one can use pandas frame constructed on
# scipy sparse matrix, using mtx.data, mtx.indices
# get dense repr of weights in SVM classifier: row, col, val
# row is zeroes so can be skipped
dense_repr = find(clf_best.coef_)
j_v = [(j, abs(v)) for j, v in zip(dense_repr[1], dense_repr[2])]
# sort by value
j_v_sorted = sorted(j_v, key=lambda x: x[1])
# get map index to word (simple array)
feature_map = vectorizer.get_feature_names()
# most weighted features(words)
top_words = [feature_map[tpl[0]] for tpl in j_v_sorted[-10:]]
result = ','.join(sorted(top_words)); result

