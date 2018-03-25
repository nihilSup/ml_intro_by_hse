import pandas as pd
from sklearn.metrics import accuracy_score, precision_score,\
    recall_score, f1_score, roc_auc_score, precision_recall_curve

ml_folder = 'D:\\work\\python\\ml\\data\\'
clf_data = pd.read_csv(ml_folder+'classification.csv')
# classification errors data
tp_data = clf_data[(clf_data['true'] == True) & (clf_data['pred'] == True)]
fp_data = clf_data[(clf_data['true'] == False) & (clf_data['pred'] == True)]
fn_data = clf_data[(clf_data['true'] == True) & (clf_data['pred'] == False)]
tn_data = clf_data[(clf_data['true'] == False) & (clf_data['pred'] == False)]
# classification errors values
tp = len(tp_data)
fp = len(fp_data)
fn = len(fn_data)
tn = len(tn_data)
print(f'Classification errors (tp, fp, fn, tn): '
      f'{tp}, {fp}, {fn}, {tn}')
# main classification measures
# part of correctly classified objects among all
accuracy = (tp + tn) / (tp + fp + fn + tn)
acc_sk = accuracy_score(clf_data['true'], clf_data['pred'])
# part of correctly classified 1-st class objects in all objects which
# classifier considered as 1-st class
precision = tp / (tp + fp)
prec_sk = precision_score(clf_data['true'], clf_data['pred'])
# proportion of correctly classified 1-st class objects in all 1-st class
recall = tp / (tp + fn)
recall_sk = recall_score(clf_data['true'], clf_data['pred'])
# F measure, harmonic average
F_meas = 2*precision*recall/(precision + recall)
F_meas_sk = f1_score(clf_data['true'], clf_data['pred'])
print(f'{accuracy:.2f} {precision:.2f} {recall:.2f} {F_meas:.2f}')
# Classifiers handling
scores_data = pd.read_csv(ml_folder+'scores.csv')
y_true = scores_data['true']
y_pred = scores_data[scores_data.columns[1:]]

# function roc_auc_score returns float score
scores_rocauc = {column: roc_auc_score(y_true, y_pred[column])
                 for column in y_pred.columns}

res = sorted(scores_rocauc.items(), key=lambda i: i[1], reverse=True)
print(f'Best roc auc is {res[0][1]} by classifier {res[0][0]}')

#
best_prec = dict()
for column in y_pred.columns:
    tpl = precision_recall_curve(y_true, scores_data[column])
    # tpl contains 3 arrays: precision, recall, threshold. Each array has
    # n - 2 element. Originally PRC contains n + 1 thresholds, but this
    # implementation doesn't store 0 and 1 thresholds
    scores_df = pd.DataFrame({'precision': tpl[0], 'recall': tpl[1]})
    best_prec[column] = scores_df[scores_df['recall'] >= 0.7]['precision'].max()
print(f'Best precision witn recall >= 0.7 is classifier:'
      f' {max(best_prec, key=lambda k: best_prec[k])}')
