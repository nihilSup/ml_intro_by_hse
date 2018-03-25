import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from ml_intro.custutils import to_path

# date, first company, second, ..., thirty company
close_prices = pd.read_csv(to_path('close_prices.csv'), index_col='date')
# Train PCA - i.e transform original 30 features space to n_components space
pca = PCA(n_components=10)
pca.fit(close_prices)
# How many features needed to cover 90% of variance:
# NB! Argmax stops at first max value and returns it's index
expl = pca.explained_variance_ratio_
np.argmax(np.cumsum(expl) > 0.9) + 1
# Transform original data and get first column
approx = [row[0] for row in pca.transform(close_prices)]
# Download Dow-Jones index
djia_data = pd.read_csv(to_path('djia_index.csv'), index_col='date')
djia_data['approx'] = approx
# Calculate pearson correlation
djia_data.corr()
print(f'Pearson correlation: {djia_data.corr().iloc[0, 1]:.2f}')
# another way using numpy
np.corrcoef(djia_data, rowvar=False)
# Company which has most weight in first component
cmp = close_prices.columns[np.argmax(pca.components_[0])]
print(f'Company with most weight in first component is {cmp}')

