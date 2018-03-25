import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
import re
import scipy

from ml_intro.custutils import to_path

# data format:
# Full description - large text, Location Normalized - categorical town or some place
# ContractTime - categorical type of vacancy, SalaryNormalized - nans
data_train = pd.read_csv(to_path('salary-train.csv'))
data_test = pd.read_csv(to_path('salary-test-mini.csv'))
# Full description conversion:
#   replace all not text data by spaces, also convert to lowercase it.
#   convert by TfidfVectorizer to sparse matrix with l rows and tons of
#   feature columns


def process_text(text):
    return re.sub('[^a-zA-Z0-9]', ' ', text.lower())


data_train['FullDescription'] = data_train['FullDescription'].map(process_text)
data_test['FullDescription'] = data_test['FullDescription'].map(process_text)
tfidf_enc = TfidfVectorizer(min_df=5)
# min_df - ignore elements with freq less then x
X_train_tfidf = tfidf_enc.fit_transform(data_train['FullDescription'])
X_test_tfidf = tfidf_enc.transform(data_test['FullDescription'])
# Location Normalized, ContractTime conversion:
#   fill nans with 'nan' to count it as category
#   use DictVectorizer to one-hot encode merge of this two columns,
#   i.e. convert categorical features (merged as one dict) to set of binary
#   features


def fill_na(df):
    df['LocationNormalized'].fillna('nan', inplace=True)
    df['ContractTime'].fillna('nan', inplace=True)


fill_na(data_train)
# fill_na(data_test)
dict_enc = DictVectorizer()
X_train_categ = dict_enc.fit_transform(
    data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = dict_enc.transform(
    data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
# Combine feature sparse matrices
X_train = scipy.sparse.hstack([X_train_tfidf, X_train_categ])
X_test = scipy.sparse.hstack([X_test_tfidf, X_test_categ])
# Get y
y_train = data_train['SalaryNormalized']
# Train Ridge linear regression
reg = Ridge(alpha=1, random_state=241)
reg.fit(X_train, y_train)
# Predict test set
y_test = reg.predict(X_test)
print(f'{y_test[0]:.2f}, {y_test[1]:.2f}')
