import pandas as pd
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import cross_val_score


def resample_time(index, freq):
    return pd.date_range(index.min(), index.max(), freq=freq)


def get_offset(data, nlast):
    return len(data) - nlast if nlast > 0 else 0


def calc_r2(y_true=None, y_pred=None, nlast=0):
    """Calculates coef of determination r2 = 1. - SSR/SST
    where SSR is squared sum of residuals:
        sum( (predicted_i - true_i)**2 )
    and SST is squared sum total:
        sum( (true_i - true_mean)**2 )
    Before calculating removes trailing nans from both
    predicted and true series based on index of first
    non nan prediction"""
    try:
        # remove trailing zeroes using predicted series
        #first_not_nan = y_pred.first_valid_index()
        #y_pred = y_pred.loc[first_not_nan:, :].fillna(0)
        #y_true = y_true.loc[first_not_nan:, :].fillna(0)
        y_true = y_true.iloc[get_offset(y_true, nlast):, :].fillna(0.)
        y_pred = y_pred.iloc[get_offset(y_pred, nlast):, :].fillna(0.)
        r2 = r2_score(y_true, y_pred)
    except Exception as exc:
        print('Failed y_true:')
        print(y_true)
        print('Failed y_pred:')
        print(y_pred)
        raise exc
    return r2


def get_data(facts_groups, forecast):
    """Generator, produces tuples:
    (coordinates, facts_slice, forecast_slice)
    for every group in facts_groups"""
    for idx, sub_df in facts_groups:
        if idx in forecast.columns:
            f_slice = forecast.loc[:, [idx]]
            yield (idx, sub_df, f_slice)


def split_series(df, hist_len=0.5):
    hist_len = float(hist_len)
    if not (0 <= hist_len < 0.9):
        raise Exception(f'Incorrect delim: {delim}')
    hist_size = int(len(df) * hist_len)
    hist_df = df.iloc[0:hist_size].fillna(0.)
    test_df = df.iloc[hist_size:].fillna(0.)
    # replace hidden columns to enable arima processing:
    hist_df.columns = ['history']
    test_df.columns = ['test']
    return hist_df, test_df


def cross_validate_est(est_cls, x, y, cv, par_values, par_key,
                       scoring='accuracy', est_params=None, agg_f=np.mean):
    """Variates parameter par_key of estimator created by est_cls
    (est_params is kwargs for constructor) by taking them from
    sequence par_vlaues and cross-validates results. Cross-valdation
    is performed on x,y by strategy cv.
    agg_f - is used to aggregate array of cross validation results for
    each parameter value. Default: np.mean. If None passed then no aggregation
    performs and raw data pd frame returned """
    if not est_params:
        est_params = {}
    res = pd.DataFrame(
        [(key, cross_val_score(est_cls(**{**est_params, par_key: key}),
                               x, y, cv=cv, scoring=scoring))
         for key in par_values],
        columns=[par_key, scoring]
    )
    res.set_index([par_key], inplace=True)
    if agg_f:
        agg_name = agg_f.__name__
        res[agg_name] = res[scoring].map(lambda s: agg_f(s))
        return res
    else:
        return res


def to_path(filename):
    """returns full path to file in ml folder on my pc"""
    return 'D:\\WORK\ml\\data\\' + filename
