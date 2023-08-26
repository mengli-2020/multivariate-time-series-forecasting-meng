import re
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.tsa.stattools as ts


def covariance(x: np.ndarray, y: np.ndarray) -> float:
    """ Covariance between x and y
    """
    cov_xy = np.cov(x, y)[0][1]

    return cov_xy


def co_integration(x: np.ndarray, y: np.ndarray):
    """ Co-integration test between x and y
    """
    r, _, _ = ts.coint(x, y)

    return r


def correlation(x: np.ndarray,
                y: np.ndarray,
                method: str = "kendall"):
    """ Correlation between x and y
    """
    assert method in ["pearson", "spearman", "kendall"]

    corr, p_value = stats.kendalltau(x, y)

    return corr

def time_delay_embedding(series: pd.Series,
                         n_lags: int,
                         horizon: int,
                         return_Xy: bool = False):
    """
    Time delay embedding
    Time series for supervised learning

    :param series: time series as pd.Series
    :param n_lags: number of past values to used as explanatory variables
    :param horizon: how many values to forecast
    :param return_Xy: whether to return the lags split from future observations

    :return: pd.DataFrame with reconstructed time series
    """
    assert isinstance(series, pd.Series)

    if series.name is None:
        name = 'Series'
    else:
        name = series.name

    n_lags_iter = list(range(n_lags, -horizon, -1))

    df_list = [series.shift(i) for i in n_lags_iter]
    df = pd.concat(df_list, axis=1).dropna()
    df.columns = [f'{name}(t-{j - 1})'
                  if j > 0 else f'{name}(t+{np.abs(j) + 1})'
                  for j in n_lags_iter]

    df.columns = [re.sub('t-0', 't', x) for x in df.columns]

    if not return_Xy:
        return df

    is_future = df.columns.str.contains('\+')

    X = df.iloc[:, ~is_future]
    Y = df.iloc[:, is_future]
    if Y.shape[1] == 1:
        Y = Y.iloc[:, 0]

    return X, Y