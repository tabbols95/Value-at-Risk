import pandas as pd
import numpy as np

def change(data, periods=1):
    """
    Percentage change between the current and a prior element.
    
    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        Initial data.
    periods : int, default 1
        Periods to shift for forming percent change.
    
    Returns
    -------
    chg : pandas.DataFrame or pandas.Series
        The same type as the calling object.
    
    Examples
    --------
    **Series**
    
    >>> s = pd.Series([20, 25, 27.2], name='TST')
    >>> s
    0    20.0
    1    25.0
    2    27.2
    Name: TST, dtype: float64
    
    >>> ff.change(s)
    0      NaN
    1    0.250
    2    0.088
    Name: TST, dtype: float64
    
    **DataFrame**
    
    >>> df = pd.DataFrame({
    ...    'TST0': [1.365, 1.352, 1.587],
    ...    'TST1': [865.39, 901.12, 899.02],
    ...    'TST2': [12.83, np.nan, 12.95]},
    ...    index=['2022-01-01', '2022-01-02', '2022-01-03'])
    >>> df
                 TST0    TST1   TST2
    2022-01-01  1.365  865.39  12.83
    2022-01-02  1.352  901.12    NaN
    2022-01-03  1.587  899.02  12.95
    >>> ff.change(df)
                     TST0       TST1      TST2
    2022-01-01        NaN        NaN       NaN
    2022-01-02  -0.009524   0.041288  0.000000
    2022-01-03   0.173817  -0.002330  0.009353
    
    **Other type data**
    
    >>> ff.change(55)
    'MsgError: data type pandas.DataFrame or pandas.Series'
    """
    
    return data.pct_change(periods=periods, fill_method='ffill')

def change_ln(data, periods=1):
    """
    Logarithmic change between the current and a prior element.
    
    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        Initial data.
    periods : int, default 1
        Periods to shift for forming percent change.
    
    Returns
    -------
    chg_ln : pandas.DataFrame or pandas.Series
        The same type as the calling object.
    
    Examples
    --------
    **Series**
    
    >>> s = pd.Series([28, 30, 28], name='TST')
    >>> s
    0    28
    1    30
    2    28
    Name: TST, dtype: int64
    
    >>> ff.change_ln(s)
    0         NaN
    1    0.068993
    2   -0.068993
    Name: TST, dtype: float64
    
    >>> s = pd.Series([np.nan, 12, np.nan, np.nan, 12.5, 12.85, 12.83])
    >>> s
    0      NaN
    1    12.00
    2      NaN
    3      NaN
    4    12.50
    5    12.85
    6    12.83
    dtype: float64
    
    >>> ff.change_ln(s)
    0         NaN
    1         NaN
    2    0.000000
    3    0.000000
    4    0.040822
    5    0.027615
    6   -0.001558
    dtype: float64
    
    **DataFrame**
    
    >>> df = pd.DataFrame({
    ...    'TST1': [85, 86, 29],
    ...    'TST2': [36.2, 37.8, 39.9],
    ...    'TST3': [0.1685, np.nan, 0.1683]})
    >>> df
       TST1  TST2    TST3
    0    85  36.2  0.1685
    1    86  37.8     NaN
    2    29  39.9  0.1683
    >>> ff.change(df)
            TST1      TST2       TST3
    0        NaN       NaN        NaN
    1   0.011696  0.043250   0.000000
    2  -1.087051  0.054067  -0.001188
    """
    data = data.fillna(method='pad')
    return np.log(data/data.shift(periods))

def volatility(returns, periods=1):
    """
    Parameters
    ----------
    returns : pandas.DataFrame
    
    Returns
    -------
    volatility : pd.Series
    """
    
    # Calculating volatility
    if isinstance(returns, pd.DataFrame):
        return returns.pct_change(periods=periods).std(axis=0).mul(100)

def value_at_risk(returns, confidence_level=.05):
    """
    Return Value at Risk.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Description
    confidence_level : float, default 0.05
        Description
    
    Returns
    -------
    var : pd.Series
    """
    
    # Calculating Value at Risk
    return returns.quantile(confidence_level)