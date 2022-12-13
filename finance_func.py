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
    
    >>> s = pd.Series([2.36, np.nan, np.nan, 2.39, 2.25])
    >>> s
    0    2.36
    1     NaN
    2     NaN
    3    2.39
    4    2.25
    dtype: float64
    
    >>> ff.change(s)
    0         NaN
    1    0.000000
    2    0.000000
    3    0.012712
    4   -0.058577
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
    1   0.011765  0.044199   0.000000
    2  -0.662791  0.055556  -0.001187
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