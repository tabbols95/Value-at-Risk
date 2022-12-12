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
    """
    
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