import pandas as pd

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