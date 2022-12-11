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

