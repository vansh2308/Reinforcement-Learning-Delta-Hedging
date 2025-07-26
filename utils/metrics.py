import numpy as np

def compute_sharpe_ratio(returns, risk_free_rate=0.0):
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0.0
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    return sharpe

def compute_cumulative_pnl(rewards):
    return np.cumsum(rewards)

def compute_var(returns, confidence_level=0.95):
    """Compute Value at Risk (VaR)"""
    returns = np.array(returns)
    var = np.percentile(returns, (1 - confidence_level) * 100)
    return var

def compute_cvar(returns, confidence_level=0.95):
    """Compute Conditional Value at Risk (CVaR) / Expected Shortfall"""
    returns = np.array(returns)
    var = compute_var(returns, confidence_level)
    cvar = np.mean(returns[returns <= var])
    return cvar

def compute_drawdown(pnl_series):
    """Compute drawdown series and maximum drawdown"""
    pnl_series = np.array(pnl_series)
    peak = np.maximum.accumulate(pnl_series)
    drawdown = (pnl_series - peak) / peak
    max_drawdown = np.min(drawdown)
    return drawdown, max_drawdown

def compute_volatility(returns):
    """Compute annualized volatility"""
    returns = np.array(returns)
    return np.std(returns) * np.sqrt(252)

def compute_sortino_ratio(returns, risk_free_rate=0.0):
    """Compute Sortino ratio (Sharpe ratio using downside deviation)"""
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return 0.0
    downside_deviation = np.std(downside_returns)
    if downside_deviation == 0:
        return 0.0
    sortino = np.mean(excess_returns) / downside_deviation * np.sqrt(252)
    return sortino 