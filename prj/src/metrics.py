
from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    total = (1.0 + returns).prod()
    years = len(returns) / periods_per_year
    return total ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return returns.dropna().std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, rf_daily: float = 0.0) -> float:
    excess = returns.dropna() - rf_daily
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return np.nan
    return excess.mean() / vol * np.sqrt(periods_per_year)


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return dd.min()


def calmar_ratio(returns: pd.Series, equity: pd.Series) -> float:
    cagr = annualized_return(returns)
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return np.nan
    return cagr / mdd


def hit_rate(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    return (returns > 0).mean()


def metrics_table(backtest_results: dict) -> pd.DataFrame:
    rows = []

    for name, result in backtest_results.items():
        r = result["strategy_return"]
        eq = result["equity"]

        rows.append(
            {
                "strategy": name,
                "total_return": result["total_return"],
                "CAGR": annualized_return(r),
                "annualized_vol": annualized_volatility(r),
                "Sharpe": sharpe_ratio(r),
                "max_drawdown": max_drawdown(eq),
                "Calmar": calmar_ratio(r, eq),
                "hit_rate": hit_rate(r),
                "annualized_turnover": result["annualized_turnover"],
                "trades": result["trades"],
                "exposure": result["exposure"],
            }
        )

    return pd.DataFrame(rows).set_index("strategy")
