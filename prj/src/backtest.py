
from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    log_returns: pd.Series,
    positions: pd.Series,
    cost_bps: float = 5.0,
) -> dict[str, pd.Series | float]:
    """
    Run simple daily close-to-close backtest.

    Parameters
    ----------
    log_returns:
        Daily log returns.
    positions:
        Daily exposure. 1 = long, 0 = cash.
        Should already be shifted to avoid look-ahead.
    cost_bps:
        One-way transaction cost in basis points.

    Returns
    -------
    dict with daily returns, equity, drawdown, turnover, etc.
    """
    log_returns = log_returns.copy()
    positions = positions.reindex(log_returns.index).fillna(0.0)

    simple_asset_returns = np.exp(log_returns) - 1.0

    turnover = positions.diff().abs().fillna(positions.abs())
    cost_rate = cost_bps / 10_000.0
    costs = turnover * cost_rate

    strategy_return = positions * simple_asset_returns - costs
    equity = (1.0 + strategy_return).cumprod()

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0

    return {
        "strategy_return": strategy_return,
        "equity": equity,
        "drawdown": drawdown,
        "positions": positions,
        "turnover": turnover,
        "costs": costs,
        "total_return": equity.iloc[-1] - 1.0,
        "annualized_turnover": turnover.mean() * 252,
        "trades": int((turnover > 0).sum()),
        "exposure": positions.mean(),
    }
