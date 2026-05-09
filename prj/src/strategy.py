
from __future__ import annotations

import numpy as np
import pandas as pd


def make_regime_positions(df: pd.DataFrame, regimes: pd.Series) -> pd.Series:
    """
    Regime-aware position rule.

    Assumption after ordering states by volatility:
    - Regime 0: low-volatility / calmer environment -> long
    - Regime 1: medium-volatility / uncertain environment -> use momentum
    - Regime 2: high-volatility / stress environment -> cash

    Position is shifted by 1 day to avoid same-day look-ahead.
    """
    raw_position = pd.Series(0.0, index=df.index)

    low_vol = regimes == 0
    medium_vol = regimes == 1
    high_vol = regimes == 2

    raw_position.loc[low_vol] = 1.0
    raw_position.loc[medium_vol] = (df.loc[medium_vol, "momentum"] > 0).astype(float)
    raw_position.loc[high_vol] = 0.0

    return raw_position.shift(1).fillna(0.0)


def make_buy_hold_positions(df: pd.DataFrame) -> pd.Series:
    """
    Always long SPY.
    """
    return pd.Series(1.0, index=df.index)


def make_momentum_positions(df: pd.DataFrame) -> pd.Series:
    """
    Standalone momentum baseline:
    long if 20-day momentum is positive, otherwise cash.
    Shifted by 1 day.
    """
    pos = (df["momentum"] > 0).astype(float)
    return pos.shift(1).fillna(0.0)


def make_mean_reversion_positions(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """
    Standalone mean reversion baseline:
    long after short-term weakness, cash after short-term strength.

    Rule:
    - Compute trailing 5-day return.
    - If trailing return is negative, go long expecting rebound.
    - Else hold cash.

    Shifted by 1 day.
    """
    short_term_return = df["log_price"].diff(lookback)
    pos = (short_term_return < 0).astype(float)
    return pos.shift(1).fillna(0.0)
