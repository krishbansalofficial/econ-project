from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def make_features(
    price_df: pd.DataFrame,
    vix_df: pd.DataFrame | None = None,
    vol_window: int = 20,
    momentum_window: int = 20,
) -> pd.DataFrame:
    """
    Convert OHLCV prices into model features.

    Features:
    - log_return
    - rolling_vol
    - momentum
    - vix
    - volume_z
    - amihud_liquidity
    """
    df = price_df.copy()

    if "close" not in df.columns:
        raise ValueError("price_df must contain a 'close' column.")

    if "volume" not in df.columns:
        raise ValueError("price_df must contain a 'volume' column.")

    df["log_price"] = np.log(df["close"])
    df["log_return"] = df["log_price"].diff()

    df["rolling_vol"] = df["log_return"].rolling(vol_window).std()
    df["momentum"] = df["log_price"].diff(momentum_window)

    df["simple_return"] = df["close"].pct_change()
    df["range"] = (df["high"] - df["low"]) / df["close"]

    # Volume feature: abnormal volume relative to recent history
    df["log_volume"] = np.log(df["volume"])
    df["volume_z"] = (
        df["log_volume"] - df["log_volume"].rolling(60).mean()
    ) / df["log_volume"].rolling(60).std()

    # Amihud illiquidity proxy
    # Higher value means lower liquidity / higher price impact
    df["dollar_volume"] = df["close"] * df["volume"]
    df["amihud_liquidity"] = (
        df["log_return"].abs() / df["dollar_volume"]
    ).rolling(20).mean()

    if vix_df is not None:
        df = df.join(vix_df, how="left")
        df["vix"] = df["vix"].ffill()
    else:
        df["vix"] = np.nan

    df = df.dropna().copy()
    return df


def split_train_test(
    df: pd.DataFrame,
    split_date: str = "2021-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df.loc[df.index < split_date].copy()
    test = df.loc[df.index >= split_date].copy()

    if train.empty or test.empty:
        raise ValueError("Train or test set is empty. Check split_date.")

    return train, test


def fit_scaler(train_X: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_X)
    return scaler


def apply_scaler(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    X_scaled = scaler.transform(X)
    return pd.DataFrame(X_scaled, index=X.index, columns=X.columns)