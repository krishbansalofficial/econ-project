from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd


def load_market_data(
    symbol: str = "SPY",
    start: str = "2015-01-01",
    end: Optional[str] = None,
    source: str = "yfinance",
) -> pd.DataFrame:
    if source == "yfinance":
        return _load_yfinance(symbol, start, end)

    if source == "stooq":
        return _load_stooq(symbol, start, end)

    raise ValueError("source must be either 'yfinance' or 'stooq'")


def _load_yfinance(symbol: str, start: str, end: Optional[str]) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        actions=False,
    )

    if raw.empty:
        raise ValueError(f"No data returned for {symbol} from yfinance.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    required = ["open", "high", "low", "close", "volume"]
    df = raw[required].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df.dropna()


def _load_stooq(symbol: str, start: str, end: Optional[str]) -> pd.DataFrame:
    from pandas_datareader import data as pdr

    end = end or datetime.today().strftime("%Y-%m-%d")
    raw = pdr.DataReader(symbol, "stooq", start=start, end=end)

    if raw.empty:
        raise ValueError(f"No data returned for {symbol} from Stooq.")

    raw = raw.sort_index()
    raw = raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    required = ["open", "high", "low", "close", "volume"]
    return raw[required].dropna()


def load_vix(start: str = "2015-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """
    Download VIX index data from Yahoo Finance.

    VIX is used as a market fear / implied volatility feature.
    """
    import yfinance as yf

    raw = yf.download(
        "^VIX",
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        actions=False,
    )

    if raw.empty:
        raise ValueError("No VIX data returned from yfinance.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename(columns={"Close": "vix"})
    vix = raw[["vix"]].copy()
    vix.index = pd.to_datetime(vix.index)
    vix = vix.sort_index()

    return vix.dropna()