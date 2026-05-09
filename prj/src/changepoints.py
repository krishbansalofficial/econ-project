
from __future__ import annotations

import numpy as np
import pandas as pd


def run_pelt_changepoints(series: pd.Series, penalty: float = 5.0) -> dict:
    """
    Run PELT change point detection on a univariate series.

    Uses rolling volatility by default in main.py.
    """
    import ruptures as rpt

    clean = series.dropna()
    signal = clean.values.reshape(-1, 1)

    algo = rpt.Pelt(model="rbf").fit(signal)
    breakpoints = algo.predict(pen=penalty)

    segment_id = pd.Series(index=clean.index, dtype=int)

    start = 0
    for seg, end in enumerate(breakpoints):
        segment_id.iloc[start:end] = seg
        start = end

    return {
        "breakpoints": breakpoints,
        "segment_id": segment_id,
    }


def changepoint_positions(df: pd.DataFrame, segment_id: pd.Series) -> pd.Series:
    """
    Convert change point segments into simple defensive positions.

    For each segment, compute average rolling volatility and momentum.
    - High-vol and negative-momentum segments -> cash
    - Otherwise -> long
    """
    aligned = df.copy()
    aligned["segment_id"] = segment_id.reindex(df.index).ffill().bfill()

    segment_stats = aligned.groupby("segment_id").agg(
        mean_vol=("rolling_vol", "mean"),
        mean_momentum=("momentum", "mean"),
    )

    vol_cutoff = segment_stats["mean_vol"].quantile(0.75)

    segment_position = {}
    for seg, row in segment_stats.iterrows():
        if row["mean_vol"] >= vol_cutoff and row["mean_momentum"] < 0:
            segment_position[seg] = 0.0
        else:
            segment_position[seg] = 1.0

    pos = aligned["segment_id"].map(segment_position).astype(float)
    return pos.shift(1).fillna(0.0)
