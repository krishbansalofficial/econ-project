
from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import sharpe_ratio


def stationary_bootstrap_indices(
    n: int,
    avg_block: int = 20,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Stationary bootstrap index generator.

    Each next observation either continues the current block or starts a new block.
    Expected block length is avg_block.
    """
    if rng is None:
        rng = np.random.default_rng()

    p = 1.0 / avg_block
    indices = np.empty(n, dtype=int)

    indices[0] = rng.integers(0, n)

    for t in range(1, n):
        if rng.random() < p:
            indices[t] = rng.integers(0, n)
        else:
            indices[t] = (indices[t - 1] + 1) % n

    return indices


def sharpe_diff_bootstrap(
    strategy_returns: pd.Series,
    baseline_returns: pd.Series,
    n_boot: int = 2000,
    avg_block: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    """
    Bootstrap confidence interval for Sharpe difference.

    Difference = Sharpe(strategy) - Sharpe(baseline)
    """
    paired = pd.concat([strategy_returns, baseline_returns], axis=1).dropna()
    paired.columns = ["strategy", "baseline"]

    n = len(paired)
    rng = np.random.default_rng(seed)

    observed = sharpe_ratio(paired["strategy"]) - sharpe_ratio(paired["baseline"])

    boot_diffs = np.empty(n_boot)

    values = paired.values

    for b in range(n_boot):
        idx = stationary_bootstrap_indices(n, avg_block=avg_block, rng=rng)
        sample = values[idx]

        s = pd.Series(sample[:, 0])
        base = pd.Series(sample[:, 1])

        boot_diffs[b] = sharpe_ratio(s) - sharpe_ratio(base)

    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

    # Two-sided p-value based on bootstrap distribution centered around observed.
    centered = boot_diffs - boot_diffs.mean()
    p_value = np.mean(np.abs(centered) >= abs(observed))

    return {
        "observed_sharpe_difference": float(observed),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value_approx": float(p_value),
    }
