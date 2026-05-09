
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def fit_gaussian_hmm(
    X: pd.DataFrame,
    n_states: int = 3,
    covariance_type: str = "diag",
    n_iter: int = 200,
    n_inits: int = 10,
    random_seed: int = 42,
) -> GaussianHMM:
    """
    Fit Gaussian HMM with multiple random initializations.

    Keeps the model with the best training log-likelihood.
    """
    best_model = None
    best_score = -np.inf

    for i in range(n_inits):
        seed = random_seed + i

        model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=1e-4,
            random_state=seed,
            verbose=False,
        )

        model.fit(X.values)
        score = model.score(X.values)

        if score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        raise RuntimeError("HMM training failed.")

    return best_model


def summarize_states(df: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    """
    Summarize economic properties of each inferred state.
    """
    temp = df.copy()
    temp["regime"] = regimes

    summary = temp.groupby("regime").agg(
        mean_daily_return=("log_return", "mean"),
        annualized_return=("log_return", lambda x: x.mean() * 252),
        daily_vol=("log_return", "std"),
        annualized_vol=("log_return", lambda x: x.std() * np.sqrt(252)),
        mean_rolling_vol=("rolling_vol", "mean"),
        mean_momentum=("momentum", "mean"),
        count=("log_return", "count"),
    )

    summary["economic_label_hint"] = summary.apply(_label_hint, axis=1)
    return summary


def _label_hint(row: pd.Series) -> str:
    """
    Heuristic label helper.
    You can manually adjust labels in the report after seeing results.
    """
    if row["annualized_vol"] > 0.25 and row["annualized_return"] < 0:
        return "Risk-off / stress"
    if row["mean_momentum"] > 0 and row["annualized_return"] > 0:
        return "Risk-on / trending"
    return "Sideways / low drift"


def order_states_by_volatility(df: pd.DataFrame, raw_regimes: np.ndarray) -> dict[int, int]:
    """
    Create stable state labels by sorting raw states by realized rolling volatility.

    Lowest-vol state -> 0
    Middle-vol state -> 1
    Highest-vol state -> 2
    """
    temp = df.copy()
    temp["raw_regime"] = raw_regimes

    vol_by_state = temp.groupby("raw_regime")["rolling_vol"].mean().sort_values()
    return {raw_state: ordered_state for ordered_state, raw_state in enumerate(vol_by_state.index)}
