from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _savefig(path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_price_with_regimes(df: pd.DataFrame, regimes: pd.Series, path: str | Path):
    fig, ax = plt.subplots(figsize=(14, 6))

    scatter = ax.scatter(
        df.index,
        df["close"],
        c=regimes,
        cmap="viridis",
        s=12,
        label="Regime",
    )

    ax.plot(df.index, df["close"], linewidth=1.1, label="SPY Price")

    economic_events = {
        "COVID Crash": "2020-03-01",
        "Fed Hikes Begin": "2022-03-16",
        "Inflation Shock": "2022-06-10",
        "Regional Bank Stress": "2023-03-10",
        "AI Rally": "2023-01-01",
    }

    for label, date in economic_events.items():
        event_date = pd.to_datetime(date)

        if df.index.min() <= event_date <= df.index.max():
            ax.axvline(event_date, linestyle="--", linewidth=1)
            ax.text(
                event_date,
                df["close"].max(),
                label,
                rotation=90,
                verticalalignment="top",
                fontsize=8,
            )

    ax.set_title("SPY Price with Regimes and Major Market Events")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close Price")
    ax.legend(loc="upper left")

    plt.colorbar(scatter, ax=ax, label="Regime / Cluster")
    _savefig(path)


def plot_regime_probabilities(probs: pd.DataFrame, index, path: str | Path):
    plt.figure(figsize=(12, 6))
    plt.stackplot(index, probs.T.values, labels=probs.columns)
    plt.title("Filtered HMM Regime Probabilities")
    plt.xlabel("Date")
    plt.ylabel("Probability")
    plt.legend(loc="upper left")
    _savefig(path)


def plot_equity_curves(equity: pd.DataFrame, path: str | Path):
    plt.figure(figsize=(12, 6))

    for col in equity.columns:
        plt.plot(equity.index, equity[col], label=col, linewidth=1.2)

    plt.title("Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    _savefig(path)


def plot_drawdowns(drawdowns: pd.DataFrame, path: str | Path):
    plt.figure(figsize=(12, 6))

    for col in drawdowns.columns:
        plt.plot(drawdowns.index, drawdowns[col], label=col, linewidth=1.2)

    plt.title("Drawdowns")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    _savefig(path)


def plot_transition_matrix(transmat: np.ndarray, path: str | Path):
    plt.figure(figsize=(6, 5))
    plt.imshow(transmat)
    plt.title("HMM Transition Matrix")
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.colorbar(label="Probability")

    for i in range(transmat.shape[0]):
        for j in range(transmat.shape[1]):
            plt.text(
                j,
                i,
                f"{transmat[i, j]:.2f}",
                ha="center",
                va="center",
            )

    _savefig(path)


def plot_returns_by_regime(df: pd.DataFrame, regimes: pd.Series, path: str | Path):
    temp = pd.DataFrame(
        {
            "return": df["log_return"],
            "regime": regimes,
        }
    ).dropna()

    unique_regimes = sorted(temp["regime"].unique())
    groups = [
        temp.loc[temp["regime"] == r, "return"].values
        for r in unique_regimes
    ]

    plt.figure(figsize=(8, 6))
    plt.boxplot(groups, tick_labels=[f"Regime {r}" for r in unique_regimes])
    plt.title("Daily Returns by HMM Regime")
    plt.xlabel("Regime")
    plt.ylabel("Log Return")
    _savefig(path)


def plot_yearly_regimes(yearly_df: pd.DataFrame, path: str | Path):
    fig, ax = plt.subplots(figsize=(10, 5))

    yearly_df.plot(kind="bar", ax=ax)

    ax.set_title("Dominant Regime / Cluster by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Dominant Label")
    ax.legend(["HMM Regime", "K-Means Cluster"])

    plt.xticks(rotation=0)
    _savefig(path)