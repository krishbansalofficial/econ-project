from pathlib import Path

import pandas as pd

from src.clustering import fit_kmeans_clusters, order_clusters_by_volatility, summarize_clusters
from src.data import load_market_data, load_vix
from src.features import make_features, split_train_test, fit_scaler, apply_scaler
from src.hmm_model import fit_gaussian_hmm, summarize_states, order_states_by_volatility
from src.inference import filtered_state_probs
from src.strategy import (
    make_regime_positions,
    make_buy_hold_positions,
    make_momentum_positions,
    make_mean_reversion_positions,
)
from src.backtest import run_backtest
from src.metrics import metrics_table
from src.viz import (
    plot_price_with_regimes,
    plot_regime_probabilities,
    plot_equity_curves,
    plot_drawdowns,
    plot_transition_matrix,
    plot_returns_by_regime,
    plot_yearly_regimes,
)
from src.bootstrap import sharpe_diff_bootstrap
from src.changepoints import run_pelt_changepoints, changepoint_positions


def save_yearly_regime_tables(test_df, test_regime, test_cluster, results_dir):
    hmm_yearly = (
        pd.DataFrame({"regime": test_regime}, index=test_df.index)
        .groupby(test_df.index.year)["regime"]
        .agg(lambda x: x.value_counts().idxmax())
    )

    kmeans_yearly = (
        pd.DataFrame({"cluster": test_cluster}, index=test_df.index)
        .groupby(test_df.index.year)["cluster"]
        .agg(lambda x: x.value_counts().idxmax())
    )

    yearly = pd.DataFrame(
        {
            "dominant_hmm_regime": hmm_yearly,
            "dominant_kmeans_cluster": kmeans_yearly,
        }
    )

    yearly.to_csv(results_dir / "yearly_regime_interpretation.csv")

    print("\nDominant regime / cluster by year:")
    print(yearly)

    return yearly


def main():
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    fig_dir = project_dir / "figures"
    results_dir = project_dir / "results"

    data_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    print("Loading SPY and VIX data...")
    prices = load_market_data(symbol="SPY", start="2015-01-01", source="yfinance")
    vix = load_vix(start="2015-01-01")

    prices.to_csv(data_dir / "spy_prices.csv")
    vix.to_csv(data_dir / "vix.csv")

    print("Building features...")
    df = make_features(
        prices,
        vix_df=vix,
        vol_window=20,
        momentum_window=20,
    )
    df.to_csv(data_dir / "spy_features.csv")

    print("Splitting train/test...")
    train_df, test_df = split_train_test(df, split_date="2021-01-01")

    feature_cols = [
        "log_return",
        "rolling_vol",
        "momentum",
        "vix",
        "volume_z",
        "amihud_liquidity",
    ]

    scaler = fit_scaler(train_df[feature_cols])
    X_train = apply_scaler(train_df[feature_cols], scaler)
    X_test = apply_scaler(test_df[feature_cols], scaler)

    print("Training HMM...")
    model = fit_gaussian_hmm(
        X_train,
        n_states=3,
        covariance_type="diag",
        n_iter=200,
        n_inits=10,
        random_seed=42,
    )

    print("Computing online filtered probabilities...")
    train_probs = filtered_state_probs(model, X_train)
    test_probs = filtered_state_probs(model, X_test)

    train_regime_raw = train_probs.to_numpy().argmax(axis=1)
    test_regime_raw = test_probs.to_numpy().argmax(axis=1)

    state_order = order_states_by_volatility(train_df, train_regime_raw)
    train_regime = pd.Series(train_regime_raw, index=train_df.index).map(state_order)
    test_regime = pd.Series(test_regime_raw, index=test_df.index).map(state_order)

    regime_summary = summarize_states(train_df, train_regime)
    regime_summary.to_csv(results_dir / "regime_summary.csv")

    print("\nHMM regime summary:")
    print(regime_summary)

    print("\nTraining K-Means clustering benchmark...")
    kmeans_model, train_cluster_raw, test_cluster_raw = fit_kmeans_clusters(
        X_train,
        X_test,
        n_clusters=3,
        random_seed=42,
    )

    cluster_order = order_clusters_by_volatility(train_df, train_cluster_raw)
    train_cluster = train_cluster_raw.map(cluster_order)
    test_cluster = test_cluster_raw.map(cluster_order)

    cluster_summary = summarize_clusters(train_df, train_cluster)
    cluster_summary.to_csv(results_dir / "cluster_summary.csv")

    print("\nK-Means cluster summary:")
    print(cluster_summary)

    yearly_regimes = save_yearly_regime_tables(
        test_df,
        test_regime,
        test_cluster,
        results_dir,
    )

    print("Generating strategy positions...")
    regime_position = make_regime_positions(test_df, test_regime)
    cluster_position = make_regime_positions(test_df, test_cluster)

    buy_hold_position = make_buy_hold_positions(test_df)
    momentum_position = make_momentum_positions(test_df)
    mean_reversion_position = make_mean_reversion_positions(test_df)

    returns = test_df["log_return"]

    print("Running backtests...")
    strategies = {
        "Regime-aware HMM": run_backtest(returns, regime_position, cost_bps=5),
        "K-Means Regime Strategy": run_backtest(returns, cluster_position, cost_bps=5),
        "Buy and Hold": run_backtest(returns, buy_hold_position, cost_bps=0),
        "Standalone Momentum": run_backtest(returns, momentum_position, cost_bps=5),
        "Standalone Mean Reversion": run_backtest(returns, mean_reversion_position, cost_bps=5),
    }

    rets = pd.DataFrame({name: out["strategy_return"] for name, out in strategies.items()})
    equity = pd.DataFrame({name: out["equity"] for name, out in strategies.items()})
    drawdowns = pd.DataFrame({name: out["drawdown"] for name, out in strategies.items()})

    rets.to_csv(results_dir / "daily_strategy_returns.csv")
    equity.to_csv(results_dir / "equity_curves.csv")
    drawdowns.to_csv(results_dir / "drawdowns.csv")

    print("Computing metrics...")
    metrics = metrics_table(strategies)
    metrics.to_csv(results_dir / "metrics.csv")

    print("\nPerformance metrics:")
    print(metrics)

    print("Running bootstrap Sharpe difference tests...")
    for baseline in [
        "K-Means Regime Strategy",
        "Buy and Hold",
        "Standalone Momentum",
        "Standalone Mean Reversion",
    ]:
        boot = sharpe_diff_bootstrap(
            rets["Regime-aware HMM"],
            rets[baseline],
            n_boot=1000,
            avg_block=20,
            seed=42,
        )

        print(f"\nSharpe difference: Regime-aware HMM minus {baseline}")
        print(boot)

    print("Creating plots...")
    plot_price_with_regimes(test_df, test_regime, fig_dir / "price_with_hmm_regimes.png")
    plot_price_with_regimes(test_df, test_cluster, fig_dir / "price_with_kmeans_clusters.png")
    plot_regime_probabilities(test_probs, test_df.index, fig_dir / "regime_probabilities.png")
    plot_equity_curves(equity, fig_dir / "equity_curves.png")
    plot_drawdowns(drawdowns, fig_dir / "drawdowns.png")
    plot_transition_matrix(model.transmat_, fig_dir / "transition_matrix.png")
    plot_returns_by_regime(test_df, test_regime, fig_dir / "returns_by_hmm_regime.png")
    plot_yearly_regimes(yearly_regimes, fig_dir / "yearly_regime_interpretation.png")

    print("Running optional change point detection baseline...")
    cp_result = run_pelt_changepoints(test_df["rolling_vol"], penalty=5)
    cp_position = changepoint_positions(test_df, cp_result["segment_id"])
    cp_backtest = run_backtest(returns, cp_position, cost_bps=5)

    cp_backtest["strategy_return"].to_csv(results_dir / "changepoint_strategy_returns.csv")

    cp_metrics = metrics_table({"Change Point Baseline": cp_backtest})
    cp_metrics.to_csv(results_dir / "changepoint_metrics.csv")

    print("\nChange point baseline metrics:")
    print(cp_metrics)

    print("\nDone. Check data/, figures/, and results/.")


if __name__ == "__main__":
    main()