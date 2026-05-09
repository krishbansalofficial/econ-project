from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans


def fit_kmeans_clusters(X_train, X_test, n_clusters=3, random_seed=42):
    """
    Fit K-Means on training data only, then predict clusters for train and test.

    This avoids leakage because the clustering model is learned only from training data.
    """
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_seed,
        n_init=20,
    )

    train_clusters = model.fit_predict(X_train)
    test_clusters = model.predict(X_test)

    train_clusters = pd.Series(train_clusters, index=X_train.index, name="cluster")
    test_clusters = pd.Series(test_clusters, index=X_test.index, name="cluster")

    return model, train_clusters, test_clusters


def order_clusters_by_volatility(df, raw_clusters):
    """
    Re-label clusters so that:
    0 = lowest volatility
    1 = middle volatility
    2 = highest volatility

    This makes cluster labels economically interpretable.
    """
    temp = df.copy()
    temp["raw_cluster"] = raw_clusters

    vol_by_cluster = temp.groupby("raw_cluster")["rolling_vol"].mean().sort_values()

    return {
        raw_cluster: ordered_cluster
        for ordered_cluster, raw_cluster in enumerate(vol_by_cluster.index)
    }


def summarize_clusters(df, clusters):
    """
    Summarize each cluster using return, volatility, and momentum.
    """
    temp = df.copy()
    temp["cluster"] = clusters

    summary = temp.groupby("cluster").agg(
        mean_daily_return=("log_return", "mean"),
        annualized_return=("log_return", lambda x: x.mean() * 252),
        daily_vol=("log_return", "std"),
        annualized_vol=("log_return", lambda x: x.std() * (252 ** 0.5)),
        mean_rolling_vol=("rolling_vol", "mean"),
        mean_momentum=("momentum", "mean"),
        count=("log_return", "count"),
    )

    summary["economic_label_hint"] = summary.apply(_label_hint, axis=1)
    return summary


def _label_hint(row):
    if row["annualized_vol"] > 0.25 and row["annualized_return"] < 0:
        return "Risk-off / stress"
    if row["mean_momentum"] > 0 and row["annualized_return"] > 0:
        return "Risk-on / trending"
    return "Sideways / low drift"