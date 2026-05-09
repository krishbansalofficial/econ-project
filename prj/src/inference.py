
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp


def filtered_state_probs(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute online filtered state probabilities using forward recursion.

    This avoids look-ahead bias. At time t, the probability only uses X_1:t.

    Note:
    hmmlearn exposes emission likelihoods through a private method.
    This is commonly used when a public filtering API is unavailable.
    """
    X_values = X.values

    log_emlik = model._compute_log_likelihood(X_values)
    log_startprob = np.log(model.startprob_ + 1e-300)
    log_transmat = np.log(model.transmat_ + 1e-300)

    n_obs, n_states = log_emlik.shape
    log_alpha = np.zeros((n_obs, n_states))

    # Initial update
    log_alpha[0] = log_startprob + log_emlik[0]
    log_alpha[0] -= logsumexp(log_alpha[0])

    # Recursive filtering update
    for t in range(1, n_obs):
        for j in range(n_states):
            log_alpha[t, j] = log_emlik[t, j] + logsumexp(
                log_alpha[t - 1] + log_transmat[:, j]
            )

        log_alpha[t] -= logsumexp(log_alpha[t])

    probs = np.exp(log_alpha)

    columns = [f"state_{i}" for i in range(n_states)]
    return pd.DataFrame(probs, index=X.index, columns=columns)
