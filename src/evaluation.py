"""Evaluation helpers: MSE, Bias², Variance decomposition, bootstrap CI.

Week 8 — unified benchmark for DM vs IPS vs SNIPS.
"""

from __future__ import annotations

import numpy as np


def bootstrap_estimates(
    estimator_fn,
    n_bootstrap: int = 200,
    random_state: int = 42,
    **kwargs,
) -> np.ndarray:
    """Run an estimator on n_bootstrap resamples of the data.

    Args:
        estimator_fn: callable(**kwargs) -> float
        n_bootstrap:  number of bootstrap resamples
        random_state: seed
        **kwargs:     arrays to resample (must share first axis = n_rounds)

    Returns:
        (n_bootstrap,) array of bootstrap estimates
    """
    rng = np.random.default_rng(random_state)
    # Determine n from the first array kwarg
    n = next(len(v) for v in kwargs.values() if hasattr(v, "__len__"))
    estimates = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        resampled = {k: v[idx] if hasattr(v, "__getitem__") else v for k, v in kwargs.items()}
        estimates.append(estimator_fn(**resampled))
    return np.array(estimates)


def bias_variance_mse(
    estimates: np.ndarray,
    ground_truth: float,
) -> dict[str, float]:
    """Decompose MSE = Bias² + Variance.

    Args:
        estimates:    (n_bootstrap,) bootstrap estimates of policy value
        ground_truth: true policy value V*

    Returns:
        dict with keys: mean, bias, bias2, variance, mse, rmse, std, ci_lower, ci_upper
    """
    mean_est = float(np.mean(estimates))
    bias = mean_est - ground_truth
    bias2 = bias ** 2
    variance = float(np.var(estimates, ddof=1))
    mse = float(np.mean((estimates - ground_truth) ** 2))
    return {
        "mean": mean_est,
        "bias": bias,
        "bias2": bias2,
        "variance": variance,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "std": float(np.std(estimates, ddof=1)),
        "ci_lower": float(np.percentile(estimates, 2.5)),
        "ci_upper": float(np.percentile(estimates, 97.5)),
        "ci_width": float(np.percentile(estimates, 97.5) - np.percentile(estimates, 2.5)),
    }


def unified_benchmark(
    reward: np.ndarray,
    pscore_log: np.ndarray,
    expected_reward: np.ndarray,
    action: np.ndarray,
    pi_eval: float,
    ground_truth: float,
    n_bootstrap: int = 200,
    random_state: int = 42,
) -> dict[str, dict]:
    """Run DM, IPS, SNIPS bootstrap decomposition and return unified results.

    Args:
        reward:           (n,) observed rewards
        pscore_log:       (n,) P(a|s) under logging policy
        expected_reward:  (n,) reward model predictions f(s, a_logged)
        action:           (n,) logged actions
        pi_eval:          scalar evaluation policy probability
        ground_truth:     V* — true policy value for bias calculation
        n_bootstrap:      number of bootstrap resamples
        random_state:     seed

    Returns:
        dict keyed by estimator name, each containing bias_variance_mse result
    """
    from src.estimators import ips_with_clipping, snips_estimate

    def dm_fn(reward, expected_reward, **_):
        return float(np.mean(expected_reward))

    def ips_fn(reward, pscore_log, **_):
        return ips_with_clipping(reward, pscore_log, pi_eval)

    def snips_fn(reward, pscore_log, **_):
        return snips_estimate(reward, pscore_log, pi_eval)

    data = dict(reward=reward, pscore_log=pscore_log, expected_reward=expected_reward, action=action)

    results = {}
    for name, fn in [("DM", dm_fn), ("IPS", ips_fn), ("SNIPS", snips_fn)]:
        boots = bootstrap_estimates(fn, n_bootstrap=n_bootstrap, random_state=random_state, **data)
        results[name] = bias_variance_mse(boots, ground_truth)
        results[name]["bootstrap_samples"] = boots

    return results
