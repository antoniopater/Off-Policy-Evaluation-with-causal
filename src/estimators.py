"""OPE estimator wrappers — Weeks 2, 6, 9."""

import numpy as np
from obp.ope import (
    InverseProbabilityWeighting,
    SelfNormalizedInverseProbabilityWeighting,
    OffPolicyEvaluation,
)


def ips_with_clipping(
    reward: np.ndarray,
    pscore_log: np.ndarray,
    pi_eval: float,
    clip_lambda: float | None = None,
) -> float:
    """IPS estimator with optional propensity clipping.

    Args:
        reward:      (n_rounds,) binary rewards
        pscore_log:  (n_rounds,) P(a|s) under logging policy
        pi_eval:     scalar P(a|s) under evaluation policy (uniform = 1/n_actions)
        clip_lambda: minimum value to clip pscore_log (None = no clipping)

    Returns:
        Scalar policy value estimate
    """
    denom = np.clip(pscore_log, clip_lambda, None) if clip_lambda is not None else pscore_log
    weights = pi_eval / np.clip(denom, 1e-9, None)
    return float((weights * reward).mean())


def snips_estimate(
    reward: np.ndarray,
    pscore_log: np.ndarray,
    pi_eval: float,
    clip_lambda: float | None = None,
) -> float:
    """Self-Normalized IPS — normalizes weights to reduce variance.

    Args:
        reward:      (n_rounds,) binary rewards
        pscore_log:  (n_rounds,) P(a|s) under logging policy
        pi_eval:     scalar P(a|s) under evaluation policy
        clip_lambda: optional clipping of pscore_log

    Returns:
        Scalar policy value estimate
    """
    denom = np.clip(pscore_log, clip_lambda, None) if clip_lambda is not None else pscore_log
    weights = pi_eval / np.clip(denom, 1e-9, None)
    return float((weights * reward).sum() / weights.sum())


def clipping_experiment(
    reward: np.ndarray,
    pscore_log: np.ndarray,
    pi_eval: float,
    lambdas: list[float | None],
    n_bootstrap: int = 200,
    random_state: int = 42,
) -> dict:
    """Run IPS and SNIPS across a grid of clipping values with bootstrap CI.

    Args:
        reward:       (n_rounds,) binary rewards
        pscore_log:   (n_rounds,) propensity scores under logging policy
        pi_eval:      scalar evaluation policy probability
        lambdas:      list of clipping thresholds (None = no clipping)
        n_bootstrap:  number of bootstrap samples for variance estimate
        random_state: seed

    Returns:
        dict keyed by lambda, each with ips_mean, snips_mean, ips_std, snips_std
    """
    rng = np.random.default_rng(random_state)
    n = len(reward)
    results = {}

    for lam in lambdas:
        ips_boots, snips_boots = [], []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            ips_boots.append(ips_with_clipping(reward[idx], pscore_log[idx], pi_eval, lam))
            snips_boots.append(snips_estimate(reward[idx], pscore_log[idx], pi_eval, lam))

        results[lam] = {
            "ips_mean": ips_with_clipping(reward, pscore_log, pi_eval, lam),
            "snips_mean": snips_estimate(reward, pscore_log, pi_eval, lam),
            "ips_std": float(np.std(ips_boots)),
            "snips_std": float(np.std(snips_boots)),
            "ips_ci_lower": float(np.percentile(ips_boots, 2.5)),
            "ips_ci_upper": float(np.percentile(ips_boots, 97.5)),
            "snips_ci_lower": float(np.percentile(snips_boots, 2.5)),
            "snips_ci_upper": float(np.percentile(snips_boots, 97.5)),
        }

    return results


def build_obp_ips_estimators():
    """Return OBP IPS and SNIPS estimator instances for use with OffPolicyEvaluation."""
    return InverseProbabilityWeighting(), SelfNormalizedInverseProbabilityWeighting()
