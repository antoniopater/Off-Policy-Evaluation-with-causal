"""IPS, SNIPS i eksperyment clippingu."""

import numpy as np
from obp.ope import (
    InverseProbabilityWeighting,
    SelfNormalizedInverseProbabilityWeighting,
)


def ips_with_clipping(
    reward: np.ndarray,
    pscore_log: np.ndarray,
    pi_eval: float,
    clip_lambda: float | None = None,
) -> float:
    """IPS z opcjonalnym clippingiem pscore_log (dolna granica = clip_lambda)."""
    denom = np.clip(pscore_log, clip_lambda, None) if clip_lambda is not None else pscore_log
    weights = pi_eval / np.clip(denom, 1e-9, None)
    return float((weights * reward).mean())


def snips_estimate(
    reward: np.ndarray,
    pscore_log: np.ndarray,
    pi_eval: float,
    clip_lambda: float | None = None,
) -> float:
    """Self-normalized IPS."""
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
    """IPS i SNIPS dla siatki progów clippingu + bootstrap std/CI."""
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
    """Instancje IPS i SNIPS z biblioteki obp."""
    return InverseProbabilityWeighting(), SelfNormalizedInverseProbabilityWeighting()
