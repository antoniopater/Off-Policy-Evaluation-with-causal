"""Propensity score model utilities — Week 5."""

import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


def train_propensity_model(
    context: np.ndarray,
    action: np.ndarray,
    n_actions: int = 80,
    random_state: int = 42,
) -> XGBClassifier:
    """Train P(a|s) — 80-class softmax classifier.

    Args:
        context:      (n_rounds, n_features) user/context features
        action:       (n_rounds,) integer action labels 0..n_actions-1
        n_actions:    size of action space (default 80 for OBD)
        random_state: reproducibility seed

    Returns:
        Fitted XGBClassifier with predict_proba → (n_rounds, n_actions)
    """
    X_train, X_val, y_train, y_val = train_test_split(
        context, action,
        test_size=0.2,
        random_state=random_state,
        stratify=action,
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_actions,
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        random_state=random_state,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def extract_propensity_scores(
    model: XGBClassifier,
    context: np.ndarray,
    action: np.ndarray,
) -> np.ndarray:
    """Return P(a_i | s_i) for each observed (context, action) pair.

    Args:
        model:   trained propensity model
        context: (n_rounds, n_features)
        action:  (n_rounds,) integer action labels

    Returns:
        (n_rounds,) propensity scores in [0, 1]
    """
    proba = model.predict_proba(context)              # (n_rounds, n_actions)
    return proba[np.arange(len(action)), action]      # pick column of observed action


def effective_sample_size(weights: np.ndarray) -> tuple[float, float]:
    """Compute ESS and ESS ratio for importance weights.

    ESS ratio < 0.1 signals critical overlap violation.

    Returns:
        (ess, ess_ratio) tuple
    """
    ess = float(weights.sum() ** 2) / float((weights ** 2).sum())
    return ess, ess / len(weights)


def overlap_diagnostics(
    pscores: np.ndarray,
    action: np.ndarray,
    pi_eval: float,
    n_actions: int = 80,
) -> dict:
    """Compute overlap diagnostics for propensity scores.

    Args:
        pscores:   (n_rounds,) P(a_i|s_i) under logging policy
        action:    (n_rounds,) observed action labels
        pi_eval:   scalar P(a|s) under evaluation policy (uniform = 1/n_actions)
        n_actions: action space size

    Returns:
        dict with summary stats, ESS, and per-action min pscore
    """
    weights = pi_eval / np.clip(pscores, 1e-9, None)
    ess, ess_ratio = effective_sample_size(weights)

    min_per_action = np.array([
        pscores[action == a].min() if (action == a).sum() > 0 else np.nan
        for a in range(n_actions)
    ])

    return {
        "pscore_min": float(pscores.min()),
        "pscore_max": float(pscores.max()),
        "pscore_mean": float(pscores.mean()),
        "pct_below_001": float((pscores < 0.001).mean() * 100),
        "pct_nan": float(np.isnan(pscores).mean() * 100),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "ess": ess,
        "ess_ratio": ess_ratio,
        "min_pscore_per_action": min_per_action,
    }
