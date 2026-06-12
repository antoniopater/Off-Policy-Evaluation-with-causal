"""Model propensity P(a|s) i diagnostyka overlap."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

OBD_COMMON_CONTEXT_DIM = 20


def align_context(context: np.ndarray, n_features: int = OBD_COMMON_CONTEXT_DIM) -> np.ndarray:
    """Wspólne 20 cech kontekstu (random 20-d, bts 22-d)."""
    if context.shape[1] < n_features:
        raise ValueError(
            f"Context has {context.shape[1]} features, expected at least {n_features}"
        )
    return np.ascontiguousarray(context[:, :n_features], dtype=np.float32)


def train_propensity_model(
    context: np.ndarray,
    action: np.ndarray,
    n_actions: int = 80,
    random_state: int = 42,
    n_features: int = OBD_COMMON_CONTEXT_DIM,
) -> XGBClassifier:
    """XGBoost multiclass — P(a|s)."""
    X = align_context(context, n_features)
    y = np.asarray(action, dtype=np.int32)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
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
        tree_method="hist",
        n_jobs=1,
        random_state=random_state,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def extract_propensity_scores(
    model: XGBClassifier,
    context: np.ndarray,
    action: np.ndarray,
    n_features: int = OBD_COMMON_CONTEXT_DIM,
) -> np.ndarray:
    """P(a_i | s_i) dla zaobserwowanych par."""
    X = align_context(context, n_features)
    y = np.asarray(action, dtype=np.int32)
    proba = model.predict_proba(X)
    col_idx = np.searchsorted(model.classes_, y)
    if not np.all(model.classes_[col_idx] == y):
        raise ValueError("Action labels contain classes not seen during training")
    return proba[np.arange(len(y)), col_idx]


def propensity_calibration_curve(
    model: XGBClassifier,
    context: np.ndarray,
    action: np.ndarray,
    n_bins: int = 10,
    n_features: int = OBD_COMMON_CONTEXT_DIM,
) -> tuple[np.ndarray, np.ndarray]:
    """Krzywa kalibracji top-1 akcji."""
    from sklearn.calibration import calibration_curve

    X = align_context(context, n_features)
    y = np.asarray(action, dtype=np.int32)
    proba = model.predict_proba(X)
    pred = model.classes_[np.argmax(proba, axis=1)]
    confidence = proba.max(axis=1)
    hit = (pred == y).astype(np.int32)
    return calibration_curve(hit, confidence, n_bins=n_bins, strategy="quantile")


def effective_sample_size(weights: np.ndarray) -> tuple[float, float]:
    """ESS i ESS/n."""
    w = np.asarray(weights, dtype=np.float64)
    ess = float(w.sum() ** 2) / float((w**2).sum())
    return ess, ess / len(w)


def overlap_diagnostics(
    pscores: np.ndarray,
    action: np.ndarray,
    pi_eval: float,
    n_actions: int = 80,
) -> dict:
    """Statystyki overlap: pscore, wagi IPS, ESS."""
    pscores = np.asarray(pscores, dtype=np.float64)
    action = np.asarray(action, dtype=np.int32)
    weights = pi_eval / np.clip(pscores, 1e-9, None)
    ess, ess_ratio = effective_sample_size(weights)

    min_per_action = np.array(
        [
            pscores[action == a].min() if (action == a).any() else np.nan
            for a in range(n_actions)
        ]
    )

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
