"""Feature selection and importance analysis."""

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance

from catboost import CatBoostClassifier

from .config import SEED, CAT_PARAMS


def get_XYT(df: pd.DataFrame):
    """Extract features, target, and treatment from dataframe."""
    drop_cols = ["member_id", "signup_date", "churn", "outreach"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()
    y = df["churn"].astype(int).values
    t = df["outreach"].astype(int).values

    # Drop datetime-like from X
    dt_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.datetime64)]
    if dt_cols:
        X = X.drop(columns=dt_cols)

    # bool -> int
    for c in X.columns:
        if X[c].dtype == "bool":
            X[c] = X[c].astype(int)

    # fill missing (consistent)
    cat_cols = [c for c in X.columns if (X[c].dtype == "object") or str(X[c].dtype).startswith("category")]
    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in cat_cols:
        X[c] = X[c].astype("object").fillna("NA")
    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].fillna(0)

    return X, y, t


def cat_feature_indices(X):
    """Get indices of categorical features in dataframe."""
    cat_cols = [c for c in X.columns if (X[c].dtype == "object") or str(X[c].dtype).startswith("category")]
    return [X.columns.get_loc(c) for c in cat_cols]


def fit_catboost(X_tr, y_tr, X_va, y_va, seed):
    """Train a CatBoost classifier with the configured parameters."""
    params = CAT_PARAMS.copy()
    params["random_seed"] = seed
    model = CatBoostClassifier(**params)
    cat_idx = cat_feature_indices(X_tr)
    model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        cat_features=cat_idx if len(cat_idx) > 0 else None
    )
    return model


def dr_tau_saved(mu0, mu1, e, X, y, t):
    """
    Compute doubly robust uplift estimates (tau_saved = -tau_ate).
    
    Args:
        mu0: Model for E[Y|X,T=0]
        mu1: Model for E[Y|X,T=1]
        e: Propensity model for P(T=1|X)
        X: Features
        y: Outcomes
        t: Treatment indicators
        
    Returns:
        tuple: (tau_saved, mu0_pred, mu1_pred, propensity_pred)
    """
    mu0p = np.clip(mu0.predict_proba(X)[:, 1], 1e-5, 1 - 1e-5)
    mu1p = np.clip(mu1.predict_proba(X)[:, 1], 1e-5, 1 - 1e-5)
    ep   = np.clip(e.predict_proba(X)[:, 1],   1e-3, 1 - 1e-3)
    tau_ate = (mu1p - mu0p) + (t * (y - mu1p) / ep) - ((1 - t) * (y - mu0p) / (1 - ep))
    tau_saved = -tau_ate
    return tau_saved, mu0p, mu1p, ep


def perm_importance(model, X, y, scoring="roc_auc", n_repeats=5, seed=None):
    """Compute permutation importance for model features."""
    if seed is None:
        seed = SEED
    r = permutation_importance(
        model, X, y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1
    )
    imp = pd.DataFrame({
        "feature": X.columns,
        "perm_mean": r.importances_mean,
        "perm_std": r.importances_std,
    }).sort_values("perm_mean", ascending=False).reset_index(drop=True)
    return imp


def add_rank(df, col, name):
    """Add a rank column based on specified column values."""
    d = df[["feature", col]].copy()
    d[name] = d[col].rank(ascending=False, method="average")
    return d[["feature", name]]
