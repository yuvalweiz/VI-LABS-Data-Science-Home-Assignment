"""Model training and evaluation for uplift modeling."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

import torch
import torch.nn as nn
import torch.optim as optim

from .config import (
    SEED, FOLDS, K_FRACS, K_RATIO_GRID, MIN_GROUP, EPS_GUARD,
    CHURN_GATE_ALPHAS, FEATURE_TOPK_LIST, MODELS_TO_RUN,
    MLP_HIDDEN, MLP_DROPOUT, MLP_EPOCHS, MLP_LR, MLP_BS, MLP_PATIENCE
)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------
# Stable hash for reproducibility
# ---------------------------
def stable_hash_int(s: str, mod: int = 997) -> int:
    """Stable hash function that returns same value across Python sessions."""
    import hashlib
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % mod


# ---------------------------
# Helpers: XYT + preprocessing
# ---------------------------
def get_XYT(df: pd.DataFrame):
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


def make_preprocessor(X_df: pd.DataFrame):
    cat_cols = [c for c in X_df.columns if (X_df[c].dtype == "object") or str(X_df[c].dtype).startswith("category")]
    num_cols = [c for c in X_df.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True))]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop"
    )
    return pre


# ---------------------------
# Feature ranking helpers (uses your existing imp_all)
# ---------------------------
def get_ranked_features_from_imp_all(X_df: pd.DataFrame, imp_all=None):
    """
    Returns a list of feature names ordered best -> worse using imp_all if available.
    Falls back to X_df.columns if imp_all not present.
    
    Args:
        X_df: DataFrame with features
        imp_all: Optional DataFrame with feature importance rankings
    """
    if imp_all is not None and isinstance(imp_all, pd.DataFrame) and ("feature" in imp_all.columns):
        if "rank_mean" in imp_all.columns:
            feats = imp_all.sort_values("rank_mean", ascending=True)["feature"].astype(str).tolist()
        else:
            feats = imp_all["feature"].astype(str).tolist()
        feats = [f for f in feats if f in X_df.columns]
        if len(feats) > 0:
            return feats
    return list(X_df.columns)


def select_topk_X(X_df: pd.DataFrame, ranked_features, K: int):
    cols = ranked_features[:int(K)]
    cols = [c for c in cols if c in X_df.columns]
    if len(cols) == 0:
        cols = list(X_df.columns)
    return X_df[cols].copy(), len(cols)


# ---------------------------
# OOF Propensity ê(x) for IPW (leakage-free)
# ---------------------------
def oof_propensity(X_df: pd.DataFrame, t: np.ndarray, folds=FOLDS, seed=SEED):
    skf_t = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    e_hat = np.zeros(len(t), dtype=float)

    for fold, (tr_idx, te_idx) in enumerate(skf_t.split(np.arange(len(t)), t), start=1):
        X_tr_df = X_df.iloc[tr_idx].copy()
        t_tr = t[tr_idx]
        X_te_df = X_df.iloc[te_idx].copy()

        pre_fold = make_preprocessor(X_tr_df)
        X_tr = pre_fold.fit_transform(X_tr_df)
        X_te = pre_fold.transform(X_te_df)

        e_model = LogisticRegression(
            solver="lbfgs", max_iter=4000, n_jobs=-1,
            class_weight="balanced", C=1.0
        )
        e_model.fit(X_tr, t_tr)
        p = e_model.predict_proba(X_te)[:, 1]
        e_hat[te_idx] = np.clip(p, 1e-3, 1 - 1e-3)

    return e_hat


# ---------------------------
# IPW uplift estimator on subset S
# ---------------------------
def uplift_in_subset_ipw(y, t, e_hat, idx, min_eff=MIN_GROUP):
    tt = t[idx].astype(float)
    yy = y[idx].astype(float)
    ee = e_hat[idx].astype(float)

    w1 = tt / ee
    w0 = (1.0 - tt) / (1.0 - ee)

    n1_eff = float(w1.sum())
    n0_eff = float(w0.sum())
    if (n1_eff < min_eff) or (n0_eff < min_eff):
        return 0.0, n0_eff, n1_eff

    y1 = float((w1 * yy).sum() / max(1e-9, w1.sum()))
    y0 = float((w0 * yy).sum() / max(1e-9, w0.sum()))
    return float(y0 - y1), n0_eff, n1_eff


# ---------------------------
# Qini/AUQC (IPW)
# ---------------------------
def qini_curve_and_auqc_ipw(y, t, e_hat, score, n_bins=100):
    N = len(y)
    order = np.argsort(-score)
    xs = np.linspace(0.0, 1.0, n_bins + 1)

    overall_uplift, _, _ = uplift_in_subset_ipw(y, t, e_hat, np.arange(N))
    total_saved_overall = overall_uplift * N

    model_curve = []
    rand_curve = []

    for x in xs:
        m = max(1, int(round(x * N)))
        idx = order[:m]
        u, _, _ = uplift_in_subset_ipw(y, t, e_hat, idx)
        model_curve.append(u * m)
        rand_curve.append(total_saved_overall * x)

    model_curve = np.array(model_curve, dtype=float)
    rand_curve  = np.array(rand_curve, dtype=float)

    area_model = np.trapz(model_curve, xs)
    area_rand  = np.trapz(rand_curve, xs)
    auqc_raw = float(area_model - area_rand)

    scale = max(1e-9, abs(total_saved_overall))
    auqc = float(auqc_raw / scale)

    return xs, model_curve, rand_curve, auqc, auqc_raw, overall_uplift


def uplift_at_frac_ipw(y, t, e_hat, score, frac):
    N = len(y)
    m = max(1, int(round(frac * N)))
    idx = np.argsort(-score)[:m]
    u, n0e, n1e = uplift_in_subset_ipw(y, t, e_hat, idx)
    return {"k_frac": frac, "k_n": m, "uplift": u, "n0_eff": n0e, "n1_eff": n1e, "idx": idx}


# ---------------------------
# n selection helpers
# ---------------------------
def n_by_ratio_threshold(score, ratio_k):
    return int(np.sum(score > ratio_k))

def n_default_positive(score, eps=0.0):
    return int(np.sum(score > eps))


# ---------------------------
# (Optional) group counts for robustness reporting
# ---------------------------
def group_counts(y, t, idx):
    yy = y[idx]
    tt = t[idx]
    out = {}
    for ti in [0, 1]:
        for yi in [0, 1]:
            out[f"n_t{ti}y{yi}"] = int(np.sum((tt == ti) & (yy == yi)))
    return out


def uplift_k_report_ipw(y, t, e_hat, score, fracs=(0.10, 0.30)):
    rows = []
    for f in fracs:
        r = uplift_at_frac_ipw(y, t, e_hat, score, frac=float(f))
        gc = group_counts(y, t, r["idx"])
        rows.append({
            "K%": int(round(100 * f)),
            "n": int(r["k_n"]),
            "uplift": float(r["uplift"]),
            "saved": float(r["uplift"] * r["k_n"]),
            "n0_eff": float(r["n0_eff"]),
            "n1_eff": float(r["n1_eff"]),
            **gc
        })
    return pd.DataFrame(rows)


# ---------------------------
# Models: T-learner wrappers (sklearn)
# ---------------------------
def fit_t_learner_sklearn(model_factory, X_tr, y_tr, t_tr):
    m0 = model_factory()
    m1 = model_factory()
    m0.fit(X_tr[t_tr == 0], y_tr[t_tr == 0])
    m1.fit(X_tr[t_tr == 1], y_tr[t_tr == 1])
    return m0, m1

def predict_uplift_tlearner(m0, m1, X):
    p0 = m0.predict_proba(X)[:, 1]
    p1 = m1.predict_proba(X)[:, 1]
    return (p0 - p1), p0, p1


# ---------------------------
# Deep model: Two-Head MLP (T-learner in one net)
# ---------------------------
class TwoHeadMLP(nn.Module):
    def __init__(self, d_in, d_hidden=256, dropout=0.25):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head0 = nn.Linear(d_hidden, 1)
        self.head1 = nn.Linear(d_hidden, 1)

    def forward(self, x):
        z = self.trunk(x)
        return self.head0(z).squeeze(-1), self.head1(z).squeeze(-1)


def _safe_stratified_split_for_mlp(y_tr, t_tr, seed, test_size=0.2):
    strat = (y_tr * 2 + t_tr)
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr2, va2 = next(sss.split(np.arange(len(y_tr)), strat))
        return tr2, va2
    except Exception:
        # fallback: random split
        rng = np.random.default_rng(seed)
        idx = np.arange(len(y_tr))
        rng.shuffle(idx)
        m = int(round((1 - test_size) * len(idx)))
        return idx[:m], idx[m:]


def train_twohead_mlp_arrays(X_tr, y_tr, t_tr, X_va, y_va, t_va,
                            epochs=MLP_EPOCHS, lr=MLP_LR, batch_size=MLP_BS,
                            d_hidden=MLP_HIDDEN, dropout=MLP_DROPOUT, patience=MLP_PATIENCE,
                            seed=SEED):
    torch.manual_seed(seed)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    t_tr_t = torch.tensor(t_tr, dtype=torch.float32, device=DEVICE)

    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)
    y_va_t = torch.tensor(y_va, dtype=torch.float32, device=DEVICE)
    t_va_t = torch.tensor(t_va, dtype=torch.float32, device=DEVICE)

    model = TwoHeadMLP(X_tr.shape[1], d_hidden=d_hidden, dropout=dropout).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    pos = max(1.0, float(y_tr_t.sum().item()))
    neg = max(1.0, float((1.0 - y_tr_t).sum().item()))
    pos_weight = torch.tensor([neg / pos], device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = 1e18
    best_state = None
    bad = 0

    n = X_tr.shape[0]
    idx = np.arange(n)

    for ep in range(1, epochs + 1):
        np.random.shuffle(idx)
        model.train()
        for s in range(0, n, batch_size):
            j = idx[s:s + batch_size]
            xb = X_tr_t[j]
            yb = y_tr_t[j]
            tb = t_tr_t[j]

            mu0_logit, mu1_logit = model(xb)
            factual_logit = torch.where(tb > 0.5, mu1_logit, mu0_logit)
            loss = bce(factual_logit, yb)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            mu0v, mu1v = model(X_va_t)
            factual_v = torch.where(t_va_t > 0.5, mu1v, mu0v)
            val_loss = bce(factual_v, y_va_t).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    model.load_state_dict(best_state)
    return model


def predict_uplift_twohead_mlp(model, X):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        mu0_logit, mu1_logit = model(X_t)
        p0 = torch.sigmoid(mu0_logit).detach().cpu().numpy()
        p1 = torch.sigmoid(mu1_logit).detach().cpu().numpy()
    return (p0 - p1), p0, p1


# ---------------------------
# Boost factory (uplift model B)
# ---------------------------
def get_boost_factory(prefer_gpu=True, seed=SEED):
    try:
        import xgboost as xgb
        def xgb_factory():
            use_gpu = prefer_gpu and torch.cuda.is_available()
            return xgb.XGBClassifier(
                n_estimators=350,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=2.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method=("gpu_hist" if use_gpu else "hist"),
                random_state=seed,
                n_jobs=-1
            )
        return "xgboost", xgb_factory
    except Exception:
        pass

    def hgb_factory():
        return HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=600,
            random_state=seed
        )
    return "histgb", hgb_factory


# ---------------------------
# Score variants: tau-based + weak baselines
# ---------------------------
def build_score_variants(base_tau, base_p0, alphas,
                        add_risk_baseline=True,
                        add_random_baseline=True,
                        seed=SEED):
    """
    Returns list of (variant_name, score).

    Main policy variants:
      - BASE: tau
      - CHURN_INTENSITY=a: tau * (p0 ** a)

    Weak baselines (sanity):
      - RISK_p0 : rank by churn risk only (p0)
      - RAND    : random ranking
    """
    p0 = np.clip(base_p0, 1e-6, 1 - 1e-6)

    variants = [("BASE", base_tau.copy())]
    for a in alphas:
        score = base_tau * np.power(p0, float(a))
        variants.append((f"CHURN_INTENSITY={a}", score))

    if add_risk_baseline:
        variants.append(("RISK_p0", p0.copy()))

    if add_random_baseline:
        rng = np.random.default_rng(int(seed))
        variants.append(("RAND", rng.standard_normal(len(base_tau)).astype(float)))

    return variants


# ---------------------------
# OOF training + predictions (accepts X_df directly)
# ---------------------------
def run_oof_uplift_models_X(X_df, y, t, models_to_run=None, seed=SEED):
    """
    Returns:
      e_hat, oof_tau, oof_p0, oof_p1
    """
    if models_to_run is None:
        models_to_run = MODELS_TO_RUN

    strat = (y * 2 + t)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)

    print("\n[Propensity] computing OOF ê(x) ...")
    e_hat = oof_propensity(X_df, t, folds=FOLDS, seed=seed)
    print("[Propensity] ê(x) mean:", float(e_hat.mean()), "min/max:", float(e_hat.min()), float(e_hat.max()))

    oof_tau = {m: np.zeros(len(y), dtype=float) for m in models_to_run}
    oof_p0  = {m: np.zeros(len(y), dtype=float) for m in models_to_run}
    oof_p1  = {m: np.zeros(len(y), dtype=float) for m in models_to_run}

    boost_backend, _ = get_boost_factory(prefer_gpu=True, seed=seed)
    print(f"[Boost] backend: {boost_backend}")

    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.arange(len(y)), strat), start=1):
        print("\n" + "="*80)
        print(f"FOLD {fold}/{FOLDS} | train={len(tr_idx)} test={len(te_idx)}")
        print("="*80)

        X_tr_df = X_df.iloc[tr_idx].copy()
        y_tr = y[tr_idx]
        t_tr = t[tr_idx]
        X_te_df = X_df.iloc[te_idx].copy()

        pre_fold = make_preprocessor(X_tr_df)
        X_tr = pre_fold.fit_transform(X_tr_df)
        X_te = pre_fold.transform(X_te_df)

        # A) Logistic T-learner
        if "A_logreg_tlearner" in models_to_run:
            def lr_factory():
                return LogisticRegression(
                    solver="lbfgs",
                    max_iter=5000,
                    n_jobs=-1,
                    class_weight="balanced",
                    C=0.7
                )
            m0, m1 = fit_t_learner_sklearn(lr_factory, X_tr, y_tr, t_tr)
            tau, p0, p1 = predict_uplift_tlearner(m0, m1, X_te)
            oof_tau["A_logreg_tlearner"][te_idx] = tau
            oof_p0["A_logreg_tlearner"][te_idx]  = p0
            oof_p1["A_logreg_tlearner"][te_idx]  = p1

        # B) Boosted Trees T-learner
        if "B_boost_tlearner" in models_to_run:
            _, bf = get_boost_factory(prefer_gpu=True, seed=seed + 10 * fold)
            b0 = bf()
            b1 = bf()
            b0.fit(X_tr[t_tr == 0], y_tr[t_tr == 0])
            b1.fit(X_tr[t_tr == 1], y_tr[t_tr == 1])
            p0 = b0.predict_proba(X_te)[:, 1]
            p1 = b1.predict_proba(X_te)[:, 1]
            tau = p0 - p1
            oof_tau["B_boost_tlearner"][te_idx] = tau
            oof_p0["B_boost_tlearner"][te_idx]  = p0
            oof_p1["B_boost_tlearner"][te_idx]  = p1

        # C) Deep (Two-Head MLP)
        if "C_mlp_twohead" in models_to_run:
            # inner split for early stopping (within fold)
            tr2, va2 = _safe_stratified_split_for_mlp(y_tr, t_tr, seed=seed + 999 + fold, test_size=0.2)

            X_tr2, y_tr2, t_tr2 = X_tr[tr2], y_tr[tr2], t_tr[tr2]
            X_va2, y_va2, t_va2 = X_tr[va2], y_tr[va2], t_tr[va2]

            mlp = train_twohead_mlp_arrays(
                X_tr2, y_tr2, t_tr2,
                X_va2, y_va2, t_va2,
                epochs=MLP_EPOCHS, lr=MLP_LR, batch_size=MLP_BS,
                d_hidden=MLP_HIDDEN, dropout=MLP_DROPOUT, patience=MLP_PATIENCE,
                seed=seed + 1234 + fold
            )
            tau, p0, p1 = predict_uplift_twohead_mlp(mlp, X_te)

            oof_tau["C_mlp_twohead"][te_idx] = tau
            oof_p0["C_mlp_twohead"][te_idx]  = p0
            oof_p1["C_mlp_twohead"][te_idx]  = p1

    return e_hat, oof_tau, oof_p0, oof_p1


def parse_alpha(v):
    if v == "BASE":
        return 0.0
    if v.startswith("CHURN_INTENSITY="):
        return float(v.split("=")[1])
    return np.nan


# ---------------------------
# Plot helpers (winner plots)
# ---------------------------
def plot_qini_with_fill(xs, model_curve, rand_curve, title):
    plt.figure()
    plt.plot(xs, model_curve, linewidth=3, label="Model")
    plt.plot(xs, rand_curve, linestyle="--", alpha=0.7, label="Random")
    plt.fill_between(xs, rand_curve, model_curve, alpha=0.3)
    plt.xlabel("% Targeted")
    plt.ylabel("Cumulative churn saves (IPW-estimated)")
    plt.title(title)
    plt.legend()
    plt.show()
