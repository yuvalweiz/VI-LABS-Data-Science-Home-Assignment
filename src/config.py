"""Configuration constants for VI Labs assignment."""

import pandas as pd

# ============================================================
# Observation Window
# ============================================================
OBS_START = pd.Timestamp("2025-07-01")
OBS_END_INCL = pd.Timestamp("2025-07-15")
OBS_END_EXCL = OBS_END_INCL + pd.Timedelta(days=1)

# ============================================================
# Feature Engineering
# ============================================================
BIG_RECENCY = 999

# ============================================================
# Model Training
# ============================================================
SEED = 42
USE_GPU = True
TASK_TYPE = "GPU" if USE_GPU else "CPU"
FOLDS = 5

# Feature selection
WEB_MODE_FOR_IMPORTANCE = "counts+conc"
HOLDOUT_SIZE = 0.2

# CatBoost parameters
CAT_PARAMS = dict(
    loss_function="Logloss",
    eval_metric="Logloss",
    iterations=3000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    od_type="Iter",
    od_wait=200,
    random_seed=SEED,
    task_type=TASK_TYPE,
    verbose=200,
    use_best_model=True,
)

# ============================================================
# Grid Search Parameters
# ============================================================
# Reporting points for uplift evaluation
K_FRACS = [0.10, 0.20, 0.30, 0.40, 0.50]

# n selection when cost is unknown: k = c/v (break-even uplift threshold)
K_RATIO_GRID = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

# Effective group size safety in IPW estimates
MIN_GROUP = 30

# Optional conservative tau threshold for "positive effect"
EPS_GUARD = 0.005

# Churn-prob "intensity" on p0
CHURN_GATE_ALPHAS = [0.2, 0.6, 1.0, 1.2, 1.5]

# Feature top-K options
FEATURE_TOPK_LIST = [20, 40, 60, 80, 100]

# Models to run
MODELS_TO_RUN = ["A_logreg_tlearner", "B_boost_tlearner", "C_mlp_twohead"]

# ============================================================
# Deep Learning Parameters
# ============================================================
MLP_HIDDEN = 256
MLP_DROPOUT = 0.25
MLP_EPOCHS = 40
MLP_LR = 8e-4
MLP_BS = 256
MLP_PATIENCE = 8
