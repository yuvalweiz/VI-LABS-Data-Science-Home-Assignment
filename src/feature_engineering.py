"""Feature engineering module for churn prediction."""

import numpy as np
import pandas as pd
import re
from urllib.parse import urlparse

from .config import OBS_START, OBS_END_EXCL, BIG_RECENCY



def safe_div(num, den):
    den = den.replace(0, np.nan)
    return (num / den).fillna(0.0)

def entropy_from_counts(row_counts):
    x = np.asarray(row_counts, dtype=float)
    s = x.sum()
    if s <= 0:
        return 0.0
    p = x / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def gini_from_counts(row_counts):
    x = np.asarray(row_counts, dtype=float)
    x = x[x >= 0]
    if x.size == 0:
        return 0.0
    s = x.sum()
    if s <= 0:
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    index = np.arange(1, n + 1)
    g = (2 * (index * x_sorted).sum()) / (n * s) - (n + 1) / n
    return float(g)

def url_struct_features(url_series):
    """
    Parse URL into lightweight structural features.
    Returns DataFrame with:
      url_domain, url_path_depth, url_len
    """
    u = url_series.fillna("").astype(str)

    domains, depths, lens = [], [], []
    for s in u.values:
        lens.append(len(s))
        try:
            p = urlparse(s)
            dom = (p.netloc or "").lower()
            path = (p.path or "").strip("/")
            depth = 0 if path == "" else path.count("/") + 1
        except Exception:
            dom, depth = "", 0
        domains.append(dom)
        depths.append(depth)

    return pd.DataFrame(
        {"url_domain": domains, "url_path_depth": depths, "url_len": lens},
        index=url_series.index
    )

def _title_hits_per_member(dfw, title_col, keywords):
    """
    Counts visits per member_id where title contains any of the keywords.
    """
    if len(keywords) == 0:
        return pd.Series(dtype=float)
    pat = "|".join([re.escape(k) for k in keywords])
    mask = dfw[title_col].str.contains(pat, case=False, regex=True)
    return dfw.loc[mask].groupby("member_id").size()

# -------------------------
# Main builder
# -------------------------
def build_features(df_churn, df_app, df_web, df_claims, web_mode="counts"):
    """
    Returns one row per member_id with engineered features.
    web_mode:
      - "agg_only"
      - "counts"       : per-title counts
      - "ratios"       : per-title ratios
      - "counts+conc"  : counts + concentration/diversity + URL structure + (conc features exist)
    """
    # ============================================================
    # 1) Base labels
    # ============================================================
    feature_df = df_churn.copy()
    feature_df["tenure_days_at_obs_start"] = (OBS_START - feature_df["signup_date"]).dt.days.clip(lower=0)

    # ============================================================
    # 2) CLAIMS FEATURES (upgraded)
    # ============================================================
    dfc = df_claims.copy()
    dfc = dfc.drop_duplicates(subset=["member_id", "icd_code", "diagnosis_date"])

    claims_agg = dfc.groupby("member_id").agg(
        total_claims=("icd_code", "size"),
        unique_icd_count=("icd_code", "nunique"),
        first_claim_date=("diagnosis_date", "min"),
        last_claim_date=("diagnosis_date", "max"),
        active_claim_days=("diagnosis_date", "nunique"),
    ).reset_index()

    priority = ["E11.9", "I10", "Z71.3"]
    priority_map = {c: c.replace(".", "_") for c in priority}

    priority_flags = (
        dfc.loc[dfc["icd_code"].isin(priority), ["member_id", "icd_code"]]
           .drop_duplicates(["member_id", "icd_code"])
           .assign(val=1)
           .pivot_table(index="member_id", columns="icd_code", values="val", fill_value=0)
           .reset_index()
    )
    for code in priority:
        if code not in priority_flags.columns:
            priority_flags[code] = 0
    priority_flags = priority_flags.rename(columns={code: f"has_{priority_map[code]}" for code in priority})

    claims_feat = claims_agg.merge(priority_flags, on="member_id", how="left")
    claims_feat["has_any_claims"] = 1

    # comorbidity score (used also later for cohorts + uplift proxy)
    claims_feat["comorbidity_score"] = (
        claims_feat["has_E11_9"].fillna(0)
        + claims_feat["has_I10"].fillna(0)
        + claims_feat["has_Z71_3"].fillna(0)
    )

    claims_feat["days_since_last_claim"]  = (OBS_END_EXCL - claims_feat["last_claim_date"]).dt.days
    claims_feat["days_since_first_claim"] = (OBS_END_EXCL - claims_feat["first_claim_date"]).dt.days

    for w in [30, 90, 365]:
        cutoff = OBS_END_EXCL - pd.Timedelta(days=w)
        recent = dfc.loc[dfc["diagnosis_date"] >= cutoff].groupby("member_id").size()
        claims_feat[f"claims_last_{w}d"] = claims_feat["member_id"].map(recent).fillna(0)

    dfc["icd_clean"] = dfc["icd_code"].astype(str).fillna("")
    dfc["icd_family_1"] = dfc["icd_clean"].str[:1]
    dfc["icd_family_3"] = dfc["icd_clean"].str.replace(".", "", regex=False).str[:3]
    # --- severity buckets by ICD prefix (customize list) ---
    dfc["icd_nodot"] = dfc["icd_clean"].str.replace(".", "", regex=False)

    severity_groups = {
        "icd_M54_back_pain": ["M54"],   # back pain
        "icd_E11_diabetes": ["E11"],    # diabetes
        "icd_I10_htn": ["I10"],         # hypertension
        "icd_K21_gerd": ["K21"],        # reflux
        "icd_Z713_diet": ["Z713"],      # dietary counseling (Z71.3 without dot)
    }

    for feat, prefixes in severity_groups.items():
        mask = False
        for p in prefixes:
            mask = mask | dfc["icd_nodot"].str.startswith(p)
        counts = dfc.loc[mask].groupby("member_id").size()
        claims_feat[feat] = claims_feat["member_id"].map(counts).fillna(0)

    # a simple weighted severity score (tweak weights)
    claims_feat["claims_severity"] = (
        2.0 * claims_feat["icd_E11_diabetes"]
        + 1.5 * claims_feat["icd_I10_htn"]
        + 1.0 * claims_feat["icd_M54_back_pain"]
        + 0.7 * claims_feat["icd_K21_gerd"]
        + 0.5 * claims_feat["icd_Z713_diet"]
    )

    fam1 = dfc.groupby("member_id")["icd_family_1"].nunique().rename("unique_icd_family1")
    fam3 = dfc.groupby("member_id")["icd_family_3"].nunique().rename("unique_icd_family3")
    claims_feat = claims_feat.merge(fam1.reset_index(), on="member_id", how="left")
    claims_feat = claims_feat.merge(fam3.reset_index(), on="member_id", how="left")

    feature_df = feature_df.merge(claims_feat, on="member_id", how="left")

    fill0 = [
        "has_any_claims","total_claims","unique_icd_count","active_claim_days","comorbidity_score",
        "has_E11_9","has_I10","has_Z71_3",
        "claims_last_30d","claims_last_90d","claims_last_365d",
        "unique_icd_family1","unique_icd_family3",
    ]
    for c in fill0:
        feature_df[c] = feature_df[c].fillna(0)

    for c in ["days_since_last_claim","days_since_first_claim"]:
        feature_df[c] = feature_df[c].fillna(BIG_RECENCY)

    # ============================================================
    # 3) APP FEATURES
    # ============================================================
    dfa = df_app.copy()
    dfa["date"] = dfa["timestamp"].dt.date
    dfa["hour"] = dfa["timestamp"].dt.hour

    app_agg = dfa.groupby("member_id").agg(
        total_app_sessions=("timestamp", "size"),
        app_active_days=("date", "nunique"),
        last_app_ts=("timestamp", "max"),
    ).reset_index()

    night_hours = {22, 23, 0, 1, 2, 3, 4, 5}
    night_counts = (
        dfa.loc[dfa["hour"].isin(night_hours)]
           .groupby("member_id").size()
           .rename("app_night_sessions").reset_index()
    )

    split_point = OBS_START + pd.Timedelta(days=7)
    early_counts = dfa.loc[dfa["timestamp"] < split_point].groupby("member_id").size().rename("app_early_sessions")
    late_counts  = dfa.loc[dfa["timestamp"] >= split_point].groupby("member_id").size().rename("app_late_sessions")
    early_late = pd.concat([early_counts, late_counts], axis=1).fillna(0).reset_index()
    early_late["app_active_both_periods"] = (
        (early_late["app_early_sessions"] > 0) & (early_late["app_late_sessions"] > 0)
    ).astype(int)

    app_feat = (
        app_agg.merge(night_counts, on="member_id", how="left")
               .merge(early_late, on="member_id", how="left")
    )
    app_feat["has_app_usage"] = 1
    app_feat["app_night_sessions"] = app_feat["app_night_sessions"].fillna(0)
    app_feat["sessions_per_active_day"] = safe_div(app_feat["total_app_sessions"], app_feat["app_active_days"])
    app_feat["days_since_last_app_session"] = (OBS_END_EXCL - app_feat["last_app_ts"]).dt.days

    feature_df = feature_df.merge(app_feat.drop(columns=["last_app_ts"]), on="member_id", how="left")

    app_fill = [
        "has_app_usage","total_app_sessions","app_active_days","app_night_sessions",
        "app_early_sessions","app_late_sessions","app_active_both_periods","sessions_per_active_day"
    ]
    for c in app_fill:
        feature_df[c] = feature_df[c].fillna(0)
    feature_df["days_since_last_app_session"] = feature_df["days_since_last_app_session"].fillna(BIG_RECENCY)

    # ============================================================
    # 4) WEB FEATURES (upgraded + counts+conc implemented)
    # ============================================================
    dfw = df_web.copy()
    dfw["date"] = dfw["timestamp"].dt.date

    title_col = "title"
    dfw[title_col] = dfw[title_col].fillna("").astype(str)

    # URL structure (no one-hot)
    if "url" in dfw.columns:
        url_struct = url_struct_features(dfw["url"])
        dfw = pd.concat([dfw, url_struct], axis=1)
    else:
        dfw["url"] = ""
        dfw["url_domain"] = ""
        dfw["url_path_depth"] = 0
        dfw["url_len"] = 0

    web_agg = dfw.groupby("member_id").agg(
        total_web_visits=("timestamp", "size"),
        web_active_days=("date", "nunique"),
        unique_titles_visited=(title_col, "nunique"),
        unique_urls_visited=("url", "nunique"),
        last_web_ts=("timestamp", "max"),
        avg_url_depth=("url_path_depth", "mean"),
        avg_url_len=("url_len", "mean"),
        unique_domains=("url_domain", "nunique"),
    ).reset_index()

    web_agg["has_web_usage"] = 1
    web_agg["days_since_last_web_visit"] = (OBS_END_EXCL - web_agg["last_web_ts"]).dt.days
    WINDOW_DAYS = (OBS_END_EXCL - OBS_START).days  # should be 15, but your stated window is 14; keep consistent with OBS
    WINDOW_DAYS = max(1, WINDOW_DAYS)

    # App decay
    feature_df["app_visits_per_day"] = feature_df["app_active_days"] / WINDOW_DAYS
    feature_df["app_sessions_per_day"] = feature_df["total_app_sessions"] / WINDOW_DAYS

    # Web decay
    feature_df["web_visits_per_day"] = feature_df.get("web_active_days", 0) / WINDOW_DAYS
    feature_df["web_sessions_per_day"] = feature_df.get("total_web_visits", 0) / WINDOW_DAYS

    # Ramp-down proxy: late minus early (you already computed early/late counts)
    feature_df["app_late_minus_early"] = feature_df.get("app_late_sessions", 0) - feature_df.get("app_early_sessions", 0)
    feature_df["web_health_late_minus_early"] = feature_df.get("web_health_late_minus_early", 0)

    # Title counts (26 columns)
    title_counts = (
        dfw.groupby(["member_id", title_col]).size().unstack(fill_value=0)
    )

    def clean_name(s):
        s = str(s).strip()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^A-Za-z0-9_]+", "", s)
        return s

    title_counts.columns = [f"web_title_cnt__{clean_name(c)}" for c in title_counts.columns]
    title_counts = title_counts.reset_index()

    web_feat = web_agg.merge(title_counts, on="member_id", how="left").fillna(0)

    # Ratios per title
    cnt_cols = [c for c in web_feat.columns if c.startswith("web_title_cnt__")]
    for c in cnt_cols:
        rname = c.replace("web_title_cnt__", "web_title_ratio__")
        web_feat[rname] = safe_div(web_feat[c], web_feat["total_web_visits"])

    # Concentration/diversity (ONLY useful in counts+conc)
    if len(cnt_cols) > 0:
        mat = web_feat[cnt_cols].values
        web_feat["web_title_entropy"] = [entropy_from_counts(row) for row in mat]
        K = len(cnt_cols)
        web_feat["web_title_entropy_norm"] = safe_div(
            web_feat["web_title_entropy"], pd.Series([np.log(K)] * len(web_feat))
        )

        row_sums = mat.sum(axis=1)
        row_sums = np.where(row_sums <= 0, 1.0, row_sums)
        sorted_mat = np.sort(mat, axis=1)[:, ::-1]
        web_feat["web_title_top1_share"] = sorted_mat[:, 0] / row_sums
        web_feat["web_title_top3_share"] = (
            sorted_mat[:, :3].sum(axis=1) / row_sums if K >= 3 else web_feat["web_title_top1_share"]
        )
        web_feat["web_title_gini"] = [gini_from_counts(row) for row in mat]
    else:
        web_feat["web_title_entropy"] = 0.0
        web_feat["web_title_entropy_norm"] = 0.0
        web_feat["web_title_top1_share"] = 0.0
        web_feat["web_title_top3_share"] = 0.0
        web_feat["web_title_gini"] = 0.0

    # Health vs non-health
    health_title_keywords = [
        "diet","nutrition","fiber","meal","meals",
        "exercise","movement","cardio","strength",
        "sleep","insomnia",
        "stress","meditation","resilience",
        "diabetes","hypertension","blood","cholesterol","hba1c","cardiometabolic",
        "healthy","weight"
    ]
    pat_health = "|".join([re.escape(k) for k in health_title_keywords])
    is_health = dfw[title_col].str.contains(pat_health, case=False, regex=True)
    health_counts = dfw.loc[is_health].groupby("member_id").size().rename("web_health_visits")

    web_feat["web_health_visits"] = web_feat["member_id"].map(health_counts).fillna(0)
    web_feat["web_other_visits"] = web_feat["total_web_visits"] - web_feat["web_health_visits"]
    web_feat["web_health_ratio"] = safe_div(web_feat["web_health_visits"], web_feat["total_web_visits"])
    web_feat["web_other_ratio"]  = safe_div(web_feat["web_other_visits"], web_feat["total_web_visits"])

    # Early vs late health shift
    split_point = OBS_START + pd.Timedelta(days=7)
    early = dfw.loc[dfw["timestamp"] < split_point]
    late  = dfw.loc[dfw["timestamp"] >= split_point]

    early_health = (
        early.loc[early[title_col].str.contains(pat_health, case=False, regex=True)]
             .groupby("member_id").size().rename("web_health_early")
    )
    late_health  = (
        late.loc[late[title_col].str.contains(pat_health, case=False, regex=True)]
            .groupby("member_id").size().rename("web_health_late")
    )

    web_feat = web_feat.merge(early_health.reset_index(), on="member_id", how="left")
    web_feat = web_feat.merge(late_health.reset_index(), on="member_id", how="left")
    web_feat["web_health_early"] = web_feat["web_health_early"].fillna(0)
    web_feat["web_health_late"]  = web_feat["web_health_late"].fillna(0)
    web_feat["web_health_late_minus_early"] = web_feat["web_health_late"] - web_feat["web_health_early"]

    # Select which web columns to keep
    base_web_cols = [
        "member_id","total_web_visits","web_active_days","unique_titles_visited","unique_urls_visited",
        "unique_domains","avg_url_depth","avg_url_len",
        "has_web_usage","days_since_last_web_visit",
        "web_health_visits","web_other_visits","web_health_ratio","web_other_ratio",
        "web_health_early","web_health_late","web_health_late_minus_early",
    ]

    ratio_cols = [c for c in web_feat.columns if c.startswith("web_title_ratio__")]
    conc_cols = ["web_title_entropy","web_title_entropy_norm","web_title_top1_share","web_title_top3_share","web_title_gini"]

    if web_mode == "agg_only":
        keep = base_web_cols
    elif web_mode == "counts":
        keep = base_web_cols + cnt_cols
    elif web_mode == "ratios":
        keep = base_web_cols + ratio_cols
    elif web_mode == "counts+conc":
        keep = base_web_cols + cnt_cols + conc_cols
    else:
        raise ValueError(f"Unknown web_mode={web_mode}")

    web_keep = web_feat[keep].copy()
    feature_df = feature_df.merge(web_keep, on="member_id", how="left")

    # Fill web missing for no-web members
    if "has_web_usage" in feature_df.columns:
        feature_df["has_web_usage"] = feature_df["has_web_usage"].fillna(0)
    if "days_since_last_web_visit" in feature_df.columns:
        feature_df["days_since_last_web_visit"] = feature_df["days_since_last_web_visit"].fillna(BIG_RECENCY)

    # Fill numeric columns produced by web with 0
    drop_like = {"member_id","signup_date","churn","outreach"}
    for c in feature_df.columns:
        if c in drop_like:
            continue
        if np.issubdtype(feature_df[c].dtype, np.number):
            feature_df[c] = feature_df[c].fillna(0)

    # ============================================================
    # 5) CROSS-CHANNEL
    # ============================================================
    feature_df["total_digital_interactions"] = feature_df["total_app_sessions"] + feature_df.get("total_web_visits", 0)
    feature_df["channels_used"] = feature_df["has_any_claims"] + feature_df["has_app_usage"] + feature_df.get("has_web_usage", 0)
    feature_df["days_since_last_digital"] = feature_df[["days_since_last_app_session","days_since_last_web_visit"]].min(axis=1)

    feature_df["digital_nonuser"] = (feature_df["total_digital_interactions"] == 0).astype(int)
    feature_df["recently_disengaged"] = (feature_df["days_since_last_digital"] >= 11).astype(int)

    # ============================================================
    # 6) NEW COMPOSITE FEATURES (your table)
    # ============================================================

    # ---- clinical_cohort: Any(Z71/I10/E11) -> 0/1/2+ (multi-bin)
    # uses comorbidity_score (already exists)
    feature_df["clinical_cohort"] = np.clip(feature_df["comorbidity_score"].astype(int), 0, 2)

    # ---- rfm_norm:
    # (1/(1+recency_app_web)) * (sessions+visits+claims)/tenure
    # recency_app_web ~ days_since_last_digital (already computed)
    rec = feature_df["days_since_last_digital"].replace([np.inf, -np.inf], np.nan).fillna(BIG_RECENCY).astype(float)
    tenure = feature_df["tenure_days_at_obs_start"].clip(lower=1).astype(float)
    volume = (
        feature_df.get("total_app_sessions", 0).astype(float)
        + feature_df.get("total_web_visits", 0).astype(float)
        + feature_df.get("total_claims", 0).astype(float)
    )
    feature_df["rfm_norm"] = (1.0 / (1.0 + rec)) * (volume / tenure)

    # ---- risk_uplift_proxy:
    # (e11 + i10 + z71) * health_ratio  ~ comorbidity_score * web_health_ratio
    feature_df["risk_uplift_proxy"] = feature_df["comorbidity_score"].astype(float) * feature_df.get("web_health_ratio", 0).astype(float)
    # signup recency relative to OBS_END
    feature_df["days_since_signup"] = (OBS_END_EXCL - feature_df["signup_date"]).dt.days.clip(lower=0)

    # ---- wellco_engagement:
    # cardiometab + nutrition + movement + sleep + resilience page hits
    # We approximate via keyword groups on titles; counts are computed from raw dfw.
    cardiometab_kw = ["diabetes","hba1c","hypertension","blood","cholesterol","cardio","cardiometabolic","heart"]
    nutrition_kw   = ["diet","nutrition","fiber","meal","meals","protein","carb","calorie","calories"]
    movement_kw    = ["exercise","movement","strength","workout","training","walk","walking","run","running"]
    sleep_kw       = ["sleep","insomnia","bedtime"]
    resilience_kw  = ["stress","meditation","resilience","mindfulness","anxiety","relax"]

    cardiometab_hits = dfw.pipe(_title_hits_per_member, title_col=title_col, keywords=cardiometab_kw)
    nutrition_hits   = dfw.pipe(_title_hits_per_member, title_col=title_col, keywords=nutrition_kw)
    movement_hits    = dfw.pipe(_title_hits_per_member, title_col=title_col, keywords=movement_kw)
    sleep_hits       = dfw.pipe(_title_hits_per_member, title_col=title_col, keywords=sleep_kw)
    resilience_hits  = dfw.pipe(_title_hits_per_member, title_col=title_col, keywords=resilience_kw)

    for name, ser in [
        ("web_cardiometab_visits", cardiometab_hits),
        ("web_nutrition_visits", nutrition_hits),
        ("web_movement_visits", movement_hits),
        ("web_sleep_visits", sleep_hits),
        ("web_resilience_visits", resilience_hits),
    ]:
        feature_df[name] = feature_df["member_id"].map(ser).fillna(0)

    feature_df["wellco_engagement"] = (
        feature_df["web_cardiometab_visits"]
        + feature_df["web_nutrition_visits"]
        + feature_df["web_movement_visits"]
        + feature_df["web_sleep_visits"]
        + feature_df["web_resilience_visits"]
    )

    return feature_df
