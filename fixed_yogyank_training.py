"""
Yogyank Entitlement Score - Fixed Baseline Training Script
Audited and rewritten for: leakage control, reproducibility,
model/policy separation, preprocessing hygiene, and explainability.

Run:
    python fixed_yogyank_training.py

Outputs (in artifacts/):
    xgboost_baseline.pkl    - trained model pipeline
    feature_list.json       - features and their availability assumptions
    schema_contract.json    - input schema with expected dtypes and ranges
    version_metadata.json   - run provenance
    validation_summary.json - held-out validation metrics
"""

import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor


warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration – single place for all tunables
# ---------------------------------------------------------------------------
DATA_PATH = "farmer_scoring_sample_yogyank_round1.csv"
ARTIFACTS_DIR = Path("artifacts")
RANDOM_SEED = 42
VALIDATION_YEAR = 2024          # hold out the most-recent year for temporal validation
TEST_FRACTION = 0.20            # fraction of pre-2024 data kept as a random test set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature registry – explicit availability assumptions
# ---------------------------------------------------------------------------
# ISSUE FIXED: original script silently included features without documenting
# whether they exist before the scoring date. We separate safe features from
# leaky ones and explain reasoning for each.

FEATURE_REGISTRY = {
    "land_area_acres": {
        "type": "numeric",
        "available_before_scoring": True,
        "note": "Land record (khasra/patta) – pre-existing at application time.",
    },
    "crop_type": {
        "type": "categorical",
        "available_before_scoring": True,
        "note": "Declared by farmer at application; confirmed via mandi/FPO records.",
    },
    "pm_kisan_status": {
        "type": "categorical",
        "available_before_scoring": True,
        "note": "PM Kisan beneficiary flag from govt DB – queryable before disbursal.",
        "CAUTION": "May update during the loan tenure; use status as-of application date only.",
    },
    "historical_repayment_score": {
        "type": "numeric",
        "available_before_scoring": True,
        "note": "Credit bureau / cooperative history – computed before application.",
    },
    "irrigation_type": {
        "type": "categorical",
        "available_before_scoring": True,
        "note": "Land record attribute – available pre-scoring.",
    },
    "land_ownership": {
        "type": "categorical",
        "available_before_scoring": True,
        "note": "Land record – available pre-scoring.",
    },
    "annual_income_inr": {
        "type": "numeric",
        "available_before_scoring": True,
        "note": "Self-declared / PMFBY estimate. May require verification.",
    },
    "liability_ratio_pct": {
        "type": "numeric",
        "available_before_scoring": True,
        "note": "Existing loan obligations – credit bureau.",
    },
    "rainfall_deviation_pct": {
        "type": "numeric",
        "available_before_scoring": False,
        "note": (
            "ASSUMPTION: Season-end rainfall deviation is UNKNOWN at scoring time "
            "for the current season. Using prior-season value if available is an "
            "approximation. EXCLUDED from model to avoid leakage."
        ),
    },
    "ndvi_score": {
        "type": "numeric",
        "available_before_scoring": False,
        "note": (
            "ASSUMPTION: NDVI from satellite imagery at harvest is UNKNOWN at "
            "loan-application time. EXCLUDED from model to avoid leakage."
        ),
    },
    "defaulted_in_next_12_months": {
        "type": "target_proxy",
        "available_before_scoring": False,
        "note": (
            "CRITICAL LEAKAGE – this is a future outcome label. "
            "Including it as a feature would allow the model to directly 'see' "
            "whether the farmer defaulted, producing artificially perfect metrics "
            "that vanish at deployment. EXCLUDED."
        ),
    },
}

SAFE_NUMERIC_FEATURES = [
    "land_area_acres",
    "historical_repayment_score",
    "annual_income_inr",
    "liability_ratio_pct",
]

SAFE_CATEGORICAL_FEATURES = [
    "crop_type",
    "pm_kisan_status",
    "irrigation_type",
    "land_ownership",
]

ALL_SAFE_FEATURES = SAFE_NUMERIC_FEATURES + SAFE_CATEGORICAL_FEATURES
TARGET = "target_entitlement_score"
SPLIT_YEAR_COL = "application_year"


# ---------------------------------------------------------------------------
# Policy rules – SEPARATED from the model
# ---------------------------------------------------------------------------
# ISSUE FIXED: the original script mutated the target BEFORE train/test split,
# contaminating the test set with a policy rule and making the validation
# metric reflect post-adjustment scores rather than raw model output.
# Policy adjustments must be applied AFTER the model predicts, never on labels.

def apply_business_policy(raw_scores: pd.Series, df: pd.DataFrame) -> pd.Series:
    """
    Apply post-prediction business rules to raw model scores.

    This function is intentionally separate from the model pipeline so that:
    - Policy changes don't require retraining
    - Audit logs can record both raw and adjusted scores
    - Reason codes can be generated per rule

    Args:
        raw_scores: model predictions (pd.Series or np.ndarray)
        df: original feature dataframe aligned to raw_scores

    Returns:
        adjusted_scores: pd.Series with policy applied
    """
    scores = raw_scores.copy()

    # Rule 1: PM Kisan non-beneficiaries receive a downward adjustment.
    # NOTE: The -150 deduction from the original script is preserved here.
    # HOWEVER, this rule should be reviewed by a domain expert:
    #   (a) Is -150 calibrated against the actual score range (~421–980)?
    #   (b) Should this rule be a hard deduction or a soft penalty via feature weight?
    # For now we apply it transparently and flag it for review.
    pm_kisan_mask = df["pm_kisan_status"] == "No"
    scores[pm_kisan_mask.values] -= 150

    return scores


def generate_reason_codes(df: pd.DataFrame, raw_scores: pd.Series) -> pd.DataFrame:
    """
    Generate stable, human-readable reason codes for each score.
    Reason codes explain the top factors; they are rule-based and stable
    (not dependent on SHAP or model internals changing between runs).
    """
    reasons = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        codes = []

        if row["historical_repayment_score"] < 45:
            codes.append("LOW_REPAYMENT_HISTORY")
        if row["liability_ratio_pct"] > 60:
            codes.append("HIGH_LIABILITY_RATIO")
        if row["land_area_acres"] < 2:
            codes.append("SMALL_LAND_HOLDING")
        if row["pm_kisan_status"] == "No":
            codes.append("NOT_PM_KISAN_BENEFICIARY")
        if row["annual_income_inr"] < 100000:
            codes.append("LOW_ANNUAL_INCOME")

        reasons.append("; ".join(codes) if codes else "NO_ADVERSE_FACTORS")

    return pd.Series(reasons, name="reason_codes")


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------

def load_and_validate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info("Loaded %d rows × %d cols from %s", len(df), len(df.columns), path)

    required_cols = ALL_SAFE_FEATURES + [TARGET, SPLIT_YEAR_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Report missingness
    miss = df[ALL_SAFE_FEATURES].isnull().sum()
    miss = miss[miss > 0]
    if not miss.empty:
        log.warning("Missingness in features:\n%s", miss.to_string())

    return df


# ---------------------------------------------------------------------------
# Temporal train/test split
# ---------------------------------------------------------------------------
# ISSUE FIXED: the original script used random shuffle split, which lets
# future observations leak into training. A scoring engine should be
# validated on data it has never 'seen' – ideally a later time period.
# We hold out the most-recent year entirely as the temporal test set.

def temporal_split(df: pd.DataFrame):
    """
    Split strategy:
    - TEMPORAL TEST  : application_year == VALIDATION_YEAR
      Simulates scoring new farmers arriving after the model was trained.
    - TRAIN + RANDOM TEST: application_year < VALIDATION_YEAR
      Further split 80/20 randomly for standard cross-validation.

    Limitation: with only 3 years of data, temporal test may have
    distribution shift from policy/macro changes in 2024, not just
    model generalization gap. Do not over-interpret the temporal score.
    """
    temporal_test = df[df[SPLIT_YEAR_COL] == VALIDATION_YEAR].copy()
    historical = df[df[SPLIT_YEAR_COL] < VALIDATION_YEAR].copy()

    log.info(
        "Temporal split: %d historical rows (years < %d), %d temporal test rows (year == %d)",
        len(historical), VALIDATION_YEAR, len(temporal_test), VALIDATION_YEAR,
    )

    if len(temporal_test) == 0:
        raise ValueError(
            f"No rows for validation year {VALIDATION_YEAR}. "
            "Check VALIDATION_YEAR or the dataset."
        )

    # Further random split within historical for a holdout
    from sklearn.model_selection import train_test_split
    X_hist = historical[ALL_SAFE_FEATURES]
    y_hist = historical[TARGET]
    X_train, X_rand_test, y_train, y_rand_test = train_test_split(
        X_hist, y_hist,
        test_size=TEST_FRACTION,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    X_temporal = temporal_test[ALL_SAFE_FEATURES]
    y_temporal = temporal_test[TARGET]

    return X_train, y_train, X_rand_test, y_rand_test, X_temporal, y_temporal, temporal_test


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------
# ISSUE FIXED: original script reused the same LabelEncoder instance for
# two different columns, causing the second fit_transform to silently
# overwrite the first encoder's vocabulary. This makes the saved model
# unusable for inference (the encoder state is wrong).
# Fix: use sklearn ColumnTransformer so each column has its own encoder,
# and the entire preprocessing → model pipeline is a single serializable object.

def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                SAFE_CATEGORICAL_FEATURES,
            ),
        ],
        remainder="passthrough",   # numeric features pass through unchanged
    )

    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=1,
        tree_method="hist",
        eval_metric="rmse",
    )

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

def save_artifacts(pipeline, validation_summary: dict, df_full: pd.DataFrame):
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # Model pipeline
    model_path = ARTIFACTS_DIR / "xgboost_baseline.pkl"
    joblib.dump(pipeline, model_path)
    log.info("Model pipeline saved to %s", model_path)

    # Feature list with availability metadata
    feature_list = {
        "safe_features_used": ALL_SAFE_FEATURES,
        "excluded_features": {
            k: v["note"]
            for k, v in FEATURE_REGISTRY.items()
            if not v.get("available_before_scoring", True)
        },
        "feature_metadata": FEATURE_REGISTRY,
    }
    with open(ARTIFACTS_DIR / "feature_list.json", "w") as f:
        json.dump(feature_list, f, indent=2)

    # Schema contract
    schema = {
        "input_features": {
            feat: {
                "dtype": str(df_full[feat].dtype),
                "sample_values": df_full[feat].dropna().unique()[:5].tolist()
                if feat in SAFE_CATEGORICAL_FEATURES
                else None,
                "range": [
                    round(float(df_full[feat].min()), 3),
                    round(float(df_full[feat].max()), 3),
                ]
                if feat in SAFE_NUMERIC_FEATURES
                else None,
            }
            for feat in ALL_SAFE_FEATURES
        },
        "target": TARGET,
        "target_range_observed": [
            round(float(df_full[TARGET].min()), 3),
            round(float(df_full[TARGET].max()), 3),
        ],
    }
    with open(ARTIFACTS_DIR / "schema_contract.json", "w") as f:
        json.dump(schema, f, indent=2)

    # Version metadata
    metadata = {
        "training_script": "fixed_yogyank_training.py",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "validation_year": VALIDATION_YEAR,
        "features": ALL_SAFE_FEATURES,
        "target": TARGET,
        "model_class": "XGBRegressor",
        "pipeline_steps": ["OrdinalEncoder (ColumnTransformer)", "XGBRegressor"],
    }
    with open(ARTIFACTS_DIR / "version_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Validation summary
    with open(ARTIFACTS_DIR / "validation_summary.json", "w") as f:
        json.dump(validation_summary, f, indent=2)

    log.info("All artifacts saved to %s/", ARTIFACTS_DIR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train():
    log.info("=== Yogyank Baseline Training (Fixed) ===")

    df = load_and_validate(DATA_PATH)

    (
        X_train, y_train,
        X_rand_test, y_rand_test,
        X_temporal, y_temporal,
        temporal_test_df,
    ) = temporal_split(df)

    pipeline = build_pipeline()

    log.info("Training on %d samples…", len(X_train))
    pipeline.fit(X_train, y_train)

    # Evaluate – random holdout (within historical years)
    preds_rand = pipeline.predict(X_rand_test)
    r2_rand = r2_score(y_rand_test, preds_rand)
    mae_rand = mean_absolute_error(y_rand_test, preds_rand)

    # Evaluate – temporal holdout (unseen future year)
    preds_temporal_raw = pipeline.predict(X_temporal)
    preds_temporal_adjusted = apply_business_policy(
        pd.Series(preds_temporal_raw), temporal_test_df.reset_index(drop=True)
    )
    r2_temporal = r2_score(y_temporal, preds_temporal_raw)
    mae_temporal = mean_absolute_error(y_temporal, preds_temporal_raw)

    log.info("--- Validation Results ---")
    log.info("Random holdout  R2=%.4f  MAE=%.2f", r2_rand, mae_rand)
    log.info("Temporal holdout R2=%.4f  MAE=%.2f  (year=%d)", r2_temporal, mae_temporal, VALIDATION_YEAR)

    validation_summary = {
        "random_holdout": {
            "n": len(y_rand_test),
            "r2": round(r2_rand, 4),
            "mae": round(mae_rand, 4),
            "note": "Random split within historical years. May be optimistic.",
        },
        "temporal_holdout": {
            "n": len(y_temporal),
            "year": VALIDATION_YEAR,
            "r2": round(r2_temporal, 4),
            "mae": round(mae_temporal, 4),
            "note": (
                "Holdout on the most-recent year. Better simulation of deployment. "
                "Trust this metric more than the random holdout. "
                "Limitation: single-year holdout; vulnerable to year-specific effects."
            ),
        },
        "policy_note": (
            "Validation metrics are computed on RAW model predictions (before policy adjustment). "
            "Policy rules (PM Kisan deduction) are applied post-prediction in production, "
            "not during training."
        ),
    }

    save_artifacts(pipeline, validation_summary, df)

    log.info("=== Training complete ===")
    return pipeline, validation_summary


if __name__ == "__main__":
    train()
