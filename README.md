# Yogyank Entitlement Score — Baseline Submission

## Timing
| | |
|---|---|
| Start time (IST) | *(record your start time here)* |
| End time (IST) | *(record your end time here)* |
| Approximate time spent | ~90 minutes |

---

## Setup

```bash
pip install pandas scikit-learn xgboost joblib
```

Python 3.9+ recommended.

---

## How to Run

Place the dataset in the same directory as the script:

```
fixed_yogyank_training.py
farmer_scoring_sample_yogyank_round1.csv
```

Then run:

```bash
python fixed_yogyank_training.py
```

---

## Files Generated

All outputs go into `artifacts/`:

| File | Description |
|---|---|
| `xgboost_baseline.pkl` | Trained sklearn Pipeline (preprocessor + XGBRegressor) |
| `feature_list.json` | Features used, excluded, and availability assumptions |
| `schema_contract.json` | Input schema: dtypes, ranges, valid category values |
| `version_metadata.json` | Run provenance: timestamp, seed, model class |
| `validation_summary.json` | R² and MAE for random and temporal holdouts |

---

## What Was Completed

- [x] Full audit of `broken_yogyank_training.py` (see `audit_memo.md`)
- [x] Fixed training script with all critical issues addressed
- [x] Temporal validation split (hold out 2024 as unseen year)
- [x] Model/policy separation (`apply_business_policy()` post-prediction)
- [x] Correct preprocessing pipeline (ColumnTransformer, serialized with model)
- [x] Feature registry with availability assumptions
- [x] Reason code generator (rule-based, stable across retrains)
- [x] Artifacts folder with model, schema, metadata, and validation summary
- [x] audit_memo.md
- [x] LLM_NOTES.md

## What Was Skipped (Due to Time)

- Walk-forward cross-validation across all three years
- SHAP-based feature importance plots (deprioritized; rule-based reason codes are more stable)
- Unit tests for preprocessing and policy functions
- Hyperparameter tuning (baseline kept deliberately simple)
- Detailed investigation of PM Kisan −150 deduction calibration

---

## Assumptions

### Feature Availability at Scoring Time

| Feature | Used? | Assumption |
|---|---|---|
| `land_area_acres` | ✅ Yes | Land records (khasra/patta) exist before loan application |
| `crop_type` | ✅ Yes | Declared at application; verifiable from mandi/FPO records |
| `pm_kisan_status` | ✅ Yes | Queryable from govt DB at application time (use as-of snapshot) |
| `historical_repayment_score` | ✅ Yes | Credit bureau / cooperative history — pre-application |
| `irrigation_type` | ✅ Yes | Land record attribute — pre-application |
| `land_ownership` | ✅ Yes | Land record — pre-application |
| `annual_income_inr` | ✅ Yes | Self-declared / PMFBY estimate — available at application |
| `liability_ratio_pct` | ✅ Yes | Existing loan obligations from credit bureau |
| `rainfall_deviation_pct` | ❌ Excluded | Season-end metric — unknown at loan time for current season |
| `ndvi_score` | ❌ Excluded | Satellite harvest score — unknown at loan time |
| `defaulted_in_next_12_months` | ❌ Excluded (leakage!) | **Future outcome — must never be used as a feature** |

### Other Assumptions

- `application_year` column reliably represents when the scoring decision was made.
- `target_entitlement_score` in the dataset represents the label *before* any policy adjustment (i.e., it is a pure model target). The PM Kisan −150 policy is therefore applied post-prediction only.
- The 5,000-row dataset is a synthetic sample; real production data may have different distributions, especially for categorical features with regional variation (districts, sales channels).

---

## Validation Approach

**Strategy:** Temporal holdout — hold out all rows with `application_year == 2024` (1,394 rows) as the unseen test set. Train on `application_year < 2024` (3,606 rows, further split 80/20 randomly for a secondary holdout).

**Why temporal, not random:** A scoring engine always operates on *future* applicants. A random split allows future data into training, which is optimistic and misleading. The temporal split better simulates deployment.

**Results:**

| Split | N | R² | MAE |
|---|---|---|---|
| Random holdout (within 2022–2023) | 722 | 0.697 | 47.4 |
| **Temporal holdout (2024)** | **1,394** | **0.687** | **48.4** |

**Do I trust this result?** Cautiously. The small gap between random and temporal R² suggests the model is not severely overfit to historical years. However, with only one year of temporal data, a single anomalous year (drought, policy shift, macro shock) could distort either direction. I would require at least 2–3 years of temporal holdouts before trusting this for production.

**I do not trust:** the PM Kisan −150 deduction as a validated, calibrated business rule. It is preserved in `apply_business_policy()` but flagged for domain expert review.
