# Yogyank Entitlement Score — Audit Memo
**Script audited:** `broken_yogyank_training.py`  
**Auditor:** Aeasha (ML Engineer candidate)  

---

## 1. What Was Dangerous in the Original Script

### Issue 1 — Critical Data Leakage: `defaulted_in_next_12_months` used as a feature

**Severity: CRITICAL**

The feature list includes `defaulted_in_next_12_months`. This column records whether the farmer defaulted *in the 12 months following* the loan application — a future outcome that is, by definition, unknowable at scoring time.

Feeding this into the model is a direct form of label leakage. The model learns to predict the entitlement score partly by observing who defaulted. At deployment, this column does not exist, so the model receives a fundamentally different input distribution than it trained on. The reported validation R² is untrustworthy because the metric was computed on a test set where this column was still present and informative.

**Why the "Wow!" R² is suspicious:** An R² that looks excellent on a randomly split test set but would collapse at deployment is the canonical signature of leakage. The comment "Model is performing well. Validation score looks good. Ready for production" is exactly the kind of overconfidence that leakage enables.

---

### Issue 2 — Target Mutation Before Split: Policy Rule Applied to Labels

**Severity: HIGH**

```python
df.loc[df["pm_kisan_status"] == "No", "target_entitlement_score"] -= 150
```

This line modifies the target column **before** the train/test split. The effect:

- The test set labels are the already-adjusted scores, so the model is evaluated on the post-policy outcome, not on its raw predictive power.
- If the policy changes (e.g., the deduction becomes -100 or is removed entirely), the model must be retrained because the policy is baked into the labels the model learned.
- Policy and model are entangled — this is an auditing and governance failure, not just a code quality issue.

Business rules must be applied **after** prediction, never by mutating training labels.

---

### Issue 3 — Shared `LabelEncoder` Across Two Columns

**Severity: HIGH**

```python
encoder = LabelEncoder()
X["crop_type"]       = encoder.fit_transform(X["crop_type"])
X["pm_kisan_status"] = encoder.fit_transform(X["pm_kisan_status"])
```

The same `encoder` object is `fit_transform`-ed twice. After the second call, `encoder.classes_` holds the categories for `pm_kisan_status` only — the `crop_type` vocabulary is discarded. If you serialize this encoder and try to decode predictions or handle unseen values during inference, the encoder is simply wrong. The model artifact would be unusable for production inference without the specific in-memory state from that single training run.

---

### Issue 4 — No Preprocessing Saved With the Model

**Severity: HIGH**

Only `model` (the XGBRegressor) is saved:
```python
joblib.dump(model, "xgboost_baseline.pkl")
```

The preprocessing steps (encoding crop_type, pm_kisan_status) are in-memory transformations that are **not persisted**. To score a new farmer, whoever runs inference would need to reconstruct the same encodings, with the same category-to-integer mapping, in the same order. This is unreproducible and will silently produce wrong predictions if the category ordering differs.

---

### Issue 5 — Random Shuffle Split Does Not Simulate Future Scoring

**Severity: MEDIUM**

```python
train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
```

A random split allows future rows (e.g., from 2024) to appear in the training set, while 2022 rows appear in the test set. A scoring engine is always used on *future* applicants — the validation split should reflect this. A random split produces an optimistic metric that does not represent deployment performance.

---

### Issue 6 — No Missingness Handling

**Severity: MEDIUM**

`rainfall_deviation_pct` has 750 missing values (15%) and `ndvi_score` has 750 missing values. The original script neither acknowledges nor handles these nulls. XGBoost will handle NaN internally, but:

- Downstream code or a different model would fail silently.
- No assumption is stated about *why* these values are missing (is it MCAR, MAR, or MNAR?).
- Both columns are excluded in the fixed script for leakage reasons anyway, making this moot for the baseline — but it should be documented.

---

### Issue 7 — No Feature Availability Documentation

**Severity: MEDIUM**

No comment or documentation explains whether each feature is available *before* the scoring date. Features like `rainfall_deviation_pct` (season-end outcome) and `ndvi_score` (satellite imagery at harvest) are unknowable at application time and constitute future-data leakage.

---

### Issue 8 — No Reproducibility Guarantees Beyond `random_state`

**Severity: LOW**

No logging of library versions, training date, or dataset hash. If the data file changes and someone reruns the script, there is no way to know the model changed.

---

## 2. What Was Changed and Why

| Area | Change | Reason |
|---|---|---|
| **Leakage** | Removed `defaulted_in_next_12_months` from features | Future outcome — unavailable at scoring time |
| **Leakage** | Removed `rainfall_deviation_pct` and `ndvi_score` | Season-end/harvest metrics — unavailable at loan application time |
| **Policy separation** | Moved PM Kisan rule out of label mutation into `apply_business_policy()` called post-prediction | Policy must be separate from model for governance, auditability, and independent changeability |
| **Preprocessing** | Replaced dual-reused `LabelEncoder` with `ColumnTransformer` + `OrdinalEncoder` | Each column gets its own encoder; full pipeline serialized as one object |
| **Serialization** | `joblib.dump(pipeline, ...)` saves the whole Pipeline (preprocessor + model) | Inference is reproducible without reconstructing transforms |
| **Validation** | Temporal split: hold out `application_year == 2024` | Simulates scoring future farmers; avoids future-into-past leakage |
| **Validation** | Also report random holdout for comparison | Baseline reference; temporal is the one to trust |
| **Metrics** | Added MAE alongside R² | MAE is interpretable in score units; R² alone is insufficient |
| **Explainability** | Added `generate_reason_codes()` — rule-based, stable | Reason codes must survive model updates; SHAP values are not stable across retrains |
| **Reproducibility** | Added `version_metadata.json` with run provenance | Audit trail |
| **Feature documentation** | `FEATURE_REGISTRY` with availability assumptions per feature | Explicit contract; makes temporal leakage risks visible |
| **Logging** | Structured logging throughout | Reproducible run output |

---

## 3. Validation Approach

**Choice:** Temporal holdout — hold out all rows with `application_year == 2024` as the test set. Train on `application_year < 2024`.

**Why:** A scoring engine scores applicants who arrive *after* the model was trained. A random split allows the model to train on 2024 data and test on 2022 data, which reverses the temporal order. The temporal holdout better simulates the deployment scenario.

**Limitation:** With only three years of data (2022–2024), the temporal test set is a single year. Year-specific effects (a drought year, a policy change, a crop price shock) could make 2024 systematically different from 2022–2023 in ways unrelated to model generalization. The temporal R² gap vs. random R² should be monitored but not over-interpreted from a single year.

**Results:**
- Random holdout R² = 0.70, MAE = 47.4
- Temporal holdout R² = 0.69, MAE = 48.4

The small gap is reassuring but not conclusive with one year of temporal data.

---

## 4. What Limitations Remain

**One thing I would not trust yet:** The PM Kisan deduction of −150 points. This is a large, flat deduction applied to ~30% of farmers. It was not derived from the data — it appears to be a manually set policy constant. Applied to a score range of ~421–980, a −150 deduction can push borderline farmers into a very low tier. No documentation exists for how this value was chosen, whether it is calibrated to actual repayment risk, or whether it is legally defensible as a credit-scoring input in India's regulatory context.

**One thing I would improve with more time:** Proper temporal cross-validation (walk-forward validation) across all three years — train on 2022 → test on 2023, train on 2022–2023 → test on 2024 — rather than a single year holdout. This would give a more stable estimate of generalization error and reveal whether performance degrades as the model ages.
