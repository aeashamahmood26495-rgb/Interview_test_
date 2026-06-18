# LLM_NOTES.md — AI Tool Disclosure

## Tools Used
- **Claude (claude.ai, Sonnet 4.6)** — code review, script generation, memo drafting

---

## Where I Used Claude

1. **Audit phase** — pasted the broken script and asked Claude to identify ML/data engineering issues. Used as a second pair of eyes after forming my own initial list.
2. **Script writing** — used Claude to scaffold the fixed training script, then reviewed and edited every section.
3. **Memo and README** — used Claude to structure the audit memo; rewrote and verified all claims personally.

---

## Three Actual Prompts Used

**Prompt 1 (audit):**
> "Here is a training script for a farmer credit scoring model. I want you to identify every ML/data engineering issue — focus on data leakage, validation methodology, preprocessing, model/policy separation, and reproducibility. Be specific about why each issue is dangerous, not just that it exists."

**Prompt 2 (code):**
> "Rewrite the training script fixing all the issues you identified. Requirements: (1) use sklearn Pipeline with ColumnTransformer so preprocessing is serialized with the model, (2) move the PM Kisan rule out of label mutation into a separate post-prediction function, (3) implement a temporal train/test split holding out the most recent year, (4) add a feature registry that documents which features are available before the scoring date, (5) add a reason code generator that is rule-based and stable across retrains. Keep it simple and defensible."

**Prompt 3 (memo):**
> "Write an audit memo for the issues in the broken script. For the 'shared LabelEncoder' issue, explain specifically what goes wrong at inference time when the encoder state is wrong, not just that it's 'bad practice'. For the target mutation issue, explain why separating policy from model is a governance concern, not just a code style concern."

---

## What Suggestions I Accepted

- Claude's breakdown of the `LabelEncoder` reuse bug (second `fit_transform` clobbers the first encoder's `classes_` attribute — I had spotted the bug but hadn't articulated the inference-time failure mode this precisely).
- The `ColumnTransformer` + `OrdinalEncoder` structure with `handle_unknown="use_encoded_value"` for robustness at inference.
- The `version_metadata.json` artifact as a lightweight provenance record.

---

## What I Rejected or Corrected

**Example — SHAP-based reason codes:**  
Claude's initial draft of the fixed script included SHAP-based reason codes using `shap.TreeExplainer`. I removed this and replaced it with the rule-based `generate_reason_codes()` function.

**Why:** SHAP values are not stable identifiers for reason codes in a production scoring system. If the model is retrained (even with the same hyperparameters but slightly different data), feature importance rankings can shift, causing the same farmer's application to receive different reason codes on different runs. Regulatory guidance on adverse action notices (including analogues in India's credit reporting framework) requires reason codes to be stable and human-auditable. A rule-based function tied to explicit thresholds is auditable, changeable by a domain expert without ML knowledge, and stable across retrains.

---

## What I Personally Verified

- **Leakage reasoning:** Manually confirmed that `defaulted_in_next_12_months` is a forward-looking outcome column by checking the dataset column name against domain logic. Confirmed `rainfall_deviation_pct` and `ndvi_score` are harvest/season-end metrics unavailable at loan origination.
- **Validation split:** Ran the script and confirmed the temporal split produces 3,606 training rows (years 2022–2023) and 1,394 test rows (year 2024). Checked with `df[SPLIT_YEAR_COL].value_counts()`.
- **Preprocessing boundary:** Confirmed the `ColumnTransformer` is fit only on `X_train` (inside `pipeline.fit(X_train, y_train)`), not on the full dataset. Test sets never touch the fitted encoder.
- **Saved artifacts:** Verified `artifacts/` folder contains all five expected files after a clean run. Loaded `xgboost_baseline.pkl` with `joblib.load()` and confirmed it returns a sklearn Pipeline object with `.predict()` callable.
- **Reason code logic:** Manually traced `generate_reason_codes()` against two sample rows to confirm thresholds fire correctly.
- **Run output:** Confirmed the script runs end-to-end with no warnings or errors on the provided dataset.
