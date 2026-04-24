# Predictive Paradox — Summary Report
**Next-Hour Electricity Demand Forecasting**

---

## What This Project Does

Builds a machine learning pipeline to predict `demand_mw` at time `t+1` — i.e., the next hour's electricity demand on the national grid — using information available up to and including time `t`. Three data sources are merged: hourly PGCB demand/generation data (`PGCB_date_power_demand.xlsx`), hourly weather observations (`weather_data.xlsx`), and annual World Bank macroeconomic indicators (`economic_full_1.csv`).

---

## Model Selection

| Model | Used | Notes | Verdict |
|---|---|---|---|
| **Random Forest** | ✅ | Robust to noise, handles tabular data well, no feature scaling needed | **Primary & Final** |
| LightGBM / XGBoost | ❌ | Not used in this notebook | — |
| LSTM / Transformer | ❌ | Not permitted | Ruled out |
| ARIMA / Prophet | ❌ | Not permitted | Ruled out |

**Why Random Forest?** `RandomForestRegressor` from scikit-learn was chosen for its robustness on tabular data with mixed feature types, ability to handle non-linear relationships between lag/rolling features and demand, built-in feature importance ranking, and ease of use without requiring feature scaling. The model was configured with `n_estimators=10`, `max_depth=8`, and `n_jobs=-1` for parallel training.

---

## Data Cleaning Rationale

### Step-by-Step Pipeline (Steps 1–12)

**Loading:** Three datasets are loaded — power demand (Excel), weather (Excel), and economic indicators (CSV).

**Duplicates:** Duplicate rows are dropped from all three datasets immediately after loading to avoid training on repeated observations.

**Date Parsing:** The first column of both power and weather DataFrames is coerced into a proper `datetime` object using `pd.to_datetime(..., errors="coerce")`. Any rows where the date conversion fails (returns `NaT`) are dropped, and both datasets are sorted chronologically before merging.

**Missing Value Imputation:** Numeric missing values are filled using the **column-wise median** (`fillna(median(numeric_only=True))`). The median is preferred over the mean because electricity demand distributions can be skewed by peak events, making the median a more robust central estimate. In the rebuild step (Step 20), infinite values are also replaced with `NaN` and a forward-fill followed by backward-fill is applied to handle edge gaps.

### Merging Datasets
The power and weather datasets are merged on the `date` column using a left join (keeping all power rows). A `year` column is extracted from the date and used as the join key to merge annual economic indicators — the assumption being that macroeconomic conditions are stable within a given year and represent the slow background trend around which hourly demand fluctuates.

---

## Feature Engineering

Since Random Forest treats each row independently with no inherent temporal memory, all historical context must be manually encoded as features.

### Calendar / Time Features
| Feature | Why |
|---|---|
| `hour` | Demand peaks at morning and evening hours |
| `day` | Captures within-month variation |
| `month` | Seasonal demand patterns (summer cooling, winter heating) |
| `dayofweek` | Weekday vs weekend consumption patterns |
| `is_weekend` | Explicit binary flag — weekend demand is systematically lower |

### Lag Features
These are the most critical feature group — they give the model a short-term "memory" of recent demand.

| Feature | What It Captures |
|---|---|
| `lag_1` | Demand from one hour ago — strongest single predictor |
| `lag_2` | Two hours ago — helps detect short-term trend direction |
| `lag_3` | Three hours ago — intraday trend context |
| `lag_24` | Same hour yesterday — captures the daily seasonality cycle |

All lag features use `.shift(N)` on the sorted DataFrame. The `.shift(-1)` applied to create the target column ensures the model predicts the *next* hour, not the current one.

### Rolling Mean Features
| Feature | What It Captures |
|---|---|
| `roll_mean_3` | Short-term 3-hour smoothed demand |
| `roll_mean_6` | 6-hour smoothed demand level |
| `roll_mean_24` | 24-hour smoothed demand level — daily baseline |

All rolling features apply `.shift(1)` before `.rolling(N).mean()` to prevent data leakage — the current hour's actual value never contributes to its own input features.

### Categorical Encoding
Any remaining object/string columns are handled via `pd.get_dummies(..., drop_first=True)` before training to ensure all features are numeric.

### Target Column
`target = demand_mw.shift(-1)` — the demand value one hour into the future. The last row (which has no future value) is dropped before training.

---

## Validation Approach

- All rows sorted chronologically by `date` before any processing
- **Strict temporal split: 80% training / 20% testing** based on row order — no shuffling
- This ensures the model is only ever evaluated on data it has never seen during training, mimicking real operational conditions where past data is used to forecast the future
- Rolling and lag features are computed on the full sorted series before splitting — safe because each feature only looks backward via `.shift()`

---

## Feature Importance Insights

The top 10 features by Random Forest impurity-based importance (Step 25) are expected to be:

1. `lag_1` — previous hour demand is the strongest single signal
2. `lag_24` — same hour yesterday captures daily seasonality
3. `roll_mean_24` — smoothed 24-hour demand level acts as a baseline
4. `roll_mean_6` / `roll_mean_3` — shorter-window smoothed trends
5. `lag_2`, `lag_3` — recent trend direction
6. `hour` — time-of-day demand cycle
7. `month` — seasonal variation
8. `dayofweek` / `is_weekend` — weekday vs weekend patterns
9. Economic and weather features — provide background context with lower but non-zero importance

---

## Results Summary

| Metric | Result |
|---|---|
| **Model** | Random Forest (`n_estimators=10`, `max_depth=8`) |
| **Evaluation Metric** | MAPE (Mean Absolute Percentage Error) |
| **Split** | 80% train / 20% test (chronological) |
| **MAPE Score** | Computed in Step 22 via `mean_absolute_percentage_error(y_test, pred) * 100` |

A low MAPE (ideally below 5–6%) on hourly electricity demand is considered acceptable for a single classical ML model without hyperparameter tuning. Given the shallow tree depth (`max_depth=8`) and small forest size (`n_estimators=10`), there is meaningful room for improvement by increasing tree count, adding more lag features, or switching to a gradient-boosted model.

---

## Pipeline Overview (Step-by-Step)

| Step | Action |
|---|---|
| 1 | Import libraries (`pandas`, `numpy`, `sklearn`, `matplotlib`, `seaborn`) |
| 2–3 | Load and inspect all three datasets |
| 4–5 | Check columns, data types, and missing values |
| 6 | Remove duplicate rows |
| 7–8 | Parse dates, drop invalid date rows, sort chronologically |
| 9 | Fill numeric missing values with column median |
| 10 | Rename first column to `date` for convenience |
| 11 | Merge power + weather on `date` |
| 12 | Extract year, merge economic data |
| 13 | Create calendar features (`hour`, `day`, `month`, `day_of_week`, `week`, `is_weekend`) |
| 14 | Create lag features (`lag_1`, `lag_2`, `lag_3`, `lag_24`) |
| 15 | Create rolling mean features (`roll_mean_3`, `roll_mean_6`, `roll_mean_24`) |
| 16 | Create target column (`demand_mw.shift(-1)`) |
| 17 | Drop rows with NaN from lag/rolling/target creation |
| 18 | Prepare `X` (drop `date`, `target`, object columns) and `y` |
| 19 | Chronological 80/20 split |
| 20 | Full rebuild + `RandomForestRegressor` training |
| 21 | Generate predictions on test set |
| 22 | Compute MAPE score |
| 23 | Build actual vs predicted comparison table |
| 24 | Plot Actual vs Predicted line chart (first 100 rows) |
| 25–26 | Compute and plot top-10 feature importances |

---

## Common Errors and Fixes

| Error | Fix |
|---|---|
| `FileNotFoundError` | Ensure `.xlsx` / `.csv` files are in the same directory as the notebook |
| All lag features are NaN | DataFrame must be sorted by date before calling `.shift()` |
| `ValueError: could not convert string to float` | Use `pd.get_dummies()` to encode categorical columns before training |
| MAPE = `inf` or very large | Zero values in `y_test` — mask them: `y_test[y_test != 0]` before MAPE calculation |
| Memory error on large rolling windows | Ensure `.rolling()` is called on a sorted, indexed Series, not an unindexed DataFrame |
| `KeyError: 'demand_mw'` | Check actual column name with `print(data.columns)` and update `demand_col` accordingly |
| Last row has `NaN` target | Expected — drop it with `dropna(subset=["target"])` |

---
