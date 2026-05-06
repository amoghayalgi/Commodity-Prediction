# Commodity-Prediction
The goal of this project is to leverage cross‑market signals to produce stable, long‑term forecasts that support trading strategy optimization and global risk management.

## Ridge Rank-Y Ensemble — Full Method Summary

**Task:** MITSUI commodity prediction · Leakage-aware offline evaluation pipeline
**Final model:** `ensemble_equal` · **Offline test score: `0.281504`** (134 test dates, date_id 1827–1960)

> *Note: All scores are offline estimates computed from official Kaggle leaderboard score codes.*

---

### 1. Evaluation Windows

Three strictly chronological windows with no leakage between them:

| Window | date_id | Purpose |
|---|---|---|
| Train | 20–1568 | Fit all base models; tune hyperparameters via TimeSeriesSplit CV |
| Validation | 1569–1826 | Select ensemble weights; compare model families |
| Final Test | 1827–1960 | Generate final predictions; report final offline score |

- All model fitting uses **train window only**
- Ensemble weights are selected on **validation window only**
- `test_ground_truth` is loaded but held out until final offline scoring
- Momentum signals are generated from historical train labels shifted by the appropriate lag offset — no future labels leak into training or validation features

---

### 2. Feature Data (X)

- **Base feature pool:** 2,850 columns after removing high-missing and zero-variance columns (using training-window statistics only); missing values filled with 0
- **Augmentation note:** Lag-summary augmentation was applied but generated **0 new columns** in this run — `label_target_*_lag*` columns were not usable in the train window after cleaning. As a result, `A_no_aug` and `A_aug` are numerically identical.

| Split | Approx. rows | date_id |
|---|---|---|
| X_train_c | 1,549 | 20–1568 |
| X_val_c[:258] | 258 | 1569–1826 |
| X_val_c[258:] (test) | 134 | 1827–1960 |

---

### 3. Label Data (Y)

Two target formulations used across the model family:

- **Raw Y** — original return values from `train_labels.csv`
- **Rank Y** — daily cross-sectional rank transformed to [−1, 1], aligning the learning objective with within-day relative ordering rather than exact return magnitude

| Label set | date_id | Source |
|---|---|---|
| Training labels | 20–1568 | `train_labels.csv` aligned to train window |
| Validation labels | 1569–1826 | `train_labels.csv` aligned with NaN preserved |
| Final test labels | 1827–1960 | Reconstructed from `test_labels_lag_1.csv` through `test_labels_lag_4.csv`, aligned by `label_date_id` |

---

### 4. Model Family

Eight candidates form the final ensemble:

| Model | Core idea | Features | Target | Rationale |
|---|---|---|---|---|
| `A_no_aug` | Shared multi-output Ridge, one model per lag group (106 targets/group) | All 2,850 base columns | Rank Y → [−1, 1] | Stable baseline with strong regularization and shared structure |
| `A_aug` | Same as A_no_aug with lag-summary augmentation | All 2,850 base columns + lag features (none added this run) | Rank Y → [−1, 1] | Tests whether compact lag summaries add incremental signal |
| `B_raw` | Per-target Ridge, top-50 feature selection | Top-50 per target by \|Pearson r\| on train | Raw Y (unranked) | Tests target-specific feature subsets on raw return prediction |
| `B_rank` | Per-target Ridge, top-50 feature selection | Top-50 per target by \|Pearson r\| on train | Rank Y → [−1, 1] | Tests per-target modeling with a rank-aligned objective |
| `C_conservative` | Shared Ridge, fixed high regularization (α = 10,000) | All 2,850 base columns | Rank Y → [−1, 1] | Simpler benchmark prioritizing out-of-sample stability |
| `C_pca50` | Shared Ridge after PCA dimensionality reduction | 50 PCA components (~88.6% explained variance) | Rank Y → [−1, 1] | Tests whether compressing noisy features into orthogonal factors improves generalization |
| `D_momentum` | Persistence / lag-correction predictor — no training | Last observed return shifted by lag offset | None (no learned transform) | Transparent sanity check for naive momentum signal |
| `E_hgbr` | Shared HistGradientBoosting via MultiOutputRegressor | All 2,850 base columns | Rank Y → [−1, 1] | Non-linear tree-based competitor to the Ridge family |

**Final model:**
```
ensemble_equal = 0.125 × A_no_aug + 0.125 × A_aug + 0.125 × B_raw + 0.125 × B_rank + 0.125 × C_conservative + 0.125 × C_pca50 + 0.125 × D_momentum + 0.125 × E_hgbr
```

---

### 5. Model Settings and Tuned Parameters

**Exp A — Shared Ridge (`A_no_aug`, `A_aug`)**
- Alpha grid: [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
- CV: 5-fold TimeSeriesSplit with Spearman-IC-aligned criterion
- All four lag groups selected **α = 1,000** — consistent preference for strong L2 shrinkage

**Exp B — Per-target Ridge (`B_raw`, `B_rank`)**
- Feature selection: top-50 per target by |Pearson r| on training window
- Alpha tuned independently per target; full distribution across 424 targets:

| Model | 0.0001 | 0.001 | 0.01 | 0.1 | 1 | 10 | 100 | 1,000 | 10,000 | 100,000 |
|---|---|---|---|---|---|---|---|---|---|---|
| B_raw | 8 | 4 | 10 | 11 | 7 | 17 | 60 | 106 | 85 | 116 |
| B_rank | 10 | 1 | 2 | 1 | 4 | 8 | 38 | 149 | 123 | 88 |

Per-target tuning spans nine orders of magnitude, reflecting large heterogeneity in how much shrinkage each target requires.

**Exp C — Conservative and PCA Ridge**
- `C_conservative`: fixed α = 10,000; no CV tuning
- `C_pca50`: 50 PCA components; α tuned by 5-fold CV

| Lag group | Best α (`C_pca50`) | PCA explained variance |
|---|---|---|
| Lag 1 | 10,000 | 88.6% |
| Lag 2 | 1,000 | 88.6% |
| Lag 3 | 1,000 | 88.6% |
| Lag 4 | 1,000 | 88.6% |

**Exp D — Momentum (`D_momentum`)**
- No training, no alpha
- Prediction(date d, target i) = last known return at d − (lag_k + 1)
- Lag group 1 → lookback 2 rows; Lag 2 → 3 rows; Lag 3 → 4 rows; Lag 4 → 5 rows

**Exp E — HistGradientBoosting (`E_hgbr`)**

| Setting | Value |
|---|---|
| max_iter | 80 |
| max_leaf_nodes | 15 |
| min_samples_leaf | 50 |
| learning_rate | 0.10 |
| l2_regularization | 1.0 |
| max_features | 0.5 |
| random_state | 42 |
| Wrapper | MultiOutputRegressor (n_jobs = −1) |
| Models trained | 4 (one per lag group, 106 targets each) |

---

### 6. Results

| Model | Val (258 rows) | Test (134 rows) | Comment |
|---|---|---|---|
| `A_no_aug` | 0.185347 | 0.224590 | Shared Ridge baseline; identical to A_aug in this run |
| `A_aug` | 0.185347 | 0.224590 | No effective augmentation columns added |
| `B_raw` | 0.231981 | 0.051013 | Per-target raw-Y; generalizes poorly |
| `B_rank` | 0.347119 | 0.106978 | Best single-model validation; severe test drop |
| `C_conservative` | 0.280793 | **0.315748** | Best single-model test score |
| `C_pca50` | 0.286845 | 0.143220 | Good validation; weaker test generalization |
| `D_momentum` | −0.000951 | −0.163262 | Persistence benchmark underperforms |
| `E_hgbr` | 0.124955 | 0.246965 | Respectable test score |
| `ensemble_optimized` | 0.392803 | 0.189018 | Overfits to validation regime |
| **`ensemble_equal`** | **0.244744** | **0.281504 ← final** | Pre-committed submission |

---

### 7. Ensemble Logic

| Ensemble | Rule | Val score | Test score |
|---|---|---|---|
| `ensemble_optimized` | Weights optimized on validation window | 0.392803 | 0.189018 |
| `ensemble_equal` | Equal weight across all 8 candidates | 0.244744 | **0.281504** |

`ensemble_equal` was pre-committed as the final model before any test-set scoring. The choice reflects a robustness preference: an optimized rule that reacts strongly to one validation regime can under-generalize when the test regime shifts. Equal weighting sacrifices some validation peak performance in exchange for more reliable future generalization.

---

### 8. Key Findings

- **Shared Ridge is the core of the solution** — target-specific models look attractive on validation but consistently lose generalization on the final test
- **`A_no_aug` = `A_aug` in this run** — lag-summary augmentation produced no usable columns; this is a pipeline finding, not a modeling result
- **`B_rank` overfits to validation** — best single-model validation score (0.347) but drops to 0.107 on test; strong evidence of regime-specific overfitting in per-target feature selection
- **`C_conservative` is the best single model on test** (0.316) — confirms that heavy regularization improves out-of-sample robustness
- **Optimized ensemble fails to generalize** — validation-optimized weights (0.393) collapse to 0.189 on test; equal weighting is more reliable
- **`D_momentum` adds no signal** — negative test score confirms that naive persistence provides no predictive value in this setting

---

### 9. Output Files

`output_Ridge_RankY_final.csv` · `Final_Score_Summary.csv` · `Ensemble_Weights.csv` · `All_Alpha_Parameters.csv`




---
# LightGBM, XGBoost, CatBoost
The notebook implements a predictive modeling framework to forecast commodity prices or related financial targets. The primary objective is to optimize for a specific evaluation metric: the Rank Correlation Sharpe Ratio.  

## 1. Data Processing & Feature Engineering
**Data Split:** The dataset is divided into training, monitoring (18% of training data), and holdout validation sets. The monitoring tail is specifically used for hyperparameter tuning and early stopping.  

**Feature Selection Logic:** A custom function, get_pair_specific_features, filters relevant features based on specific target IDs and instrument pairs. It incorporates "Macro Core" features such as FX_USD, SPY, VIX, and HYG to provide market context.  

**Lag Management:** The pipeline includes logic to filter out conflicting time lags, ensuring only the intended lags (e.g., t_lag) are used for specific models.  

## 2. Modeling Strategy
The project utilizes a powerful ensemble approach, importing the following gradient-boosting libraries:

**LightGBM:** Used as the primary regressor for optimization.  

**XGBoost & CatBoost:** Integrated for potential model diversity.  

**Optuna:** Employed for automated hyperparameter optimization, specifically minimizing the Mean Absolute Error (MAE) through time-series cross-validation.  

## 3. Best Hyperparameters
--- LightGBM ---
{
  "objective": "regression_l1",
  "metric": "mae",
  "boosting_type": "gbdt",
  "bagging_freq": 5,
  "verbosity": -1,
  "learning_rate": 0.018930322028652653,
  "num_leaves": 71,
  "feature_fraction": 0.8252957746512221,
  "bagging_fraction": 0.6719345241664011,
  "lambda_l1": 1.028542539395875,
  "lambda_l2": 0.014934744734860941,
  "min_data_in_leaf": 23
}

--- XGBoost (sklearn API; n_estimators set at train time) ---
{
  "objective": "reg:absoluteerror",
  "tree_method": "hist",
  "learning_rate": 0.011760188143916252,
  "max_depth": 11,
  "subsample": 0.6740498075439663,
  "colsample_bytree": 0.6274322213703454,
  "reg_alpha": 0.6490187056934515,
  "reg_lambda": 3.8182015265198737,
  "min_child_weight": 5
}

--- CatBoost (iterations set at train time) ---
{
  "learning_rate": 0.011005367809458285,
  "depth": 9,
  "l2_leaf_reg": 3.613728617871899,
  "subsample": 0.6356190063209927,
  "rsm": 0.8574039697263778,
  "min_data_in_leaf": 77
}

## 4.Best Model

HOLDOUT VALIDATION (report-only; X_val_all / Y_val_all)

- LightGBM — Holdout validation Sharpe: 0.1189
- XGBoost — Holdout validation Sharpe: 0.0683
- CatBoost — Holdout validation Sharpe: 0.1478

## 5. Evaluation Metrics
The notebook defines a custom scoring function that calculates:

**Rank Correlation:** Measuring the relationship between the ranks of predicted and actual values.  

**Sharpe Ratio:** Calculated by dividing the mean of daily rank correlations by their standard deviation, rewarding both accuracy and consistency
