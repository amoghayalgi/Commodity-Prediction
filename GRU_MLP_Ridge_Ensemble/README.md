# Mitsui Commodity Prediction Challenge

Kaggle competition approach using a multi-model ensemble (GRU, MLP, Ridge) to predict commodity price returns across 4 forecast horizons (lag-1 through lag-4).

**Competition:** [MITSUI&CO. Commodity Prediction Challenge](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge)  
**Metric:** Spearman IC Sharpe Ratio — mean daily cross-sectional Spearman rank correlation divided by its standard deviation

---

## Approach Overview

The pipeline trains 9 component models covering three model families, all outputting predictions for all 4 lag horizons simultaneously. A weighted ensemble selects the best combination using a tune/held-out split of the validation set.

**Key design choices:**
- Targets are vol-normalized and cross-sectionally rank-transformed before training
- Multi-lag auxiliary heads (lags 2–4 via forward-shifted pseudo-labels) act as regularizers
- GRU receives only raw (non-engineered) features to let it learn its own temporal patterns
- Ridge meta-models stack GRU and MLP embeddings as a learned second stage
- Ensemble weights are selected by optimizing Spearman IC Sharpe on a tune split (60% of val), then reported on a held-out split (40% of val)

---

## Models & Hyperparameters

### 1. Shared Ridge — Exp A (`Ridge_A`)

One Ridge model per lag (4 total), trained on all features jointly across all targets.

| Hyperparameter | Value |
|---|---|
| Targets | Rank-transformed (cross-sectional, scaled to [−1, 1]) |
| Feature set | All engineered features + cross-sectional rank features |
| Alpha candidates | 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000 |
| Alpha selection | 5-fold TimeSeriesSplit CV (Spearman IC) |
| Lags trained | 4 (averaged for final prediction) |

---

### 2. Per-Target Ridge — Exp B (`Ridge_per_target`)

One Ridge model per target per lag (424 × 4 = 1696 models), each using only the top-50 features most correlated with that target.

| Hyperparameter | Value |
|---|---|
| Feature selection | Top 50 by Pearson correlation (per target, computed on observed rows only) |
| Targets | Rank-transformed + StandardScaled per target |
| Alpha candidates | Same grid as Exp A |
| Alpha selection | 5-fold TimeSeriesSplit CV on lag-1 rank-Y; reused across lags |
| Lags trained | 4 (averaged for final prediction) |

---

### 3. MLP Encoder (`MLP`)

Multi-task feedforward network predicting all targets across all 4 lags simultaneously.

| Hyperparameter | Value |
|---|---|
| Architecture | BN → Linear(512) → ReLU → Dropout(0.3) → Linear(256) → ReLU → Dropout(0.3) → Linear(128, embedding) → ReLU → Linear(out) |
| Embedding dim | 128 |
| Output dim | 424 × 4 = 1696 |
| Targets | Vol-normalized (clipped to [−5, 5]) |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 64 |
| Max epochs | 80 |
| Early stopping patience | 10 (based on val MSE over all 4 lags) |

---

### 4. GRU Encoder — Window 4 (`GRU_w4`)

Multi-task GRU over a 4-step rolling window, trained only on raw (non-engineered) features.

| Hyperparameter | Value |
|---|---|
| Input | Raw features only (no lags, rolling stats, or rank features) |
| Architecture | GRU(256, layers=2, dropout=0.2) → Linear(128) → ReLU → Dropout(0.2) → Linear(out) |
| Embedding dim | 256 (last hidden state) |
| Output dim | 424 × 4 = 1696 |
| Sequence window | 4 timesteps |
| Targets | Vol-normalized (clipped to [−5, 5]) |
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Max epochs | 80 |
| Early stopping patience | 10 (based on val MSE) |
| Val warmup | Last 3 training rows prepended to val for sequence continuity |

---

### 5. GRU Encoder — Window 10 (`GRU_w10`)

Same architecture as GRU_w4, but with a longer 10-step window to capture medium-term patterns.

| Hyperparameter | Value |
|---|---|
| Architecture | Identical to GRU_w4 |
| Sequence window | 10 timesteps |
| Val warmup | Last 9 training rows prepended to val |
| All other params | Identical to GRU_w4 |

---

### 6. GRU Average (`GRU_avg`)

Simple average of `GRU_w4` and `GRU_w10` predictions — no additional training.

---

### 7–9. Ridge Meta-Models on Embeddings (`Ridge_GRU`, `Ridge_GRU_MLP`, `Ridge_mega`)

Three Ridge models trained on different combinations of learned embeddings as a second-stage stacker.

| Name | Input Features | Description |
|---|---|---|
| `Ridge_GRU` | GRU_w4 emb (256) + GRU_w10 emb (256) | GRU embeddings only |
| `Ridge_GRU_MLP` | GRU_w4 emb + GRU_w10 emb + MLP emb (128) | GRU + MLP embeddings |
| `Ridge_mega` | GRU_w4 emb + GRU_w10 emb + MLP emb + raw scaled features | All embeddings + original features |

| Shared Hyperparameter | Value |
|---|---|
| Alpha selection | Grid search on official val Sharpe (lag-1) |
| Alpha candidates | Same grid as Exp A |
| Lags trained | 4 separate models per configuration |
| Targets | Vol-normalized (lag-k per model) |

---

## Ensemble

Ensemble weights are selected from three strategies evaluated on the tune split (first 60% of the val set), with the winning strategy confirmed on the held-out split (last 40%).

**Strategies evaluated:**
- **Strategy A** — Equal weights across all passing candidates
- **Strategy B** — Top-3 models by individual tune score, weighted equally (1/3 each)
- **Strategy C** — Softmax-weighted by individual tune score

**Candidate filtering:** Any model with a negative all-lag Sharpe on the full val set is excluded before weight selection.

### Final Ensemble Weights

The weights below reflect the best strategy selected at runtime and will vary depending on model training outcomes. The structure is:

```
Final prediction = Σ (w_i × predictions_i)   for all passing candidates i
```

All weights are non-negative and sum to 1. Per-run weights are printed in the notebook output under `"BEST STRATEGY"`.

---

## Results

### Validation (last 15% of training dates, time-based split)

| Model | Val Sharpe (lag-1) | Val Sharpe (all-lag avg) |
|---|---|---|
| Ridge_A | logged at runtime | logged at runtime |
| Ridge_per_target | logged at runtime | logged at runtime |
| MLP | logged at runtime | logged at runtime |
| GRU_w4 | logged at runtime | logged at runtime |
| GRU_w10 | logged at runtime | logged at runtime |
| GRU_avg | logged at runtime | logged at runtime |
| Ridge_GRU | logged at runtime | logged at runtime |
| Ridge_GRU_MLP | logged at runtime | logged at runtime |
| Ridge_mega | logged at runtime | logged at runtime |
| **Ensemble (lag-1)** | **printed as `best_sc_lag1`** | — |
| **Ensemble (held-out 40%)** | — | **printed as `best_sc_held_full`** |

> All scores are printed in full in the notebook's `VALIDATION PERFORMANCE SUMMARY` cell.

### Offline Test Evaluation (ground truth labels)

Evaluated against all 4 lag label files using the official Spearman IC Sharpe metric per lag block, then averaged.

| Metric | Value |
|---|---|
| Test Sharpe — lag-1 | printed as `Test Sharpe (lag-1, ...)` |
| Test Sharpe — lag-2 | printed as `Test Sharpe (lag-2, ...)` |
| Test Sharpe — lag-3 | printed as `Test Sharpe (lag-3, ...)` |
| Test Sharpe — lag-4 | printed as `Test Sharpe (lag-4, ...)` |
| **Average over all lags** | **printed as `Offline test Sharpe`** |

> Run the final cell `OFFLINE TEST EVALUATION USING GROUND TRUTHS` to reproduce these numbers.

---

## Repo Structure

```
├── GRU_MLP_Ridge.ipynb     # Full pipeline: preprocessing, training, ensemble, evaluation
├── README.md
```

### Required input files (place in the same directory as the notebook)

```
train.csv
train_labels.csv
test.csv
test_labels_lag_1.csv
test_labels_lag_2.csv
test_labels_lag_3.csv
test_labels_lag_4.csv
```

---

## Dependencies

```
numpy
pandas
scipy
torch
scikit-learn
```

GPU is used automatically if available (`cuda`), otherwise falls back to CPU.

---

## Reproducing

1. Place all data files in the same directory as the notebook
2. Run all cells top to bottom
3. Final test scores are printed in the last cell under `OFFLINE TEST EVALUATION USING GROUND TRUTHS`
4. `ground_truths.csv` is written to disk as a reconstructed ground truth matrix for reference