# **Transformer Model — MITSUI Commodity Prediction Challenge**

### **Lag-1 (1-Day-Ahead) Return Prediction**

The goal of this model is to leverage sequential cross-market signals through a Transformer encoder architecture to produce stable 1-day-ahead return forecasts, supporting trading strategy optimization and global risk management.

---

## **Pipeline Overview**

`Load cleaned splits → Add temporal encoding → Scale X + PCA → Standardize Y → Build rolling-window sequences → Train Transformer encoder → Score on observed days only → Export predictions & weights`

Inputs are the **already-cleaned and engineered** outputs of `data_cleaning_v2.ipynb`. X contains lag features, rolling stats, return/momentum, cross-market spreads, label-lag features, and `_observed` flag columns. Y NaNs were pre-filled with 0 for market-closed days, and the first 20 warm-up rows were dropped during cleaning.

---

## **Architecture**

Encoder-only Transformer:

Input (batch, T=20, 50\)  
  → Linear projection → Positional Encoding (sinusoidal)  
  → 3 × TransformerEncoderLayer (Pre-LN, multi-head attention \+ FFN)  
  → Mean pooling over sequence  
  → MLP head (Linear → GELU → Dropout → Linear)  
  → (batch, n\_targets)

| Hyperparameter | Value |
| ----- | ----- |
| `d_model` | 128 |
| `nhead` | 4 |
| `num_encoder_layers` | 3 |
| `dim_feedforward` | 256 |
| `dropout` | 0.1 |
| Sequence length (T) | 20 |
| PCA components | 50 |
| Batch size | 64 |

Positional encoding follows the classic Vaswani et al. (2017) sinusoidal scheme. Pre-LayerNorm (`norm_first=True`) is used for training stability.

---

## **Preprocessing**

**Feature matrix (X):** Residual NaNs (from lag/rolling warm-up at series start) are filled with 0, then `StandardScaler` is fit on train only. PCA is then applied to reduce to 50 components.

**Target matrix (Y):** Each target is independently standardized (mean/std from train). Predictions are inverse-transformed back to original scale before evaluation and export.

**Temporal encoding:** Since `dateid` was dropped during cleaning, cyclical day-of-week features (`dow_sin`, `dow_cos`) are reconstructed from a business-day range anchored at the known train/val split.

**Sequence construction:** Rolling windows of T=20 trading days are built. Each sample uses days `[i-T .. i-1]` as input and day `i` as the label. The first T rows of each split are lost to warm-up and are expected.

---

## **Best Hyperparameters**

## { "d\_model": 128, "nhead": 4, "num\_encoder\_layers": 3, "dim\_feedforward": 256, "dropout": 0.1, "sequence\_length": 20, "input\_features": 50, "batch\_size": 64, "learning\_rate": 0.001, "weight\_decay": 0.0001, "max\_epochs": 30, "patience": 5, "grad\_clip\_norm": 1.0, "pca\_components": 50, "optimizer": "AdamW", "scheduler": "CosineAnnealingLR" }

## **Training**

| Setting | Value |
| ----- | ----- |
| Loss | MSE on standardized Y |
| Optimizer | AdamW (`lr=1e-3`, `wd=1e-4`) |
| Scheduler | CosineAnnealingLR (`T_max=EPOCHS`) |
| Gradient clipping | max norm 1.0 |
| Max epochs | 30 |
| Early stopping | Patience \= 5 epochs on val MSE |

The best model state (by val MSE) is restored after early stopping triggers.

---

## **Evaluation**

Validation metrics are computed **only on observed trading days** using the `obs_val` mask. Each target has its own `<target>_observed` column because different markets follow different holiday calendars. Scoring against filled zeros (market-closed rows) would artificially inflate results.

Metrics reported per target:

* **MSE** (primary) — on inverse-transformed, original-scale predictions  
* **Pearson r** (secondary) — rank-order correlation between predicted and actual returns

---

## **Output Files**

| File | Description |
| ----- | ----- |
| `transformer_best.pt` | Best model weights (by val MSE) |
| `transformer_val_preds.csv` | Validation predictions in original return scale |
| `transformer_val_metrics.csv` | Per-target MSE and Pearson r on observed days only |
| `transformer_training_curve.png` | Train vs. val MSE across epochs |

---

## **Key Design Decisions**

* **Observed-flag columns kept in X** — `_observed` flags are intentional input signals that allow the model to learn to discount filled-zero rows for closed markets.  
* **PCA before attention** — reduces the feature space from \~2850+ dimensions to 50 components before the Transformer, since attention becomes noisy with many raw features.  
* **Mean pooling** — aggregates the full sequence representation rather than using only the final token, improving stability across variable market conditions.  
* **Standardized Y, inverse-transformed predictions** — training in standardized space normalizes target scale across commodities; evaluation happens in original return space against the obs\_val mask.

---

## **Scaling to All 424 Targets**

Once lag-1 is validated, scaling to the full target set requires zero pipeline changes. Simply swap the target CSVs:

\# Lag-1 only (106 targets):  
Y\_train \= pd.read\_csv('Y\_train\_lag1.csv')  
Y\_val   \= pd.read\_csv('Y\_val\_lag1.csv')

\# All targets (424 targets):  
Y\_train \= pd.read\_csv('Y\_train.csv')  
Y\_val   \= pd.read\_csv('Y\_val.csv')

\# X stays identical — all engineered features are shared across all targets.

---

## **Dependencies**

numpy, pandas, scikit-learn, torch, scipy, matplotlib

GPU is used automatically if available (`torch.cuda.is_available()`). The full pipeline runs on CPU as well, though training will be significantly slower.

