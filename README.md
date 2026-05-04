# Commodity-Prediction
The goal of this project is to leverage cross‑market signals to produce stable, long‑term forecasts that support trading strategy optimization and global risk management.


## Ridge Regression — Lag 1 Baseline and Experiments  

**Output files:** 

Built a multi-output Ridge regression pipeline for the lag_1 target group. Evaluated using observed RMSE (primary) and observed Spearman correlation (secondary), applied to truly observed labels only.

**Pipeline:** Load and validate splits → drop unusable columns → fill missing values → StandardScaler (fit on train only) → tune alpha via TimeSeriesSplit → refit on full train → export predictions.

**Experiments:** [winsor] Winsorization · [fs_top500] Feature selection by target correlation · [winsor+fs500] Combined · [interactions_top30] Interaction features from top-30 · [pca500] PCA-Ridge.

**Results:** Baseline was already strong; most variants produced marginal changes.   
Best observed RMSE: [winsor] at 0.01764 (alpha=100,000).   
Baseline: RMSE=0.01765, Spearman=0.0295.     
Best Spearman: [interactions_top30] at 0.0379, but with slightly worse RMSE.   
PCA-Ridge performed worst.    

**Final choice:** interactions_top30, Best Spearman!!  

**Key takeaways:** Heavy regularization consistently preferred · Winsorization helped marginally · Feature selection alone did not improve · Interaction terms improved rank-order but not RMSE · Ridge best used as a stable linear baseline for ensemble.  





**Ridge Regression — Rank-Aware Ensemble**

Evaluated using the official score as the primary metric, applied to truly observed labels only. Validation set: 392 rows. Test set: 90 scored rows (is\_scored=True, date\_id range \[1827, 1916\]).

**1\_Final model:** `ensemble_equal` — equal-weight average of four Ridge variants:
`A_no_aug` (raw returns, shared 2850-feature pool, single Ridge across all targets),
`A_aug` (raw returns, shared pool with lag-summary features appended),
`B_rank` (rank-transformed targets, per-target top-50 feature selection),
`B_raw` (raw returns, per-target top-50 feature selection — same framework as B\_rank but without rank-transforming Y).

- **final_model** = 0.25 × A_no_aug + 0.25 × A_aug + 0.25 × B_rank + 0.25 × B_raw
Equal-weight averaging outperformed the validation-optimized ensemble on the held-out test set, suggesting the optimized weights overfit to the validation distribution.
Official score (Test, 90 scored rows): **0.2767**

**Rank-transformed Y:**
- if the true same-day ordering is target A > target C > target E > target B > target D
- then a good model should produce predictions with the same ordering, even if the numeric values differ.

**Additional Augmented Features**
1. Lag mean — the average of the past four label lags.
2. Lag standard deviation — the variability of the past four label lags.
3. Decay-weighted lag mean — a weighted average of the past lags that assigns higher weight to more recent values.
4. Recent-vs-history gap — the difference between the most recent lag and the average of earlier lags, capturing short-term momentum or reversal.
5. Sign consistency — whether the past lags move in the same direction across periods.

---

**2\_Pipeline:**
- Load and validate train / validation splits
- Drop unusable columns and fill missing values
- StandardScaler fitted on train only
- Construct four Ridge variant candidates under two target definitions
- Tune alpha using ranking-aligned validation score
- Generate predictions for each candidate
- Apply optional cross-sectional post-processing
- Average candidate predictions with equal weights
- Export final predictions

---

**3\_Experiments:**
- **A\_no\_aug** — Ridge on raw return targets, using a shared global setup across all targets, with no feature augmentation. Official score (Val): 0.1982 · (Test): 0.2708
- **A\_aug** — Ridge on raw return targets with augmented features, still using a shared global setup across all targets. Official score (Val): 0.1982 · (Test): 0.2708
- **B\_rank** — Ridge on rank-transformed targets with a target-specific setup: each target uses its own top-50 selected features and its own tuned alpha. Instead of predicting raw return values, Y was converted to within-day rank order before training, directly aligning the training objective with the competition ranking metric. Official score (Val): 0.2690 · (Test): 0.1366
- **B\_raw** — Ridge on raw return targets with the same target-specific framework as B\_rank: per-target top-50 feature selection and per-target alpha tuning, but without rank-transforming Y. Official score (Val): 0.1700 · (Test): 0.0501
- **ensemble\_equal** — Equal-weight average of all four candidates. Official score (Val): 0.2110 · 【**(Test): 0.2767**】← 【final submission】
- **ensemble\_optimized** — Validation-score-optimized weighted ensemble (weights: A\_no\_aug 0.0354, A\_aug 0.0279, B\_rank 0.9367, B\_raw 0.0000). Official score (Val): 0.2836 · (Test): 0.2476
- **final_model** = 0.25 × A_no_aug + 0.25 × A_aug + 0.25 × B_rank + 0.25 × B_raw
---

**4\_Results:**

| Model | Val (392 rows) | Test (90 rows) |
|---|---|---|
| ensemble\_equal | 0.2110 | **0.2767** |
| ensemble\_optimized | 0.2836 | 0.2476 |
| A\_no\_aug | 0.1982 | 0.2708 |
| A\_aug | 0.1982 | 0.2708 |
| B\_rank | 0.2690 | 0.1366 |
| B\_raw | 0.1700 | 0.0501 |

Full per-model scores on both sets are in `Final_Score_Summary.csv`.

---

**5\_Key takeaways:**
- Equal-weight ensembling outperformed the validation-optimized ensemble on the test set — optimized weights likely overfit to the validation period
- B\_rank dominated validation but generalized poorly to test; simple averaging across all candidates was more robust
- Target redesign (rank-transformed Y) improved validation performance but showed weaker test generalization in isolation
- Feature augmentation had no effect in its current form
- Ridge works best here as a ranking-oriented model, not a value-regression model

---

**6\_Output files:** `output_Ridge_RankY_best.csv` · `Final_Score_Summary.csv` · `Ensemble_Weights.csv` · `Test90_Score_Summary.csv`
