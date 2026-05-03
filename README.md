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


## Ridge Regression — Rank-Aware Ensemble

Evaluated using the official score as the primary metric, applied to truly observed labels only.   

**1_Final model:** `ensemble_optimized` — a validation-optimized weighted ensemble of four Ridge variants:   
`A_no_aug` (raw returns, no augmentation),   
`A_aug` (raw returns, augmented features),   
`B_rank` (rank-transformed targets),   
`B_raw` (raw returns under per-target framework). 

**Ensemble weights** were optimized directly on the official validation score rather than equal-weighted, since weak candidates dragged the average down.   

**Rank-transformed Y:**     
* if the true same-day ordering is  
        target A > target C > target E > target B > target D  
* then a good model should produce predictions with the same ordering, even if the numeric values differ.  

**Addiional Augmented Features**
1. Lag mean
    The average of the past four label lags.
2. Lag standard deviation
    The variability of the past four label lags.
3. Decay-weighted lag mean
    A weighted average of the past lags that assigns higher weight to more recent values.
4. Recent-vs-history gap
    The difference between the most recent lag and the average of earlier lags, capturing short-term momentum or reversal.
5. Sign consistency
    Whether the past lags move in the same direction across periods.

【 Official score: **0.2836**】      

**2_Pipeline:**
- Load and validate train / validation splits  
- Drop unusable columns and fill missing values  
- StandardScaler fitted on train only  
- Construct four Ridge variant candidates under two target definitions  
- Tune alpha using ranking-aligned validation score  
- Generate predictions for each candidate  
- Apply optional cross-sectional post-processing  
- Optimize ensemble weights on validation predictions  
- Export final predictions  

**3_Experiments:**
- **A_no_aug** — Ridge on raw return targets, no feature augmentation. Official score: 0.1982  
- **A_aug** — Ridge on raw return targets with augmented features. Official score: 0.1982 
- **B_rank** — Ridge on daily cross-sectional rank-transformed targets. Instead of predicting raw return values, Y was converted to within-day rank order before training, directly aligning the training objective with the competition ranking metric. Official score: 0.2690
- **B_raw** — Ridge on raw return targets under the new per-target framework. Official score: 0.1700 
- **ensemble_equal** — Equal-weight average of all four candidates. Official score: 0.2110  
- **ensemble_optimized** — Validation-score-optimized weighted ensemble. Official score: **0.2836**  

**4_Results:**

| Model | Official Score |
|---|---|
| ensemble_optimized | **0.2836** |
| B_rank | 0.2690 |
| ensemble_equal | 0.2110 |
| A_no_aug | 0.1982 |
| A_aug | 0.1982 |
| B_raw | 0.1700 |

**5_Key takeaways:**
- Target redesign mattered most — rank-transformed Y produced the largest single gain over raw return regression
- Optimized ensemble weighting was necessary — equal-weight averaging diluted performance
- Feature augmentation had no effect in its current form
- Ridge works best here as a ranking-oriented model, not a value-regression model
- The competition challenge is correct daily cross-sectional ordering, not exact return magnitude

**6_Output files:** `output_Ridge_RankY_best.csv` · `Final_Score_Summary.csv` · `Ensemble_Weights.csv`
