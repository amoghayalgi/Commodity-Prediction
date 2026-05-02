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

