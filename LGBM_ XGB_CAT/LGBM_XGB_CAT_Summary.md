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
