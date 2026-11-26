
# House-Price-Predictor
Ames house price prediction: leak-free scikit-learn pipeline, 8 models compared with 5-fold CV, 5 tuned. Tuned Ridge wins (CV RMSE 0.113 on log price).  Topics: machine-learning, regression, scikit-learn, xgboost, house-price-prediction, data-science, python, jupyter-notebook
=======
# House Price Prediction (Ames, Iowa)

An end-to-end regression project that predicts house sale prices from 79 property features. I built a leak-free scikit-learn pipeline, compared 8 regression models with 5-fold cross-validation, tuned the 5 strongest, and chose the final model by cross-validated error rather than by a single lucky validation split.

**Main finding:** a tuned, regularized linear model (Ridge) matched or beat gradient boosting on this dataset, with a cross-validated RMSE of about 0.113 on log price (roughly 11 to 12% typical error).

> Tech: Python, pandas, NumPy, scikit-learn, XGBoost, SciPy, Matplotlib, joblib

---

## Table of contents
1. [Problem and dataset](#problem-and-dataset)
2. [Approach](#approach)
3. [Results](#results)
4. [What drives price](#what-drives-price)
5. [Why linear models won here](#why-linear-models-won-here)
6. [Limitations](#limitations)
7. [Repository structure](#repository-structure)
8. [How to run](#how-to-run)
9. [Using the saved model](#using-the-saved-model)
10. [Future work](#future-work)
11. [Acknowledgements](#acknowledgements)

---

## Problem and dataset

**Goal:** predict `SalePrice` for residential homes in Ames, Iowa.

**Data:** the Kaggle competition *House Prices - Advanced Regression Techniques* (`train.csv`, 1,460 houses, 79 features describing size, quality, location, basement, garage, and sale details). The competition test file has no labels, so I made my own 80/20 train/validation split. The data is not included in this repository: download it from Kaggle (link under [How to run](#how-to-run)).

**Target:** I model `log(SalePrice)`, because prices are right-skewed and errors in log space correspond to percentage errors.

**Main metric:** RMSE on log price (the competition metric). I also report errors in dollars.

---

## Approach

### 1. Data cleaning
- In this dataset, a missing value in columns such as `Alley`, `Fence`, `PoolQC`, `FireplaceQu`, `Garage*`, `Bsmt*`, `MiscFeature`, and `MasVnrType` means the feature is *absent*, not unknown. I filled these with `'None'` instead of imputing the most common value, which would have given houses without a garage a garage quality.
- `GarageYrBlt` is set to 0 when there is no garage, and `MSSubClass` is treated as a category (it is a code, not a quantity).
- Removed 2 extreme outliers from the training data (`GrLivArea > 4000` with a low sale price).
- Dropped the `Id` column and near-constant categorical columns (more than 95% one value), plus the pool-area columns, which only apply to a handful of houses and mainly added noise.

### 2. Feature engineering
Combined related columns into features with a clear real-world meaning, for example total square footage, total bathrooms, house age and years since remodel (clipped at zero), and has-garage / has-basement / has-fireplace flags. See the notebook for the full list.

### 3. Preprocessing pipeline (no data leakage)
Everything below lives inside a scikit-learn `ColumnTransformer` + `Pipeline`, so every transformation is learned from training data only, including inside each cross-validation fold:
- **Skewed numeric features** (absolute skew above 0.75): median imputation, `log1p`, standard scaling
- **Other numeric features:** median imputation, standard scaling
- **Quality columns** (`ExterQual`, `BsmtQual`, `KitchenQual`, `HeatingQC`, `FireplaceQu`): ordinal encoding (None < Po < Fa < TA < Gd < Ex)
- **Nominal categoricals:** one-hot encoding, with categories seen fewer than 20 times grouped into one "infrequent" column

### 4. Validation strategy
- 80/20 train/validation split (1,166 and 292 houses, after outlier removal), fixed random seed
- All model comparison and tuning uses **5-fold cross-validation on the training portion only**
- The validation set is used once, at the end, for the final model

### 5. Models
Eight models were compared: Linear Regression, Ridge, Lasso, ElasticNet, Support Vector Regression, Decision Tree, Random Forest, and XGBoost. The five strongest were tuned: Ridge, Lasso, and ElasticNet with grid search, and SVR and XGBoost with randomized search (20 and 25 iterations).

---

## Results

### Cross-validation, before and after tuning
5-fold CV, RMSE on log price (lower is better). Fold-to-fold standard deviation is about 0.008.

| Model | Untuned | Tuned | Improvement |
|---|---|---|---|
| **Ridge** | 0.1167 | **0.1134** | 0.0034 |
| Lasso | 0.1145 | 0.1139 | 0.0006 |
| ElasticNet | 0.1256 | 0.1140 | 0.0115 |
| SVR | 0.1183 | 0.1157 | 0.0026 |
| XGBoost | 0.1238 | 0.1164 | 0.0074 |
| Linear Regression | 0.1169 | not tuned | |
| Random Forest | 0.1405 | not tuned | |
| Decision Tree | 0.1963 | not tuned | |

**Selected model: Ridge** (`alpha` around 20 to 25). After tuning, all five models land within 0.003 of each other, which is smaller than the fold-to-fold noise, so I treated it as a tie and chose the simplest model that was also best on CV.

### Final evaluation on the held-out validation set (292 houses)

| Metric | Value |
|---|---|
| RMSE (log price) | 0.11862811201536988 |
| RMSE ($) | 19470.109822082795 |
| MAE ($) | 14111.510133727748 |
| R² | 0.9165218868356617 |



I also checked the residual plot and the 10 largest prediction errors. **TODO: add 2 to 3 sentences about what the worst predictions have in common** (for example neighborhood, overall quality, or sale condition).

---

## What drives price

Largest Ridge coefficients (features are standardized and the target is log price, so a coefficient of 0.06 means roughly +6% price for a one standard deviation increase):

| Feature | Coefficient |
|---|---|
| Above-ground living area (`GrLivArea`, log) | +0.065 |
| Overall quality (`OverallQual`) | +0.061 |
| Neighborhood: Stone Brook | +0.059 |
| Neighborhood: Crawford | +0.056 |
| Exterior: brick face | +0.047 |
| Typical home functionality | +0.046 |
| Overall condition (`OverallCond`) | +0.045 |
| Newly built sale (`SaleType_New`) | +0.038 |
| Neighborhood: Gilbert | -0.038 |
| Total square footage (`TotalSF`, log) | +0.037 |

Living area and build quality dominate, followed by neighborhood.

---

## Why linear models won here

- Log price is close to an additive combination of features such as size, quality, and neighborhood, and linear models capture additive effects directly.
- The dataset is small (about 1,170 training rows), which favors models with strong built-in assumptions and regularization. Tree-based models need more data to learn the same relationships.
- Hand-set hyperparameters made some models look worse than they were: tuning improved ElasticNet and XGBoost the most, but did not change the ranking much.

---

## Limitations

- One city (Ames, Iowa), sales from 2006 to 2010, with no inflation adjustment, so the model is not a general-purpose price estimator.
- Small dataset, so differences between the top models are within noise.
- The skew and near-constant-column checks used the full training set before the split. This is an unsupervised decision and a small source of leakage.
- No time-based validation: all years are mixed in the split.


Dataset: Ames Housing data compiled by Dean De Cock (2011), distributed through the Kaggle competition *House Prices - Advanced Regression Techniques*.
>>>>>>> 3ed6a9e (Added house price prediction project)
