# Implementation Guide for Code Mode

This guide provides a clear, step-by-step implementation plan for the Code mode to build the baseline model for predicting calories burned during workouts.

## Overview

The implementation will follow these phases:
1. Data loading and exploration
2. Data preprocessing
3. Feature engineering
4. Model development with XGBoost
5. Model evaluation using RMSLE
6. Final predictions and submission

## Detailed Implementation Steps

### Phase 1: Environment Setup and Data Loading

1. Import required libraries:
   - pandas for data manipulation
   - numpy for numerical operations
   - sklearn for preprocessing and metrics
   - xgboost for the regression model
   - shap for feature importance analysis (optional)

2. Load the data files:
   - `data/train_subsample.csv` for training
   - `data/test_subsample.csv` for testing

### Phase 2: Data Preprocessing

1. Handle the categorical variable (Gender):
   - Use one-hot encoding to convert Gender into numerical features

2. Prepare features and target:
   - Separate features (Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp)
   - Extract target variable (Calories)

3. Apply target transformation:
   - Apply log transformation to Calories for RMSLE optimization: `y_log = log(1 + y)`

### Phase 3: Feature Engineering

1. Calculate BMI: Weight / (Height/100)^2
2. Create interaction features:
   - Weight * Duration
   - Heart_Rate * Duration
   - Age * Weight
3. Calculate MET approximation: Calories / (Weight * Duration) * 60 (metabolic equivalent)

### Phase 4: Data Splitting

1. Split data for validation:
   - Create an 80/20 train/validation split
   - Ensure reproducibility with random state

### Phase 5: Model Initialization

1. Initialize XGBoost regressor with optimized defaults:
   - Use `XGBRegressor` with `reg:squarederror` objective
   - Set `n_estimators=1000`, `max_depth=6`, `learning_rate=0.1`
   - Enable early stopping with `early_stopping_rounds=10`

### Phase 6: Model Training

1. Train the model on the training set:
   - Use log-transformed target variable
   - Implement early stopping using validation set

### Phase 7: Model Evaluation

1. Implement robust RMSLE calculation:
   - Create a function to compute RMSLE between true and predicted values
   - Add clipping to avoid log(0) errors

2. Validate model:
   - Predict on validation set
   - Apply inverse log transformation to predictions: `y_pred = exp(y_pred_log) - 1`
   - Calculate and report RMSLE score

3. Feature importance analysis (optional):
   - Use SHAP values to understand feature contributions

### Phase 8: Hyperparameter Tuning (If Needed)

1. If baseline model doesn't achieve 0.1 RMSLE, perform basic hyperparameter tuning:
   - Focus on key parameters like `n_estimators`, `max_depth`, `learning_rate`
   - Use grid search or random search

### Phase 9: Final Model Training

1. Train final model on the full training dataset:
   - Use all available data
   - Apply same preprocessing and feature engineering

### Phase 10: Prediction and Submission

1. Generate predictions:
   - Predict on test data
   - Apply inverse log transformation

2. Create submission file:
   - Format predictions in required submission format
   - Save to `submission.csv`

## Expected File Structure

```
project/
├── data/
│   ├── train_subsample.csv
│   └── test_subsample.csv
├── baseline_model_approach.md
├── implementation_guide.md
└── calories_model.py (to be created by Code mode)
```

## Success Criteria

- RMSLE score ≤ 0.1 on validation set
- Clean, readable, and well-documented code
- Proper handling of all features and data preprocessing steps
- Correct implementation of the RMSLE evaluation metric
- Implementation of feature engineering for improved performance