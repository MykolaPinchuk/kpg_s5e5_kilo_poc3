# Code Mode Tasks

This document outlines the specific tasks that the Code mode should implement to build the baseline model for predicting calories burned during workouts.

## Task List

### Task 1: Data Loading and Exploration
- Load `data/train_subsample.csv` and `data/test_subsample.csv`
- Perform basic data exploration (shape, data types, missing values)
- Examine the distribution of the target variable (Calories)

### Task 2: Data Preprocessing
- Handle categorical variables (Gender) using one-hot encoding
- Check for and handle any missing values
- Separate features and target variables
- Apply log transformation to the target variable for RMSLE optimization

### Task 3: Feature Engineering
- Calculate BMI: Weight / (Height/100)^2
- Create interaction features:
  - Weight * Duration
  - Heart_Rate * Duration
  - Age * Weight
- Calculate MET approximation: Calories / (Weight * Duration) * 60

### Task 4: Data Splitting
- Split training data into train/validation sets (80/20 split)
- Ensure reproducibility with random state

### Task 5: Model Initialization
- Initialize XGBoost regressor with optimized parameters:
  - `n_estimators=1000`, `max_depth=6`, `learning_rate=0.1`
  - Enable early stopping with `early_stopping_rounds=10`
  - Use `reg:squarederror` objective for regression

### Task 6: Model Training
- Train the model on the training set
- Use log-transformed target variable
- Implement early stopping using validation set

### Task 7: Model Evaluation
- Implement robust RMSLE calculation function with clipping to avoid log(0) errors
- Predict on validation set
- Apply inverse log transformation to predictions
- Calculate and report RMSLE score

### Task 8: Feature Importance Analysis (Optional)
- Use SHAP values to understand feature contributions to predictions

### Task 9: Hyperparameter Tuning (If Needed)
- If baseline model doesn't achieve 0.1 RMSLE, perform basic hyperparameter tuning
- Focus on key parameters like `n_estimators`, `max_depth`, `learning_rate`
- Use grid search or random search

### Task 10: Final Model Training
- Train final model on the full training dataset
- Use all available data with same preprocessing and feature engineering

### Task 11: Prediction and Submission
- Generate predictions on test data
- Apply inverse log transformation
- Format predictions according to submission requirements
- Save predictions to `submission.csv`

## Implementation Requirements

### Libraries to Use
- pandas for data manipulation
- numpy for numerical operations
- scikit-learn for preprocessing and metrics
- xgboost for the regression model
- shap for feature importance analysis (optional)

### Evaluation Metric
- Root Mean Squared Logarithmic Error (RMSLE)
- Implementation must be correct and match Kaggle's evaluation
- Include clipping to avoid log(0) errors

### Code Quality
- Well-documented with comments
- Clear variable names
- Modular structure where appropriate
- Error handling for common issues

## Expected Outputs

1. A trained model that can predict calories burned
2. Validation RMSLE score printed to console
3. A submission.csv file with predictions for test data
4. Clean, readable code that can be easily understood and modified
5. Feature importance analysis (optional)