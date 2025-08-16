# Corrected XGBoost Model Approach for Calories Burned Prediction

## Executive Summary

This document outlines a corrected XGBoost model approach for predicting calories burned during workouts, addressing the issues that caused the enhanced model to perform poorly (RMSLE: 0.15) compared to the baseline (RMSLE: 0.062). The corrections focus on five key areas:

1. Fixed RMSLE implementation issues (removed target log transformation, simplified objective)
2. Simplified feature engineering to core features only (BMI, Weight×Duration, Heart_Rate×Duration)
3. Implemented faster hyperparameter optimization (100+ trials with full dataset)
4. Corrected outlier handling (apply after feature engineering, use RobustScaler)
5. Fixed validation inconsistencies (consistent log-space evaluation)

These corrections will result in a streamlined approach that maintains performance while enabling rapid iteration.

## 1. Root Cause Analysis of Enhanced Model Issues

### 1.1 Mathematical Inconsistency Between Custom RMSLE Objective and Log Transformation
- The enhanced model used a custom RMSLE objective function but also applied log transformation to the target variable
- This created a mathematical inconsistency where the objective was optimizing for RMSLE on log-transformed values
- The evaluation was done in the original space, leading to suboptimal parameter optimization

### 1.2 Overengineered Features Introduced Noise
- Complex feature engineering with physiologically accurate MET calculation introduced noise
- Additional interaction features and polynomial terms increased dimensionality without proportional benefit
- Overfitting occurred due to the complex feature set

### 1.3 Problems with Hyperparameter Optimization Process
- Only 3 trials were used for hyperparameter optimization instead of 100+
- Optimization was performed on a 10% subset of data instead of the full dataset
- This led to suboptimal parameter selection

### 1.4 Issues with Outlier Handling
- Outlier handling was applied before feature engineering, which may not be optimal
- Simple winsorization was used without considering the impact on complex features

### 1.5 Validation Inconsistencies
- The custom RMSLE objective didn't align with the evaluation function
- Inconsistent handling of log transformations between training and evaluation

## 2. Corrected Approach Design

### 2.1 Simplified RMSLE Implementation
- Remove target log transformation that was being used with the custom RMSLE objective
- Use raw target values with a consistent RMSLE objective function
- Ensure the evaluation is done in the same space as the optimization

#### Implementation:
```python
# Custom RMSLE objective function for XGBoost
def rmsle_objective(y_pred, y_true):
    """
    Custom objective function for RMSLE
    """
    # Calculate gradients and hessians for RMSLE with raw values
    grad = (np.log1p(y_pred) - np.log1p(y_true)) / (y_pred + 1)
    hess = (1 / (y_pred + 1)) * (1 - (np.log1p(y_pred) - np.log1p(y_true)) / (y_pred + 1))
    return grad, hess

# Custom RMSLE evaluation function for XGBoost
def rmsle_eval(y_pred, y_true):
    """
    Custom evaluation function for RMSLE
    """
    y_true_values = y_true.get_label()
    # Calculate RMSLE with raw values
    rmsle_value = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true_values))**2))
    return 'RMSLE', rmsle_value
```

### 2.2 Simplified Feature Engineering
Focus only on core features that have clear physiological relevance:
1. **BMI**: Weight / (Height/100)^2
2. **Weight×Duration**: Weight * Duration
3. **Heart_Rate×Duration**: Heart_Rate * Duration

#### Implementation:
```python
def add_core_features(df):
    """
    Add only core features to the dataset
    """
    df = df.copy()
    
    # Calculate BMI: Weight / (Height/100)^2
    df['BMI'] = df['Weight'] / ((df['Height']/100) ** 2)
    
    # Create core interaction features
    df['Weight_Duration'] = df['Weight'] * df['Duration']
    df['HeartRate_Duration'] = df['Heart_Rate'] * df['Duration']
    
    return df
```

### 2.3 Faster Hyperparameter Optimization
- Use the full dataset instead of a subset for hyperparameter optimization
- Increase the number of trials to 100+ for better exploration of the parameter space
- Use efficient optimization algorithms to keep the process reasonably fast
- Implement early stopping to prevent unnecessary computation

#### Implementation:
```python
def optimize_hyperparameters(X, y, n_trials=100):
    """
    Optimize hyperparameters using Optuna with full dataset
    """
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',  # Use built-in objective for simplicity
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'random_state': 42
        }
        
        # Create model with trial parameters
        model = xgb.XGBRegressor(**params)
        
        # Use cross-validation to evaluate
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_log_error')
        return -scores.mean()  # Return negative because Optuna minimizes
    
    # Create study and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params
```

### 2.4 Corrected Outlier Handling
- Apply outlier handling AFTER feature engineering to ensure consistency
- Use RobustScaler instead of standard scaling methods
- Implement a more robust outlier detection method that considers the feature engineering transformations

#### Implementation:
```python
from sklearn.preprocessing import RobustScaler

def handle_outliers(df, columns):
    """
    Handle outliers using RobustScaler after feature engineering
    """
    df = df.copy()
    
    # Apply RobustScaler to handle outliers
    scaler = RobustScaler()
    df[columns] = scaler.fit_transform(df[columns])
    
    return df, scaler

# Alternative outlier handling approach
def detect_and_handle_outliers(df, columns):
    """
    Detect and handle outliers using IQR method after feature engineering
    """
    df = df.copy()
    
    # Detect outliers using IQR method
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing them
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df
```

### 2.5 Consistent Validation Approach
- Ensure that the evaluation metric is consistently computed in log space
- Make sure the custom RMSLE objective aligns with the evaluation function
- Validate that predictions are properly handled during evaluation
- Implement consistent cross-validation that maintains the log-space evaluation

#### Implementation:
```python
# Function to calculate RMSLE with clipping to avoid log(0) errors
def rmsle(y_true, y_pred):
    """
    Calculate Root Mean Squared Logarithmic Error (RMSLE)
    """
    # Clip predictions to avoid log(0)
    y_pred_clipped = np.clip(y_pred, 1e-15, None)
    y_true_clipped = np.clip(y_true, 1e-15, None)
    return np.sqrt(mean_squared_log_error(y_true_clipped, y_pred_clipped))

# Cross-validation with consistent RMSLE evaluation
def cross_validate_model(model, X, y, cv=5):
    """
    Cross-validate model with consistent RMSLE evaluation
    """
    rmsle_scores = []
    
    for train_idx, val_idx in KFold(n_splits=cv, shuffle=True, random_state=42).split(X):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model.fit(X_train_fold, y_train_fold)
        
        # Predict and evaluate
        y_pred_fold = model.predict(X_val_fold)
        fold_rmsle = rmsle(y_val_fold, y_pred_fold)
        rmsle_scores.append(fold_rmsle)
    
    return np.array(rmsle_scores)
```

## 3. Implementation Plan

### 3.1 Phase 1: Data Preprocessing and Feature Engineering
1. Load and explore data
2. Handle categorical variables (Gender) using one-hot encoding
3. Apply simplified feature engineering (BMI, Weight×Duration, Heart_Rate×Duration)
4. Apply corrected outlier handling after feature engineering

### 3.2 Phase 2: Model Development with Corrected RMSLE
1. Implement simplified RMSLE objective and evaluation functions
2. Initialize XGBoost regressor with corrected approach
3. Train model with early stopping

### 3.3 Phase 3: Hyperparameter Optimization
1. Set up hyperparameter optimization with 100+ trials on full dataset
2. Use efficient optimization algorithm (Optuna)
3. Validate best parameters

### 3.4 Phase 4: Model Evaluation and Validation
1. Perform cross-validation with consistent RMSLE evaluation
2. Evaluate feature importance
3. Validate performance against baseline

### 3.5 Phase 5: Final Model Training and Prediction
1. Train final model on full dataset with optimized parameters
2. Generate predictions on test data
3. Create submission files

## 4. Expected Improvements and Benefits

### 4.1 Performance Improvements
- **RMSLE Performance**: Expected to achieve similar or better performance than baseline (RMSLE ≤ 0.062)
- **Reduced Overfitting**: Simplified features and proper regularization will reduce overfitting
- **Better Generalization**: Consistent validation approach will improve generalization

### 4.2 Efficiency Improvements
- **Faster Training**: Simplified feature set will reduce training time
- **Faster Hyperparameter Optimization**: Efficient optimization with full dataset will find better parameters faster
- **Streamlined Process**: Reduced complexity will make the process more maintainable

### 4.3 Robustness Improvements
- **Better Outlier Handling**: RobustScaler and post-feature-engineering outlier handling will improve robustness
- **Consistent Evaluation**: Aligned objective and evaluation functions will provide more reliable results
- **Simplified Interpretability**: Core features are more interpretable than complex engineered features

## 5. Risk Mitigation

### 5.1 Overfitting Prevention
- Use cross-validation for all evaluations
- Implement early stopping with patience
- Monitor train/validation metric divergence
- Use appropriate regularization parameters

### 5.2 Computational Complexity
- Limit feature engineering to core variables only
- Use efficient hyperparameter optimization algorithms
- Implement parallel processing where possible

### 5.3 Implementation Validation
- Maintain backward compatibility with baseline model
- Implement comprehensive testing for new features
- Validate results at each enhancement step

## 6. Success Metrics

### 6.1 Primary Metrics
- **Validation RMSLE**: ≤ 0.062 (at least as good as baseline)
- **Test RMSLE**: Consistent performance with validation

### 6.2 Secondary Metrics
- **Training Time**: Reduced compared to enhanced model
- **Feature Importance**: More balanced distribution across core features
- **Cross-Validation Consistency**: Low variance across folds
- **Robustness**: Stable performance with different data splits

## 7. Timeline

### 7.1 Implementation Phases
1. **Phase 1**: Data preprocessing and feature engineering (1 day)
2. **Phase 2**: Model development with corrected RMSLE (1 day)
3. **Phase 3**: Hyperparameter optimization (2-3 days)
4. **Phase 4**: Model evaluation and validation (1 day)
5. **Phase 5**: Final model training and prediction (1 day)

### 7.2 Total Estimated Time
- **Total Time**: 6-7 days for full implementation and validation
- **Time Savings**: Significantly faster than the enhanced approach due to streamlined process

## 8. Conclusion

The corrected XGBoost model approach addresses all the issues that caused the enhanced model to perform poorly while maintaining or improving performance. By focusing on a streamlined approach with:

1. Corrected RMSLE implementation
2. Simplified feature engineering
3. Efficient hyperparameter optimization
4. Proper outlier handling
5. Consistent validation

This approach will achieve the target performance while being more efficient and maintainable than the overengineered enhanced model. The focus on core features and proper mathematical consistency will result in a robust model that generalizes well to new data.