# Enhanced XGBoost Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing the enhanced XGBoost model for calories burned prediction. The implementation follows a phased approach, prioritizing high-impact improvements first.

## Phase 1: Enhanced Feature Engineering

### Step 1: Implement Physiologically Accurate MET Calculation

```python
def calculate_physiological_met(df):
    """
    Calculate physiologically accurate MET values
    """
    df = df.copy()
    
    # Calculate BMR using Harris-Benedict equation
    # For males: BMR = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    # For females: BMR = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    
    is_male = df['Gender_male'] if 'Gender_male' in df.columns else (df['Sex'] == 'male')
    
    df['BMR'] = np.where(
        is_male,
        88.362 + (13.397 * df['Weight']) + (4.799 * df['Height']) - (5.677 * df['Age']),
        447.593 + (9.247 * df['Weight']) + (3.098 * df['Height']) - (4.330 * df['Age'])
    )
    
    # Calculate heart rate zone
    df['Max_HR'] = 220 - df['Age']
    df['HR_Zone'] = df['Heart_Rate'] / df['Max_HR']
    
    # Activity-specific base MET values
    df['Base_MET'] = np.select(
        [df['Duration'] < 10, df['Duration'] < 30],
        [3.0, 5.0],
        default=7.0
    )
    
    # Adjust for heart rate zone
    df['HR_Adjustment'] = np.where(
        df['HR_Zone'] > 0.5,
        1.0 + (df['HR_Zone'] - 0.5) * 0.5,
        1.0
    )
    
    # Adjust for body temperature
    df['Temp_Adjustment'] = np.where(
        df['Body_Temp'] > 37.0,
        1.0 + (df['Body_Temp'] - 37.0) * 0.02,
        1.0
    )
    
    # Calculate final physiological MET
    df['Physio_MET'] = df['Base_MET'] * df['HR_Adjustment'] * df['Temp_Adjustment']
    
    return df
```

### Step 2: Add New Interaction Features

```python
def add_enhanced_features(df):
    """
    Add enhanced interaction features
    """
    df = df.copy()
    
    # Existing features from baseline
    df['BMI'] = df['Weight'] / ((df['Height']/100) ** 2)
    df['Weight_Duration'] = df['Weight'] * df['Duration']
    df['HeartRate_Duration'] = df['Heart_Rate'] * df['Duration']
    df['Age_Weight'] = df['Age'] * df['Weight']
    
    # New interaction features
    df['HR_per_min'] = df['Heart_Rate'] / df['Duration']
    df['Weight_per_min'] = df['Weight'] / df['Duration']
    df['HeartRate_BMI'] = df['Heart_Rate'] * df['BMI']
    
    # Physiological MET
    df = calculate_physiological_met(df)
    
    return df
```

### Step 3: Add Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

def add_polynomial_features(X, degree=2, feature_names=None):
    """
    Add polynomial features to the dataset
    """
    # Select only numerical features for polynomial transformation
    numerical_features = X.select_dtypes(include=[np.number]).columns
    
    # Initialize PolynomialFeatures
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    
    # Fit and transform the selected features
    X_poly = poly.fit_transform(X[numerical_features])
    
    # Get feature names
    if feature_names is None:
        feature_names = numerical_features
    
    poly_feature_names = poly.get_feature_names_out(feature_names)
    
    # Create DataFrame with polynomial features
    X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)
    
    # Combine with original features (excluding numerical ones that were transformed)
    X_combined = pd.concat([
        X.drop(columns=numerical_features),
        X_poly_df
    ], axis=1)
    
    return X_combined
```

## Phase 2: Native RMSLE Objective Implementation

### Step 1: Custom RMSLE Objective Function

```python
def rmsle_objective(y_pred, y_true):
    """
    Custom RMSLE objective function for XGBoost
    """
    # Convert to numpy arrays if needed
    if hasattr(y_true, 'get_label'):
        y_true = y_true.get_label()
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Calculate gradient and hessian for RMSLE
    grad = (np.log1p(y_pred) - np.log1p(y_true)) / (y_pred + 1)
    hess = (1 / (y_pred + 1)) * (1 - (np.log1p(y_pred) - np.log1p(y_true)) / (y_pred + 1))
    
    return grad, hess

def rmsle_eval(y_pred, y_true):
    """
    Custom RMSLE evaluation function for XGBoost
    """
    if hasattr(y_true, 'get_label'):
        y_true = y_true.get_label()
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    rmsle_value = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))
    return 'RMSLE', rmsle_value
```

### Step 2: Model Configuration with Native RMSLE

```python
# Initialize XGBoost with custom objective
model = xgb.XGBRegressor(
    objective=rmsle_objective,
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
```

## Phase 3: Enhanced Hyperparameter Optimization

### Step 1: Bayesian Optimization Setup

```python
import optuna

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization
    """
    # Define search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'random_state': 42
    }
    
    # Create model with suggested parameters
    model = xgb.XGBRegressor(**params)
    
    # Perform cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_log_error')
    rmsle_scores = np.sqrt(-scores)
    
    return np.mean(rmsle_scores)

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

## Phase 4: Outlier Handling

### Step 1: Outlier Detection

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def detect_outliers(X, method='isolation_forest'):
    """
    Detect outliers using various methods
    """
    if method == 'isolation_forest':
        detector = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = detector.fit_predict(X)
    elif method == 'local_outlier_factor':
        detector = LocalOutlierFactor(contamination=0.1)
        outlier_labels = detector.fit_predict(X)
    else:
        # Z-score method
        z_scores = np.abs((X - X.mean()) / X.std())
        outlier_labels = (z_scores < 3).all(axis=1).astype(int)
        outlier_labels = np.where(outlier_labels == 0, -1, 1)
    
    return outlier_labels == 1  # True for inliers, False for outliers
```

### Step 2: Outlier Handling

```python
def handle_outliers(X, y, method='winsorize'):
    """
    Handle outliers in the dataset
    """
    if method == 'winsorize':
        # Cap extreme values at percentiles
        for col in X.select_dtypes(include=[np.number]).columns:
            lower_percentile = X[col].quantile(0.01)
            upper_percentile = X[col].quantile(0.99)
            X[col] = np.clip(X[col], lower_percentile, upper_percentile)
    elif method == 'remove':
        # Remove outliers
        inliers = detect_outliers(X, method='isolation_forest')
        X = X[inliers]
        y = y[inliers]
    
    return X, y
```

## Implementation Workflow

### 1. Data Loading and Preprocessing

```python
# Load data
train_data = pd.read_csv('data/train_subsample.csv')
test_data = pd.read_csv('data/test_subsample.csv')

# Handle categorical variables
train_data = pd.get_dummies(train_data, columns=['Sex'], prefix='Gender')
test_data = pd.get_dummies(test_data, columns=['Sex'], prefix='Gender')

# Apply enhanced feature engineering
train_data = add_enhanced_features(train_data)
test_data = add_enhanced_features(test_data)

# Separate features and target
feature_cols = [col for col in train_data.columns if col not in ['Calories', 'id']]
X = train_data[feature_cols]
y = train_data['Calories']

# Handle outliers
X, y = handle_outliers(X, y, method='winsorize')

# Add polynomial features (optional)
# X = add_polynomial_features(X)
```

### 2. Model Training with Native RMSLE

```python
# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model with custom objective
model = xgb.XGBRegressor(
    objective=rmsle_objective,
    feval=rmsle_eval,
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1,
    reg_lambda=1,
    random_state=42
)

# Train model
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False
)
```

### 3. Evaluation

```python
# Predict on validation set
y_val_pred = model.predict(X_val)

# Calculate RMSLE
val_rmsle = rmsle(y_val, y_val_pred)
print(f"Validation RMSLE: {val_rmsle:.6f}")
```

## Expected Improvements

By implementing these enhancements in order:

1. **Phase 1 (Feature Engineering)**: 10-15% improvement in RMSLE
2. **Phase 2 (Native RMSLE)**: 5-10% improvement in RMSLE
3. **Phase 3 (Hyperparameter Optimization)**: 5-10% improvement in RMSLE
4. **Phase 4 (Regularization & Outlier Handling)**: 5% improvement in RMSLE

**Overall Expected Improvement**: 25-50% reduction in RMSLE from the current 0.061888 to approximately 0.03-0.046.