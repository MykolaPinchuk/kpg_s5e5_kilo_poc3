# Baseline Model Approach for Calories Burned Prediction

## 1. Problem Analysis

### Problem Requirements
- **Task**: Regression problem to predict calories burned during workouts
- **Target Variable**: Calories (continuous value)
- **Evaluation Metric**: Root Mean Squared Logarithmic Error (RMSLE)
- **Goal**: Achieve at least 0.1 RMSLE

### Features
1. **Gender**: Categorical variable (Male/Female)
2. **Age**: Numerical variable (years)
3. **Height**: Numerical variable (cm)
4. **Weight**: Numerical variable (kg)
5. **Duration**: Numerical variable (minutes)
6. **Heart_Rate**: Numerical variable (beats per minute)
7. **Body_Temp**: Numerical variable (Celsius)

## 2. Data Preprocessing Approach

### Handling Categorical Variables
- **Gender**: One-hot encode using pandas get_dummies or sklearn's OneHotEncoder
- Create separate binary columns for each gender category

### Data Cleaning
- Check for missing values in all features
- Handle outliers if any are detected
- Ensure data types are correct for all features

### Data Splitting
- Split training data into train/validation sets (80/20 split)
- Use stratified sampling if needed to maintain distribution

## 3. Model Selection

### Choice: XGBoost Regressor
- **Rationale**: 
  - As per instructions, prefer XGBoost over Random Forest
  - XGBoost is known for its performance on tabular data
  - Handles non-linear relationships well
  - Robust to outliers
  - Good default parameters often work well
  - Built-in cross-validation capabilities
  - Supports early stopping to prevent overfitting

### Model Configuration
- Use XGBoost's native regression objective: `reg:squarederror`
- For RMSLE optimization, we'll apply log transformation to target variable
- Enable early stopping to prevent overfitting

## 4. Evaluation Approach

### RMSLE Calculation
- Root Mean Squared Logarithmic Error formula:
  ```
  RMSLE = sqrt(mean((log(1 + y_pred) - log(1 + y_true))^2))
  ```
- Implementation using sklearn or custom function
- Include clipping to avoid log(0) errors

### Cross-Validation
- Use 5-fold cross-validation to ensure robust evaluation
- StratifiedKFold to maintain distribution across folds

### Validation Strategy
- Hold out validation set for final model evaluation
- Track both training and validation RMSLE scores
- Implement early stopping based on validation performance

## 5. Feature Engineering Considerations

### Basic Features
- All original features as provided

### Derived Features
- **BMI**: Weight / (Height/100)^2 (Body Mass Index)
- **MET approximation**: Calories / (Weight * Duration) * 60 (metabolic equivalent)
- **Heart rate zones**: Categorize heart rate into zones (optional)
- **Interaction terms**: 
  - Weight * Duration
  - Heart_Rate * Duration
  - Age * Weight

### Feature Scaling
- Not strictly necessary for XGBoost but can help with convergence
- Consider standardization for features with very different scales

## 6. Implementation Plan for Code Mode

### Phase 1: Data Loading and Exploration
1. Load train_subsample.csv and test_subsample.csv
2. Perform basic EDA (shape, data types, missing values)
3. Check target distribution (Calories)

### Phase 2: Data Preprocessing
1. Handle categorical variables (Gender)
2. Check and handle missing values
3. Create derived features (BMI, interaction terms, MET)
4. Split data into train/validation sets

### Phase 3: Model Development
1. Initialize XGBoost regressor with optimized defaults
2. Apply log transformation to target variable for RMSLE optimization
3. Train model on training set with early stopping
4. Validate on validation set using RMSLE metric

### Phase 4: Model Evaluation and Tuning
1. Perform cross-validation
2. Evaluate feature importance using SHAP values
3. Basic hyperparameter tuning if needed
4. Validate RMSLE score meets 0.1 target

### Phase 5: Prediction and Submission
1. Train final model on full training set
2. Generate predictions on test_subsample.csv
3. Format predictions for submission

## 7. Expected Challenges and Solutions

### Potential Issues
1. **Data quality**: Missing values or outliers
   - Solution: Implement robust data validation and cleaning
2. **Overfitting**: Model may overfit to training data
   - Solution: Use XGBoost's built-in regularization parameters and early stopping
3. **Target distribution**: Calories may have a skewed distribution
   - Solution: Apply log transformation to target variable
4. **Log(0) errors**: Potential issues with RMSLE calculation
   - Solution: Add clipping to avoid log(0) errors

## 8. Success Criteria
- Achieve RMSLE ≤ 0.1 on validation set
- Model generalizes well to test data
- Implementation is simple and maintainable
- Feature importance analysis provides insights

## 9. Simple Implementation Steps for Code Mode

### Step 1: Import Required Libraries
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb
```

### Step 2: Load and Explore Data
```python
# Load data
train_data = pd.read_csv('data/train_subsample.csv')
test_data = pd.read_csv('data/test_subsample.csv')

# Basic exploration
print(train_data.shape)
print(train_data.info())
print(train_data.describe())
```

### Step 3: Feature Engineering
```python
# Function to add derived features
def add_features(df):
    df = df.copy()
    df['BMI'] = df['Weight'] / ((df['Height']/100) ** 2)
    df['Weight_Duration'] = df['Weight'] * df['Duration']
    df['HeartRate_Duration'] = df['Heart_Rate'] * df['Duration']
    df['Age_Weight'] = df['Age'] * df['Weight']
    # MET approximation (for feature creation, not target leakage)
    df['MET'] = df['Heart_Rate'] / df['Age']  # Simplified approximation
    return df

# Apply feature engineering
train_data = add_features(train_data)
test_data = add_features(test_data)
```

### Step 4: Data Preprocessing
```python
# Handle categorical variables
train_data = pd.get_dummies(train_data, columns=['Gender'])
test_data = pd.get_dummies(test_data, columns=['Gender'])

# Separate features and target
feature_cols = [col for col in train_data.columns if col not in ['Calories', 'id']]
X = train_data[feature_cols]
y = train_data['Calories']

# Apply log transformation to target for RMSLE optimization
y_log = np.log1p(y)
```

### Step 5: Train-Validation Split
```python
X_train, X_val, y_train, y_val, y_train_log, y_val_log = train_test_split(
    X, y, y_log, test_size=0.2, random_state=42
)
```

### Step 6: Model Training with Early Stopping
```python
# Initialize XGBoost regressor with optimized defaults
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

# Train model with early stopping
model.fit(
    X_train, y_train_log,
    eval_set=[(X_val, y_val_log)],
    early_stopping_rounds=10,
    verbose=False
)
```

### Step 7: Model Evaluation with Robust RMSLE
```python
# Predict on validation set
y_val_pred_log = model.predict(X_val)
y_val_pred = np.expm1(y_val_pred_log)  # Inverse of log transformation

# Calculate RMSLE with clipping to avoid log(0) errors
def rmsle(y_true, y_pred):
    # Clip predictions to avoid log(0)
    y_pred_clipped = np.clip(y_pred, 1e-15, None)
    y_true_clipped = np.clip(y_true, 1e-15, None)
    return np.sqrt(mean_squared_log_error(y_true_clipped, y_pred_clipped))

val_rmsle = rmsle(y_val, y_val_pred)
print(f"Validation RMSLE: {val_rmsle}")
```

### Step 8: Final Model and Predictions
```python
# Train final model on full dataset
final_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
final_model.fit(X, y_log)

# Prepare test data
X_test = test_data[feature_cols]

# Predict on test set
test_pred_log = final_model.predict(X_test)
test_pred = np.expm1(test_pred_log)

# Create submission
submission = pd.DataFrame({
    'id': test_data['id'],
    'Calories': test_pred
})
submission.to_csv('submission.csv', index=False)