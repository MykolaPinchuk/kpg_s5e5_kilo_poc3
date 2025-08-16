import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
import warnings
import optuna
from sklearn.model_selection import cross_val_score
warnings.filterwarnings('ignore')

# Function to calculate RMSLE with clipping to avoid log(0) errors
def rmsle(y_true, y_pred):
    """
    Calculate Root Mean Squared Logarithmic Error (RMSLE)
    
    Parameters:
    y_true: array-like of shape (n_samples,) - True target values
    y_pred: array-like of shape (n_samples,) - Predicted values
    
    Returns:
    float: RMSLE score
    """
    # Clip predictions to avoid log(0)
    y_pred_clipped = np.clip(y_pred, 1e-15, None)
    y_true_clipped = np.clip(y_true, 1e-15, None)
    return np.sqrt(mean_squared_log_error(y_true_clipped, y_pred_clipped))

# Custom RMSLE objective function for XGBoost
def rmsle_objective(y_true, y_pred):
    """
    Custom objective function for RMSLE
    
    Parameters:
    y_true: array-like - True target values
    y_pred: array-like - Predicted values
    
    Returns:
    tuple: (gradients, hessians)
    """
    # Calculate gradients and hessians for RMSLE
    grad = (np.log1p(y_pred) - np.log1p(y_true)) / (y_pred + 1)
    hess = (1 / (y_pred + 1)) * (1 - (np.log1p(y_pred) - np.log1p(y_true)) / (y_pred + 1))
    return grad, hess

# Custom RMSLE evaluation function for XGBoost
def rmsle_eval(y_pred, y_true):
    """
    Custom evaluation function for RMSLE
    
    Parameters:
    y_pred: array-like - Predicted values
    y_true: xgboost.DMatrix - True target values
    
    Returns:
    tuple: (eval_name, eval_result)
    """
    y_true_values = y_true.get_label()
    # Calculate RMSLE
    rmsle_value = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true_values))**2))
    return 'RMSLE', rmsle_value

# Function to add derived features
def add_features(df, is_train=True):
    """
    Add derived features to the dataset
    
    Parameters:
    df: pandas.DataFrame - Input dataframe
    is_train: bool - Whether this is training data (for storing/calculating statistics)
    
    Returns:
    pandas.DataFrame: Dataframe with added features
    """
    df = df.copy()
    
    # Calculate BMI: Weight / (Height/100)^2
    df['BMI'] = df['Weight'] / ((df['Height']/100) ** 2)
    
    # Create interaction features
    df['Weight_Duration'] = df['Weight'] * df['Duration']
    df['HeartRate_Duration'] = df['Heart_Rate'] * df['Duration']
    df['Age_Weight'] = df['Age'] * df['Weight']
    
    # Additional interaction features
    df['HR_per_min'] = df['Heart_Rate'] / (df['Duration'] + 1e-8)  # Avoid division by zero
    df['Weight_per_min'] = df['Weight'] / (df['Duration'] + 1e-8)  # Avoid division by zero
    df['HeartRate_BMI'] = df['Heart_Rate'] * df['BMI']
    
    # Physiologically accurate MET calculation using BMR (Harris-Benedict equation)
    # For males: BMR = 88.362 + (13.397 * weight in kg) + (4.799 * height in cm) - (5.677 * age in years)
    # For females: BMR = 447.593 + (9.247 * weight in kg) + (3.098 * height in cm) - (4.330 * age in years)
    df['BMR'] = 0
    df.loc[df['Gender_male'] == 1, 'BMR'] = (
        88.362 +
        (13.397 * df.loc[df['Gender_male'] == 1, 'Weight']) +
        (4.799 * df.loc[df['Gender_male'] == 1, 'Height']) -
        (5.677 * df.loc[df['Gender_male'] == 1, 'Age'])
    )
    df.loc[df['Gender_female'] == 1, 'BMR'] = (
        447.593 +
        (9.247 * df.loc[df['Gender_female'] == 1, 'Weight']) +
        (3.098 * df.loc[df['Gender_female'] == 1, 'Height']) -
        (4.330 * df.loc[df['Gender_female'] == 1, 'Age'])
    )
    
    # Heart rate zone calculation (as percentage of maximum heart rate)
    df['Max_HR'] = 220 - df['Age']
    df['HR_Zone'] = df['Heart_Rate'] / df['Max_HR']
    
    # Activity-specific MET values based on duration
    df['Base_MET'] = 0
    df.loc[df['Duration'] < 10, 'Base_MET'] = 3.0  # Light activity
    df.loc[(df['Duration'] >= 10) & (df['Duration'] < 30), 'Base_MET'] = 5.0  # Moderate activity
    df.loc[df['Duration'] >= 30, 'Base_MET'] = 7.0  # Vigorous activity
    
    # Adjust for heart rate zone
    df['HR_Adjustment'] = 1.0
    df.loc[df['HR_Zone'] > 0.5, 'HR_Adjustment'] = 1.0 + (df.loc[df['HR_Zone'] > 0.5, 'HR_Zone'] - 0.5) * 0.5
    
    # Adjust for body temperature
    df['Temp_Adjustment'] = 1.0
    df.loc[df['Body_Temp'] > 37.0, 'Temp_Adjustment'] = 1.0 + (df.loc[df['Body_Temp'] > 37.0, 'Body_Temp'] - 37.0) * 0.02
    
    # Calculate physiologically accurate MET
    df['MET'] = df['Base_MET'] * df['HR_Adjustment'] * df['Temp_Adjustment']
    
    # Polynomial features for non-linear relationships
    df['Age_squared'] = df['Age'] ** 2
    df['Height_squared'] = df['Height'] ** 2
    df['Weight_squared'] = df['Weight'] ** 2
    df['Duration_squared'] = df['Duration'] ** 2
    df['Heart_Rate_squared'] = df['Heart_Rate'] ** 2
    df['Body_Temp_squared'] = df['Body_Temp'] ** 2
    
    # Polynomial interaction terms
    df['Age_Weight_squared'] = (df['Age'] * df['Weight']) ** 2
    df['HeartRate_Duration_squared'] = (df['Heart_Rate'] * df['Duration']) ** 2
    
    return df

# Load the data files
print("Loading data...")
train_data = pd.read_csv('data/train_subsample.csv')
test_data = pd.read_csv('data/test_subsample.csv')

# Outlier detection and handling
def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def detect_outliers_zscore(df, column, threshold=3):
    """Detect outliers using Z-score method"""
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return z_scores > threshold

def handle_outliers(df, columns):
    """Handle outliers using winsorization (cap at 1st and 99th percentiles)"""
    df = df.copy()
    for col in columns:
        lower_percentile = df[col].quantile(0.01)
        upper_percentile = df[col].quantile(0.99)
        df[col] = np.clip(df[col], lower_percentile, upper_percentile)
    return df

def optimize_hyperparameters(X, y, n_trials=50):
    """Optimize hyperparameters using Optuna"""
    def objective(trial):
        params = {
            'objective': rmsle_objective,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
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
    
    print("Best parameters:", study.best_params)
    print("Best score:", study.best_value)
    
    return study.best_params

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Handle outliers in training data
print("Detecting and handling outliers...")
outlier_columns = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
train_data = handle_outliers(train_data, outlier_columns)

# Handle categorical variables (Sex) using one-hot encoding
print("Preprocessing data...")
train_data = pd.get_dummies(train_data, columns=['Sex'], prefix='Gender')
test_data = pd.get_dummies(test_data, columns=['Sex'], prefix='Gender')

# Apply feature engineering
print("Applying feature engineering...")
train_data = add_features(train_data)
test_data = add_features(test_data)

# Separate features and target
feature_cols = [col for col in train_data.columns if col not in ['Calories', 'id']]
X = train_data[feature_cols]
y = train_data['Calories']

# Apply log transformation to target for RMSLE optimization
y_log = np.log1p(y)

print(f"Feature columns: {feature_cols}")
print(f"Number of features: {len(feature_cols)}")

# Split data into train/validation sets (80/20 split)
print("Splitting data into train/validation sets...")
X_train, X_val, y_train, y_val, y_train_log, y_val_log = train_test_split(
    X, y, y_log, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")

# Optimize hyperparameters (commented out for faster execution)
print("Optimizing hyperparameters...")
# Use a smaller subset of data for hyperparameter optimization to avoid memory issues
X_train_subset = X_train.sample(frac=0.1, random_state=42)
y_train_log_subset = y_train_log[X_train_subset.index]
best_params = optimize_hyperparameters(X_train_subset, y_train_log_subset, n_trials=3)

# Initialize XGBoost regressor with optimized defaults
print("Initializing XGBoost model...")
model = xgb.XGBRegressor(
    objective=rmsle_objective,
    **best_params,
    random_state=42
)

# Train model with early stopping
print("Training model with early stopping...")
model.fit(
    X_train, y_train_log,
    eval_set=[(X_val, y_val_log)],
    eval_metric=rmsle_eval,
    early_stopping_rounds=10,
    verbose=False
)

# Predict on validation set
print("Making predictions on validation set...")
y_val_pred_log = model.predict(X_val)
y_val_pred = np.expm1(y_val_pred_log)  # Inverse of log transformation

# Calculate RMSLE
val_rmsle = rmsle(y_val, y_val_pred)
print(f"Validation RMSLE: {val_rmsle:.6f}")

# Check if we meet the target RMSLE
if val_rmsle <= 0.1:
    print("SUCCESS: Model meets the target RMSLE of 0.1 or lower!")
else:
    print("WARNING: Model does not meet the target RMSLE of 0.1 or lower.")
    print("Consider hyperparameter tuning to improve performance.")

# Feature importance
print("\nTop 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))

# Train final model on full dataset
print("\nTraining final model on full dataset...")
final_model = xgb.XGBRegressor(
    objective=rmsle_objective,
    **best_params,
    random_state=42
)
final_model.fit(X, y_log)

# Prepare test data
X_test = test_data[feature_cols]

# Predict on test set
print("Making predictions on test set...")
test_pred_log = final_model.predict(X_test)
test_pred = np.expm1(test_pred_log)

# Create submission for test_subsample.csv
print("Creating submission file for test_subsample.csv...")
submission = pd.DataFrame({
    'id': test_data['id'],
    'Calories': test_pred
})
submission.to_csv('submission.csv', index=False)

print("Submission file saved as 'submission.csv'")

# Now generate predictions for test_full.csv
print("Generating predictions for test_full.csv...")
test_full_data = pd.read_csv('data/test_full.csv')

# Apply the same preprocessing to test_full_data
test_full_data = pd.get_dummies(test_full_data, columns=['Sex'], prefix='Gender')

# Apply feature engineering
test_full_data = add_features(test_full_data)

# Ensure test_full_data has the same columns as training data
for col in feature_cols:
    if col not in test_full_data.columns:
        test_full_data[col] = 0

# Reorder columns to match training data
X_test_full = test_full_data[feature_cols]

# Predict on test_full set
test_full_pred_log = final_model.predict(X_test_full)
test_full_pred = np.expm1(test_full_pred_log)

# Create submission for test_full.csv
print("Creating final submission file for test_full.csv...")
final_submission = pd.DataFrame({
    'id': test_full_data['id'],
    'Calories': test_full_pred
})
final_submission.to_csv('final_submission.csv', index=False)

print("Final submission file saved as 'final_submission.csv'")
print("Model implementation complete!")