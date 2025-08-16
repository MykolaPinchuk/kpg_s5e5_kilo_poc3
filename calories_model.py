import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb
import warnings
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

# Function to add derived features
def add_features(df):
    """
    Add derived features to the dataset
    
    Parameters:
    df: pandas.DataFrame - Input dataframe
    
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
    
    # Calculate MET approximation: Calories / (Weight * Duration) * 60
    # For feature creation, we'll use a simplified approximation based on heart rate and age
    df['MET'] = df['Heart_Rate'] / df['Age']  # Simplified approximation
    
    return df

# Load the data files
print("Loading data...")
train_data = pd.read_csv('data/train_subsample.csv')
test_data = pd.read_csv('data/test_subsample.csv')

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

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

# Initialize XGBoost regressor with optimized defaults
print("Initializing XGBoost model...")
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

# Train model with early stopping
print("Training model with early stopping...")
model.fit(
    X_train, y_train_log,
    eval_set=[(X_val, y_val_log)],
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