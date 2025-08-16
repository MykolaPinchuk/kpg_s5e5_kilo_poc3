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

# Function to add only core features
def add_core_features(df):
    """
    Add only core features to the dataset
    
    Parameters:
    df: pandas.DataFrame - Input dataframe
    
    Returns:
    pandas.DataFrame: Dataframe with added core features
    """
    df = df.copy()
    
    # Calculate BMI: Weight / (Height/100)^2
    df['BMI'] = df['Weight'] / ((df['Height']/100) ** 2)
    
    # Create core interaction features
    df['Weight_Duration'] = df['Weight'] * df['Duration']
    df['HeartRate_Duration'] = df['Heart_Rate'] * df['Duration']
    
    return df

# Function to handle outliers after feature engineering
def detect_and_handle_outliers(df, columns):
    """
    Detect and handle outliers using IQR method after feature engineering
    
    Parameters:
    df: pandas.DataFrame - Input dataframe
    columns: list - Columns to apply outlier handling
    
    Returns:
    pandas.DataFrame: Dataframe with outliers handled
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

# Main execution
if __name__ == "__main__":
    # Load the data files
    print("Loading data...")
    train_data = pd.read_csv('data/train_subsample.csv')
    test_data = pd.read_csv('data/test_subsample.csv')
    
    # Handle categorical variables (Sex) using one-hot encoding
    print("Preprocessing data...")
    train_data = pd.get_dummies(train_data, columns=['Sex'], prefix='Gender')
    test_data = pd.get_dummies(test_data, columns=['Sex'], prefix='Gender')
    
    # Apply simplified feature engineering
    print("Applying feature engineering...")
    train_data = add_core_features(train_data)
    test_data = add_core_features(test_data)
    
    # Apply outlier handling after feature engineering
    print("Handling outliers...")
    feature_cols_for_outliers = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI', 'Weight_Duration', 'HeartRate_Duration']
    train_data = detect_and_handle_outliers(train_data, feature_cols_for_outliers)
    
    # Separate features and target
    feature_cols = [col for col in train_data.columns if col not in ['Calories', 'id']]
    X = train_data[feature_cols]
    y = train_data['Calories']
    
    print(f"Feature columns: {feature_cols}")
    print(f"Number of features: {len(feature_cols)}")
    
    # Split data into train/validation sets (80/20 split)
    print("Splitting data into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    
    # Using optimized parameters from previous run
    best_params = {
        'n_estimators': 1033,
        'max_depth': 12,
        'learning_rate': 0.19263952908422077,
        'subsample': 0.9695937530283715,
        'colsample_bytree': 0.9090348346782959,
        'reg_alpha': 4.226033657674094,
        'reg_lambda': 6.971305796243095,
        'gamma': 1.2081072765284877,
        'min_child_weight': 2,
        'random_state': 42
    }
    
    # Initialize XGBoost regressor with optimized defaults
    print("Initializing XGBoost model...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        **best_params
    )
    
    # Train model with early stopping
    print("Training model with early stopping...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        early_stopping_rounds=10,
        verbose=False
    )
    
    # Predict on validation set
    print("Making predictions on validation set...")
    y_val_pred = model.predict(X_val)
    
    # Calculate RMSLE
    val_rmsle = rmsle(y_val, y_val_pred)
    print(f"Validation RMSLE: {val_rmsle:.6f}")
    
    # Check if we meet the target RMSLE
    if val_rmsle <= 0.1:
        print("SUCCESS: Model meets the target RMSLE of 0.1 or lower!")
    else:
        print("WARNING: Model does not meet the target RMSLE of 0.1 or lower.")
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10))
    
    print("\nModel validation complete!")