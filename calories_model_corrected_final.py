import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import optuna
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

# Custom RMSLE objective function for XGBoost
def rmsle_objective(y_pred, y_true):
    """
    Custom objective function for RMSLE
    
    Parameters:
    y_pred: array-like - Predicted values
    y_true: xgboost.DMatrix - True target values
    
    Returns:
    tuple: (gradients, hessians)
    """
    y_true_values = y_true.get_label()
    # Calculate gradients and hessians for RMSLE with raw values
    grad = (np.log1p(y_pred) - np.log1p(y_true_values)) / (y_pred + 1)
    hess = (1 / (y_pred + 1)) * (1 - (np.log1p(y_pred) - np.log1p(y_true_values)) / (y_pred + 1))
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
    # Calculate RMSLE with raw values
    rmsle_value = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true_values))**2))
    return 'RMSLE', rmsle_value

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

def optimize_hyperparameters(X, y, n_trials=100):  # Using 100 trials for final run
    """
    Optimize hyperparameters using Optuna with full dataset
    
    Parameters:
    X: pandas.DataFrame - Feature matrix
    y: pandas.Series - Target variable
    n_trials: int - Number of optimization trials
    
    Returns:
    dict: Best hyperparameters
    """
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
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
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_log_error')  # Using 5-fold CV
        return -scores.mean()  # Return negative because Optuna minimizes
    
    # Create study and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("Best parameters:", study.best_params)
    print("Best score:", study.best_value)
    
    return study.best_params

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
    
    # Optimize hyperparameters with full dataset and 100 trials
    print("Optimizing hyperparameters...")
    best_params = optimize_hyperparameters(X_train, y_train, n_trials=100)  # Using 100 trials
    
    # Initialize XGBoost regressor with optimized defaults
    print("Initializing XGBoost model...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        **best_params,
        random_state=42
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
    
    # Train final model on full dataset
    print("\nTraining final model on full dataset...")
    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        **best_params,
        random_state=42
    )
    final_model.fit(X, y)
    
    # Prepare test data
    X_test = test_data[feature_cols]
    
    # Predict on test set
    print("Making predictions on test set...")
    test_pred = final_model.predict(X_test)
    
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
    test_full_data = add_core_features(test_full_data)
    
    # Handle outliers
    test_full_data = detect_and_handle_outliers(test_full_data, feature_cols_for_outliers)
    
    # Ensure test_full_data has the same columns as training data
    for col in feature_cols:
        if col not in test_full_data.columns:
            test_full_data[col] = 0
    
    # Reorder columns to match training data
    X_test_full = test_full_data[feature_cols]
    
    # Predict on test_full set
    test_full_pred = final_model.predict(X_test_full)
    
    # Create submission for test_full.csv
    print("Creating final submission file for test_full.csv...")
    final_submission = pd.DataFrame({
        'id': test_full_data['id'],
        'Calories': test_full_pred
    })
    final_submission.to_csv('final_submission.csv', index=False)
    
    print("Final submission file saved as 'final_submission.csv'")
    print("Model implementation complete!")