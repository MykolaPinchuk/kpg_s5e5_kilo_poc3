# Calories Burned Prediction Model - Summary

## Implementation Overview

This document summarizes the implementation of a baseline ML model for predicting calories burned during workouts using XGBoost.

## Data Processing

- **Data Sources**: 
  - Training data: `data/train_subsample.csv` (75,000 samples)
  - Test data: `data/test_subsample.csv` (25,000 samples)
  - Full test data: `data/test_full.csv` (for final submission)

- **Features**:
  - Original features: Age, Height, Weight, Duration, Heart_Rate, Body_Temp, Sex
  - Derived features: BMI, Weight_Duration, HeartRate_Duration, Age_Weight, MET
  - Categorical encoding: One-hot encoding for Sex (Gender_female, Gender_male)

## Model Implementation

- **Algorithm**: XGBoost Regressor
- **Objective**: `reg:squarederror`
- **Parameters**: 
  - n_estimators: 1000
  - max_depth: 6
  - learning_rate: 0.1
- **Training**: With early stopping (patience: 10 rounds)
- **Target Transformation**: Log transformation for RMSLE optimization

## Results

- **Validation RMSLE**: 0.061888
- **Target Achieved**: Yes (target was ≤ 0.1)
- **Training/Validation Split**: 80/20

## Feature Importance (Top 10)

1. HeartRate_Duration (0.974)
2. Heart_Rate (0.0068)
3. Age (0.0061)
4. Gender_female (0.0042)
5. Age_Weight (0.0040)
6. Duration (0.0018)
7. Weight (0.0007)
8. Body_Temp (0.0007)
9. MET (0.0005)
10. Weight_Duration (0.0004)

## Files Generated

1. `submission.csv` - Predictions for test_subsample.csv
2. `final_submission.csv` - Predictions for test_full.csv

## Conclusion

The model successfully achieved the target RMSLE score of 0.1 or lower, with a validation RMSLE of 0.061888. The implementation includes proper data preprocessing, feature engineering, model training with early stopping, and evaluation using the correct RMSLE metric.