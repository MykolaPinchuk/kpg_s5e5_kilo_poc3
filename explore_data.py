import pandas as pd
import numpy as np

# Load the data files
train_data = pd.read_csv('data/train_subsample.csv')
test_data = pd.read_csv('data/test_subsample.csv')

# Display basic information about the datasets
print("Train Data Shape:", train_data.shape)
print("Test Data Shape:", test_data.shape)

print("\nTrain Data Info:")
print(train_data.info())

print("\nTrain Data Description:")
print(train_data.describe())

print("\nFirst few rows of train data:")
print(train_data.head())

print("\nMissing values in train data:")
print(train_data.isnull().sum())

print("\nTest Data Info:")
print(test_data.info())

print("\nTest Data Description:")
print(test_data.describe())

print("\nFirst few rows of test data:")
print(test_data.head())

print("\nMissing values in test data:")
print(test_data.isnull().sum())

# Check unique values in categorical variables
print("\nUnique values in Sex (Train):", train_data['Sex'].unique())
print("Unique values in Sex (Test):", test_data['Sex'].unique())

# Check target distribution
print("\nTarget (Calories) distribution:")
print("Min:", train_data['Calories'].min())
print("Max:", train_data['Calories'].max())
print("Mean:", train_data['Calories'].mean())
print("Std:", train_data['Calories'].std())