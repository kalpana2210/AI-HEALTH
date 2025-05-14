import pandas as pd

# Load the 3 datasets
df1 = pd.read_csv('diabetes_health_indicators_1.csv')
df2 = pd.read_csv('diabetes_health_indicators_2.csv')
df3 = pd.read_csv('diabetes_health_indicators_3.csv')

# Show first 5 rows of each dataset
print("First 5 rows of Dataset 1:")
print(df1.head())

print("\nFirst 5 rows of Dataset 2:")
print(df2.head())

print("\nFirst 5 rows of Dataset 3:")
print(df3.head())
# Check basic info and shape
print("\nDataset 1 Info:")
print(df1.info())

print("\nDataset 2 Info:")
print(df2.info())

print("\nDataset 3 Info:")
print(df3.info())

# Show summary statistics
print("\nSummary Statistics for Dataset 1:")
print(df1.describe())

print("\nSummary Statistics for Dataset 2:")
print(df2.describe())

print("\nSummary Statistics for Dataset 3:")
print(df3.describe())

# Show the shapes
print("\nShape of Dataset 1:", df1.shape)
print("\nShape of Dataset 2:", df2.shape)
print("\nShape of Dataset 3:", df3.shape)
# Check for missing values
print("\nMissing Values in Dataset 1:")
print(df1.isnull().sum())

print("\nMissing Values in Dataset 2:")
print(df2.isnull().sum())

print("\nMissing Values in Dataset 3:")
print(df3.isnull().sum())
# Get the basic statistical summary of the dataset
print(df1.describe())
print(df2.describe())
print(df3.describe())
# Check for any remaining missing values
print(df1.isnull().sum())
print(df2.isnull().sum())
print(df3.isnull().sum())
# Saving the cleaned datasets
df1.to_csv('cleaned_diabetes_data_1.csv', index=False)
df2.to_csv('cleaned_diabetes_data_2.csv', index=False)
df3.to_csv('cleaned_diabetes_data_3.csv', index=False)
