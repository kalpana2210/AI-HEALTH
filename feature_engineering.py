import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load datasets
df1 = pd.read_csv('cleaned_diabetes_data_1.csv')
df2 = pd.read_csv('cleaned_diabetes_data_2.csv')
df3 = pd.read_csv('cleaned_diabetes_data_3.csv')

# Initialize scaler
scaler = StandardScaler()

# Function to scale numerical columns
def scale_numerical_columns(df):
    # Select only numerical columns (excluding target column if any)
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

# Apply scaling
df1 = scale_numerical_columns(df1)
df2 = scale_numerical_columns(df2)
df3 = scale_numerical_columns(df3)

# Save final datasets
df1.to_csv('final_diabetes_data_1.csv', index=False)
df2.to_csv('final_diabetes_data_2.csv', index=False)
df3.to_csv('final_diabetes_data_3.csv', index=False)

print("✅ Feature Engineering completed! Datasets are scaled and ready for model building!")
