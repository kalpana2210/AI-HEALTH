import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df1 = pd.read_csv('cleaned_diabetes_data_1.csv')
df2 = pd.read_csv('cleaned_diabetes_data_2.csv')
df3 = pd.read_csv('cleaned_diabetes_data_3.csv')
# Histogram for 'Age'
plt.figure(figsize=(8, 6))
sns.histplot(df1['Age'], kde=True, color='blue', bins=30)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Boxplot for 'Age'
plt.figure(figsize=(8, 6))
sns.boxplot(x=df1['Age'], color='green')
plt.title('Boxplot for Age')
plt.xlabel('Age')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df1.corr(), annot=True, cmap='coolwarm', linewidths=1, linecolor='black')
plt.title('Correlation Heatmap')
plt.show()
