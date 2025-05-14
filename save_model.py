import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib  # Library to save the model

# Load the dataset (use the correct dataset file here)
df = pd.read_csv('cleaned_diabetes_data_1.csv')

# Split the data into features (X) and target (y)
X = df.drop('Outcome', axis=1)  # Assuming 'Outcome' is the target column
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the trained model to a file using joblib
joblib.dump(model, 'diabetes_model.pkl')  # Saving the model
print("Model saved successfully as 'diabetes_model.pkl'!")