import joblib
import pandas as pd  # Importing pandas correctly
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the saved model
model = joblib.load('diabetes_model.pkl')

# Load the dataset (use the correct file path)
df = pd.read_csv('cleaned_diabetes_data_1.csv')  # Adjust this if using another file

# Split the data into features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Make predictions using the loaded model
y_pred = model.predict(X)

# Evaluate the model
accuracy = accuracy_score(y, y_pred)
conf_matrix = confusion_matrix(y, y_pred)
class_report = classification_report(y, y_pred)

# Print evaluation results
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
