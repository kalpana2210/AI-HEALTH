# STEP 1: IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, mean_squared_error

# STEP 2: LOAD DATA
df = pd.read_csv('cleaned_diabetes_data_1.csv')  # change this filename if needed

# STEP 3: FEATURES & TARGET
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# STEP 4: SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 5: SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# STEP 6: MODEL TRAINING
lr_model = LogisticRegression()
rf_model = RandomForestClassifier()

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# STEP 7: PREDICTION
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# STEP 8: CONFUSION MATRIX (Random Forest)
cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Non-Diabetic", "Diabetic"],
            yticklabels=["Non-Diabetic", "Diabetic"])
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# STEP 9: ROC Curve
fpr, tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('Random Forest ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# STEP 10: COMPARISON TABLE
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [accuracy_score(y_test, lr_pred), accuracy_score(y_test, rf_pred)],
    'Precision': [precision_score(y_test, lr_pred), precision_score(y_test, rf_pred)],
    'Recall': [recall_score(y_test, lr_pred), recall_score(y_test, rf_pred)],
    'F1-Score': [f1_score(y_test, lr_pred), f1_score(y_test, rf_pred)]
})
print("\nModel Comparison Table:\n")
print(results)

# STEP 11: RMSE
rmse_rf = np.sqrt(mean_squared_error(y_test, rf_pred))
print(f"\nRandom Forest RMSE: {rmse_rf:.2f}")
