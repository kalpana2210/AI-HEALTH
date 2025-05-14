from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from firebase_config import initialize_firebase, save_prediction, get_user_predictions
from firebase_admin import firestore

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "https://ai-health-1-m13x.onrender.com"}})

@app.route('/')
def home():
    return render_template('index.html')


# Initialize Firebase
db = initialize_firebase()

# Load data and train model
data = pd.read_csv('diabetes_health_indicators_1.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model with adjusted parameters
model = RandomForestClassifier(
    n_estimators=200,  # More trees
    max_depth=10,      # Deeper trees
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # Balance the classes
    random_state=42
)
model.fit(X_scaled, y)

@app.route('/predict', methods=['POST','OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '',200
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        features = np.array([
            float(data['pregnancies']),
            float(data['glucose']),
            float(data['bloodPressure']),
            float(data['skinthickness']),
            float(data['insulin']),
            float(data['bmi']),
            float(data['diabetesPedigreeFunction']),
            float(data['age'])
        ]).reshape(1, -1)
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        # Get prediction probability
        proba = model.predict_proba(features_scaled)[0]
        risk_percentage = round(proba[1] * 100, 2)  # Probability of high risk
        
        # Prepare data for Firebase
        firebase_data = {
            'user_data': data,
            'prediction': int(prediction),
            'risk_percentage': risk_percentage,
            'timestamp': firestore.SERVER_TIMESTAMP
        }
        
        # Save prediction to Firebase
        if db:
            save_prediction(db, firebase_data)
        
        return jsonify({
            'prediction': int(prediction),
            'risk_percentage': risk_percentage
        })
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/user_predictions/<user_id>', methods=['GET'])
def get_predictions(user_id):
    try:
        if db:
            predictions = get_user_predictions(db, user_id)
            return jsonify(predictions)
        return jsonify({'error': 'Database not initialized'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
