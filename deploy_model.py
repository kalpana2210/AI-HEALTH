from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model (make sure the model file is in the correct directory)
model = joblib.load('diabetes_model.pkl')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        # Extract features from the JSON data (the 'features' key should be in your data)
        features = np.array(data['features']).reshape(1, -1)
        
        # Predict using the model
        prediction = model.predict(features)
        
        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        # If any error occurs, return an error message
        return jsonify({'error': str(e)}), 500

# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)
