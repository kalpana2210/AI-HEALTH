import firebase_admin
from firebase_admin import credentials, firestore

def initialize_firebase():
    try:
        # Initialize Firebase with your service account key
        cred = credentials.Certificate('serviceAccountKey.json')
        firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        return None

def save_prediction(db, data):
    try:
        # Add data directly to predictions collection
        db.collection('predictions').add(data)
        return True
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return False

def get_user_predictions(db, user_id):
    try:
        # Get predictions for a specific user
        predictions = db.collection('predictions').where('user_id', '==', user_id).get()
        return [doc.to_dict() for doc in predictions]
    except Exception as e:
        print(f"Error getting user predictions: {e}")
        return [] 