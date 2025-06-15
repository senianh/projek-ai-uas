# difficulty_classifier.py
import joblib
import os

model_path = os.path.join("model", "difficulty_model.pkl")
model = joblib.load(model_path)

def predict_difficulty(sentence):
    return model.predict([sentence])[0]
