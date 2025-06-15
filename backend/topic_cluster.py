# topic_cluster.py
import joblib
import os

model_path = os.path.join("model", "kmeans_topic.pkl")
model, vectorizer = joblib.load(model_path)

def predict_cluster(sentence):
    X = vectorizer.transform([sentence])
    return model.predict(X)[0]
