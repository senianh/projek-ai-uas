import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# Load dataset
df = pd.read_excel(os.path.join("model", "kalimat_difficulty_100.xlsx"))
X = df["kalimat"]
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Pipeline: TF-IDF + Logistic Regression
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000, C=1.0))
])

# Train
model.fit(X_train, y_train)

# Evaluation
print("Training Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy :", model.score(X_test, y_test))

y_pred = model.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, os.path.join("model", "difficulty_model.pkl"))
print("\nModel saved to model/difficulty_model.pkl")
