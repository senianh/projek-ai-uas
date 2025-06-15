import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib
import os

# Load kalimat dari dataset
df = pd.read_excel(os.path.join("model", "kalimat_difficulty_100.xlsx"))
sentences = df["kalimat"]

# TF-IDF vektorisasi
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

# Clustering pakai KMeans (3 topik)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Simpan model dan vectorizer ke file
joblib.dump((kmeans, vectorizer), os.path.join("model", "kmeans_topic.pkl"))
print("✅ Model clustering disimpan ke: model/kmeans_topic.pkl")
