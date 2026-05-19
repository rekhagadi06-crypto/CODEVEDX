# ==========================================
# AI BASED FAKE NEWS DETECTION TOOL
# ==========================================

# Import Libraries
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv("news_dataset.csv")

print("\n===== DATASET =====\n")
print(data)

# ==========================================
# INPUT AND OUTPUT
# ==========================================

X = data["text"]
y = data["label"]

# ==========================================
# TEXT VECTORIZATION
# ==========================================

vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# ==========================================
# SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# TRAIN MODEL
# ==========================================

model = MultinomialNB()

model.fit(X_train, y_train)

print("\n===== MODEL TRAINED SUCCESSFULLY =====")

# ==========================================
# MODEL EVALUATION
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy :", round(accuracy * 100, 2), "%")

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel Saved Successfully!")

# ==========================================
# USER INPUT
# ==========================================

print("\n===== FAKE NEWS DETECTION =====")

news = input("\nEnter News Text: ")

news_vector = vectorizer.transform([news])

prediction = model.predict(news_vector)

probability = model.predict_proba(news_vector)

confidence = probability.max() * 100

print("\nPrediction :", prediction[0])

print("Confidence Score :", round(confidence, 2), "%")