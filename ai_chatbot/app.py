# ==========================================
# AI CHATBOT FOR INTERNAL HELPDESK
# ==========================================

# Import Libraries
from flask import Flask, render_template, request
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv("faq_dataset.csv")

# Questions and Answers
questions = data["question"]
answers = data["answer"]

# ==========================================
# TEXT VECTORIZATION
# ==========================================

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(questions)

# ==========================================
# TRAIN MODEL
# ==========================================

model = KNeighborsClassifier(n_neighbors=1)

model.fit(X, answers)

# Save model
joblib.dump(model, "chatbot_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model Trained Successfully!")

# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Chatbot Response
@app.route("/get")
def chatbot_response():

    user_message = request.args.get("msg")

    # Convert text into vector
    user_vector = vectorizer.transform([user_message])

    # Predict answer
    response = model.predict(user_vector)

    return str(response[0])

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)