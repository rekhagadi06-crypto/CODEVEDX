# ==========================================
# SMART RECOMMENDATION SYSTEM
# ==========================================

# Import Libraries
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv("courses.csv")

print("\n===== AVAILABLE COURSES =====\n")
print(data["course"])

# ==========================================
# TF-IDF VECTORIZATION
# ==========================================

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(data["description"])

# ==========================================
# COSINE SIMILARITY
# ==========================================

similarity = cosine_similarity(tfidf_matrix)

# ==========================================
# RECOMMENDATION FUNCTION
# ==========================================

def recommend_course(course_name):

    # Check if course exists
    if course_name not in data["course"].values:
        print("\nCourse not found!")
        return

    # Get course index
    index = data[data["course"] == course_name].index[0]

    # Similarity scores
    similarity_scores = list(enumerate(similarity[index]))

    # Sort scores
    sorted_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    print("\nRecommended Courses:\n")

    # Display recommendations
    for i in sorted_scores[1:6]:

        recommended_course = data.iloc[i[0]]["course"]

        score = round(i[1] * 100, 2)

        print(f"{recommended_course}  ---> Similarity Score: {score}%")

# ==========================================
# USER INPUT
# ==========================================

print("\n===== SMART RECOMMENDATION SYSTEM =====")

user_course = input("\nEnter a Course You Like: ")

recommend_course(user_course)