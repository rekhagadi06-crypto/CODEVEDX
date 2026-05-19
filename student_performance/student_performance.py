# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("student_data.csv")

# Display first 5 rows
print("\nFirst 5 Rows of Dataset:")
print(data.head())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Features (inputs)
X = data[["Attendance", "StudyHours", "Marks"]]

# Target (output)
y = data["FinalPerformance"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# Predict new student performance
print("\n--- Student Performance Prediction ---")

attendance = float(input("Enter Attendance Percentage: "))
study_hours = float(input("Enter Study Hours Per Day: "))
marks = float(input("Enter Internal Marks: "))

new_data = [[attendance, study_hours, marks]]

prediction = model.predict(new_data)

print("\nPredicted Final Performance:",
      round(prediction[0], 2))

# Visualization
plt.figure(figsize=(8,5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Performance")
plt.ylabel("Predicted Performance")

plt.title("Actual vs Predicted Student Performance")

plt.grid(True)

plt.show()