import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

FILE_NAME = "utility_data.csv"

# Create CSV file if not exists
if not os.path.exists(FILE_NAME):
    df = pd.DataFrame(columns=["hours_used", "electricity_bill"])
    df.to_csv(FILE_NAME, index=False)

# Function to load data
def load_data():
    return pd.read_csv(FILE_NAME)

# Function to save data
def save_data(df):
    df.to_csv(FILE_NAME, index=False)

# Add new usage data
def add_data():
    try:
        hours = float(input("Enter hours used: "))
        bill = float(input("Enter electricity bill: "))

        df = load_data()

        new_row = pd.DataFrame({
            "hours_used": [hours],
            "electricity_bill": [bill]
        })

        df = pd.concat([df, new_row], ignore_index=True)

        save_data(df)

        print("Data added successfully!")

    except Exception as e:
        print("Error:", e)

# View all data
def view_data():
    df = load_data()

    if df.empty:
        print("No data available.")
    else:
        print("\nUtility Usage Data")
        print(df)

# Train ML model
def train_model():
    df = load_data()

    if len(df) < 2:
        print("Not enough data to train model.")
        return None

    X = df[["hours_used"]]
    y = df["electricity_bill"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    print(f"Model trained successfully!")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    return model

# Predict bill
def predict_bill():
    model = train_model()

    if model is None:
        return

    try:
        hours = float(input("Enter expected hours used: "))

        prediction = model.predict([[hours]])

        print(f"Predicted Electricity Bill: ₹{prediction[0]:.2f}")

    except Exception as e:
        print("Error:", e)

# Update data
def update_data():
    df = load_data()

    if df.empty:
        print("No data available.")
        return

    print(df)

    try:
        index = int(input("Enter row index to update: "))

        if index not in df.index:
            print("Invalid index.")
            return

        hours = float(input("Enter new hours used: "))
        bill = float(input("Enter new electricity bill: "))

        df.at[index, "hours_used"] = hours
        df.at[index, "electricity_bill"] = bill

        save_data(df)

        print("Data updated successfully!")

    except Exception as e:
        print("Error:", e)

# Menu
while True:
    print("\n===== Utility Usage Prediction Tool =====")
    print("1. Add Usage Data")
    print("2. View Data")
    print("3. Update Data")
    print("4. Predict Electricity Bill")
    print("5. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        add_data()

    elif choice == "2":
        view_data()

    elif choice == "3":
        update_data()

    elif choice == "4":
        predict_bill()

    elif choice == "5":
        print("Exiting program...")
        break

    else:
        print("Invalid choice. Try again.")