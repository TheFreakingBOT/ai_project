import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Path to dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), "diabetes.csv")

def load_data():
    if os.path.exists(DATA_PATH):
        # Load real dataset
        df = pd.read_csv(DATA_PATH)
        print("Loaded real dataset: diabetes.csv")
    else:
        # Generate synthetic fallback dataset
        print("diabetes.csv not found. Using synthetic data...")
        np.random.seed(42)
        X = np.random.rand(100, 8)  # 8 features
        y = np.random.randint(0, 2, 100)  # binary target
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])
        df["Outcome"] = y
    return df

def train_model():
    df = load_data()
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "diabetes_model.joblib")
    print("Model saved as diabetes_model.joblib")

# In train.py, after saving the model
if os.path.exists(DATA_PATH):
    source = "real dataset (diabetes.csv)"
else:
    source = "synthetic fallback data"

with open("model_source.txt", "w") as f:
    f.write(source)

print(f"Model trained on {source}")

if __name__ == "__main__":
    train_model()