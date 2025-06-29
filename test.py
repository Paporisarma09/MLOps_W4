import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    df = pd.read_csv("data/iris.csv")
    X = df.drop("species", axis=1)
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = joblib.load("model.pkl")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc}")
    assert acc > 0.8

