import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import joblib
import csv

def main():
    data = pd.read_csv('data/iris.csv')

    train, test = train_test_split(
        data,
        test_size=0.4,
        stratify=data['species'],
        random_state=42
    )

    X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_train = train['species']
    X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_test = test['species']

    mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
    mod_dt.fit(X_train, y_train)
    prediction = mod_dt.predict(X_test)

    print("The accuracy of the Decision Tree is {:.3f}".format(metrics.accuracy_score(prediction, y_test)))

    # Save the model
    joblib.dump(mod_dt, "model.pkl")

    # Save metrics to CSV
    with open("metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["accuracy", metrics.accuracy_score(prediction, y_test)])

if __name__ == "__main__":
    main()

