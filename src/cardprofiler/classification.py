import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from cardprofiler.preprocessing import COLS


def split_data(df, test_size=0.2):
    X = df[COLS]
    y = df["target"]
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


def build_pipeline(classifier):
    return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])


def evaluate(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }


def tune_random_forest(X_train, y_train):
    pipe = build_pipeline(RandomForestClassifier(random_state=42))
    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 10, 20, 30],
        "classifier__min_samples_split": [2, 5, 10],
    }
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1_weighted", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid


def save_pipeline(pipe, path):
    joblib.dump(pipe, path)