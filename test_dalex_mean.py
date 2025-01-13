import os
import numpy as np
import pandas as pd
import dalex as dx
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

all_files = []

for root, _, files in os.walk('./SKAB/'):
    for file in files:
        if file.endswith(".csv"):
            all_files.append(os.path.join(root, file))

list_of_df = [
    pd.read_csv(file, sep=";", index_col="datetime", parse_dates=True)
    for file in all_files
    if "anomaly-free" not in file
]

X = []
y = []
for df in list_of_df:
    y.append(df[["anomaly"]])
    X.append(df.drop(["anomaly", "changepoint"], axis=1))

model_knn = KNeighborsClassifier()
skf = StratifiedKFold(n_splits=5)

feature_importance_results = []

for i, features in enumerate(X):
    target = y[i]["anomaly"]

    for train_idx, test_idx in skf.split(features, target):
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

        model_knn.fit(X_train, y_train)

        explainer = dx.Explainer(model_knn, X_train, y_train, label=f"KNN Model {i}")

        importance = explainer.model_parts(type='difference')

        feature_importance_results.append(importance.result)

feature_importance_df = pd.concat(feature_importance_results)
average_importance = feature_importance_df.groupby("variable")["dropout_loss"].mean()

print(average_importance)
