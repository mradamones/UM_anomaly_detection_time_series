import os
import numpy as np
import pandas as pd
import dalex as dx
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import warnings
from utils import init_experiment
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

X, y = init_experiment()

model_knn = KNeighborsClassifier()
model_dtc = DecisionTreeClassifier()
# model_svc = SVC()
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)

results = {
    "anomaly": {
        "feature_importance": [],
        "bias_results": [],
        "overfitting_results": [],
    },
    "changepoint": {
        "feature_importance": [],
        "bias_results": [],
        "overfitting_results": [],
    },
}

labels = ["anomaly", "changepoint"]
clf = {"knn": model_knn, "dtc": model_dtc}  # "svc": model_svc

feature_importance_data = []
overfitting_data = []
bias_data = []

for model_name, model in clf.items():
    for i, features in enumerate(X):
        for label in labels:
            target = np.array(y[i][label])

            fold_results = {"train_accuracy": [], "test_accuracy": [], "local_explanations": []}

            for fold_num, (train_idx, test_idx) in enumerate(skf.split(features, target)):
                X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                y_train, y_test = target[train_idx], target[test_idx]

                model.fit(X_train, y_train)

                train_preds = model.predict(X_train)
                test_preds = model.predict(X_test)
                train_accuracy = balanced_accuracy_score(y_train, train_preds)
                test_accuracy = balanced_accuracy_score(y_test, test_preds)

                fold_results["train_accuracy"].append(train_accuracy)
                fold_results["test_accuracy"].append(test_accuracy)

                explainer = dx.Explainer(model, X_train, y_train, label=f"{label} {i} fold {fold_num}", verbose=False)
                if len(np.unique(y_train)) > 1:
                    importance = explainer.model_parts(type='difference')
                    results[label]["feature_importance"].append(importance.result)

                local_explanation = explainer.predict_parts(X_test.iloc[0], type="break_down")
                fold_results["local_explanations"].append(local_explanation.result)

            bias_per_class = {}
            for cls in np.unique(target):
                class_indices = np.where(target == cls)[0]
                class_preds = model.predict(features.iloc[class_indices])
                class_accuracy = balanced_accuracy_score(target[class_indices], class_preds)
                bias_per_class[cls] = class_accuracy
            results[label]["bias_results"].append(bias_per_class)

            fold_results["avg_train_accuracy"] = np.mean(fold_results["train_accuracy"])
            fold_results["avg_test_accuracy"] = np.mean(fold_results["test_accuracy"])
            results[label]["overfitting_results"].append(fold_results)

            feature_importance_df = pd.concat(results[label]["feature_importance"])
            average_importance = feature_importance_df.groupby("variable")["dropout_loss"].mean()

            for variable, importance in average_importance.items():
                feature_importance_data.append({
                    "label": label,
                    "model": model_name,
                    "variable": variable,
                    "average_importance": importance
                })

            for res in results[label]["overfitting_results"]:
                overfitting_data.append({
                    "label": label,
                    "model": model_name,
                    "avg_train_accuracy": res["avg_train_accuracy"],
                    "avg_test_accuracy": res["avg_test_accuracy"]
                })

            for bias in results[label]["bias_results"]:
                for cls, bias_val in bias.items():
                    bias_data.append({
                        "label": label,
                        "model": model_name,
                        "class": cls,
                        "class_accuracy": bias_val
                    })
            print(f'{model_name} {label}, {i}')

pd.DataFrame(feature_importance_data).to_csv("feature_importance_results.csv", index=False)
pd.DataFrame(overfitting_data).to_csv("overfitting_results.csv", index=False)
pd.DataFrame(bias_data).to_csv("bias_results.csv", index=False)

