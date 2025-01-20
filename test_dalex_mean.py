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
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    y.append(df[["anomaly", "changepoint"]])
    X.append(df.drop(["anomaly", "changepoint"], axis=1))

model_knn = KNeighborsClassifier()
model_dtc = DecisionTreeClassifier()
model_svc = SVC()
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)

results = {
    "anomaly": {
        "feature_importance": [],
        "bias_results": [],
        "overfitting_results": [],
        "local_analysis_results": [],
    },
    "changepoint": {
        "feature_importance": [],
        "bias_results": [],
        "overfitting_results": [],
        "local_analysis_results": [],
    },
}

labels = ["anomaly", "changepoint"]
clf = {"knn": model_knn, "dtc": model_dtc, "svc": model_svc}
# model = clf["knn"]

all_results = []

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

            all_results.append({
                "file": all_files[i],
                "label": label,
                "classifier": model_name,
                "avg_train_accuracy": fold_results["avg_train_accuracy"],
                "avg_test_accuracy": fold_results["avg_test_accuracy"],
                "bias_results": bias_per_class
            })

            print(f'{model_name}, file: {str(i)}, {label}')

average_importance = {}
for label in labels:
    feature_importance_df = pd.concat(results[label]["feature_importance"])
    average_importance[label] = feature_importance_df.groupby("variable")["dropout_loss"].mean()

output_path = "results_summary.csv"
summary_data = []

for label in labels:
    print(f"\nFeature Importance for {label} (average):")
    print(average_importance[label])

    print(f"\nOverfitting Results for {label} (train/test accuracy):")
    for idx, res in enumerate(results[label]["overfitting_results"]):
        print(f"Model {idx}: Train Accuracy: {res['avg_train_accuracy']:.3f}, Test Accuracy: {res['avg_test_accuracy']:.3f}")
        summary_data.append({
            "label": label,
            "model_id": idx,
            "avg_train_accuracy": res["avg_train_accuracy"],
            "avg_test_accuracy": res["avg_test_accuracy"]
        })

    print(f"\nBias Analysis for {label}:")
    for idx, bias in enumerate(results[label]["bias_results"]):
        print(f"Model {idx}: {bias}")

pd.DataFrame(all_results).to_csv(output_path, index=False)
print(f"\nWyniki zapisano do pliku {output_path}")
