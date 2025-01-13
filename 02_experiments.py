import os
import numpy as np
import dalex as dx
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

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

anomaly_free_df = pd.read_csv("./SKAB/anomaly-free/anomaly-free.csv", index_col="datetime", sep=";", parse_dates=True)

X = []
y = []
model_knn = KNeighborsClassifier()
model_svc = SVC()
knn_res = []
svc_res = []
knn_f1 = []
svc_f1 = []
knn_far = []
svc_far = []
knn_mar = []
svc_mar = []

for df in list_of_df:
    y.append(df[["anomaly", "changepoint"]])
    X.append(df.drop(["anomaly", "changepoint"], axis=1))

skf = StratifiedKFold(n_splits=5)
for i, features in enumerate(X):
    target = y[i]["anomaly"]
    knn_fold_res = []
    svc_fold_res = []
    knn_f1_fold = []
    svc_f1_fold = []
    knn_far_fold = []
    svc_far_fold = []
    knn_mar_fold = []
    svc_mar_fold = []

    for train_idx, test_idx in skf.split(features, target):
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

        model_knn.fit(X_train, y_train)
        model_svc.fit(X_train, y_train)

        y_pred_knn = model_knn.predict(X_test)
        y_pred_svc = model_svc.predict(X_test)

        results_knn = balanced_accuracy_score(y_test, y_pred_knn)
        results_svc = balanced_accuracy_score(y_test, y_pred_svc)

        results_f1_knn = f1_score(y_test, y_pred_knn)
        results_f1_svc = f1_score(y_test, y_pred_svc)

        tn_knn, fp_knn, fn_knn, tp_knn = confusion_matrix(y_test, y_pred_knn, labels=[0, 1]).ravel()
        tn_svc, fp_svc, fn_svc, tp_svc = confusion_matrix(y_test, y_pred_svc, labels=[0, 1]).ravel()

        far_knn = fp_knn / (fp_knn + tn_knn) if (fp_knn + tn_knn) > 0 else 0
        mar_knn = fn_knn / (fn_knn + tp_knn) if (fn_knn + tp_knn) > 0 else 0
        far_svc = fp_svc / (fp_svc + tn_svc) if (fp_svc + tn_svc) > 0 else 0
        mar_svc = fn_svc / (fn_svc + tp_svc) if (fn_svc + tp_svc) > 0 else 0

        knn_fold_res.append(results_knn)
        svc_fold_res.append(results_svc)
        knn_f1_fold.append(results_f1_knn)
        svc_f1_fold.append(results_f1_svc)
        knn_far_fold.append(far_knn)
        svc_far_fold.append(far_svc)
        knn_mar_fold.append(mar_knn)
        svc_mar_fold.append(mar_svc)

    knn_res.append(np.mean(knn_fold_res))
    svc_res.append(np.mean(svc_fold_res))
    knn_f1.append(np.mean(knn_f1_fold))
    svc_f1.append(np.mean(svc_f1_fold))
    knn_far.append(np.mean(knn_far_fold))
    svc_far.append(np.mean(svc_far_fold))
    knn_mar.append(np.mean(knn_mar_fold))
    svc_mar.append(np.mean(svc_mar_fold))

print("KNN: Balanced Accuracy:", np.mean(knn_res), "F1:", np.mean(knn_f1), "FAR:", np.mean(knn_far), "MAR:", np.mean(knn_mar))
print("SVC: Balanced Accuracy:", np.mean(svc_res), "F1:", np.mean(svc_f1), "FAR:", np.mean(svc_far), "MAR:", np.mean(svc_mar))

print("Explaining KNN model with Dalex")
explainer_knn = dx.Explainer(model_knn, X_test, y_test, label="KNN Classifier")
feature_importance_knn = explainer_knn.model_parts()
feature_importance_knn.plot(title="Feature Importance - KNN")

print("Explaining SVC model with Dalex")
explainer_svc = dx.Explainer(model_svc, X_test, y_test, label="SVC Classifier")
feature_importance_svc = explainer_svc.model_parts()
feature_importance_svc.plot(title="Feature Importance - SVC")

sample = X_test.iloc[0:1]
local_explanation_knn = explainer_knn.predict_parts(sample)
local_explanation_knn.plot(title="Local Explanation - KNN")

local_explanation_svc = explainer_svc.predict_parts(sample)
local_explanation_svc.plot(title="Local Explanation - SVC")
