import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
import warnings
from utils import init_experiment
warnings.filterwarnings("ignore", category=FutureWarning)


X, y = init_experiment()

model_knn = KNeighborsClassifier()
# model_svc = SVC(probability=True)
model_dtc = DecisionTreeClassifier()
knn_res = []
dtc_res = []
knn_f1 = []
dtc_f1 = []
knn_far = []
dtc_far = []
knn_mar = []
dtc_mar = []

skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
for i, features in enumerate(X):
    target = y[i]["anomaly"]
    knn_fold_res = []
    dtc_fold_res = []
    knn_f1_fold = []
    dtc_f1_fold = []
    knn_far_fold = []
    dtc_far_fold = []
    knn_mar_fold = []
    dtc_mar_fold = []

    for train_idx, test_idx in skf.split(features, target):
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

        model_knn.fit(X_train, y_train)
        model_dtc.fit(X_train, y_train)

        y_pred_knn = model_knn.predict(X_test)
        y_pred_dtc = model_dtc.predict(X_test)

        results_knn = balanced_accuracy_score(y_test, y_pred_knn)
        results_dtc = balanced_accuracy_score(y_test, y_pred_dtc)

        results_f1_knn = f1_score(y_test, y_pred_knn)
        results_f1_dtc = f1_score(y_test, y_pred_dtc)

        tn_knn, fp_knn, fn_knn, tp_knn = confusion_matrix(y_test, y_pred_knn, labels=[0, 1]).ravel()
        tn_dtc, fp_dtc, fn_dtc, tp_dtc = confusion_matrix(y_test, y_pred_dtc, labels=[0, 1]).ravel()

        far_knn = fp_knn / (fp_knn + tn_knn) if (fp_knn + tn_knn) > 0 else 0
        mar_knn = fn_knn / (fn_knn + tp_knn) if (fn_knn + tp_knn) > 0 else 0
        far_dtc = fp_dtc / (fp_dtc + tn_dtc) if (fp_dtc + tn_dtc) > 0 else 0
        mar_dtc = fn_dtc / (fn_dtc + tp_dtc) if (fn_dtc + tp_dtc) > 0 else 0

        knn_fold_res.append(results_knn)
        dtc_fold_res.append(results_dtc)
        knn_f1_fold.append(results_f1_knn)
        dtc_f1_fold.append(results_f1_dtc)
        knn_far_fold.append(far_knn)
        dtc_far_fold.append(far_dtc)
        knn_mar_fold.append(mar_knn)
        dtc_mar_fold.append(mar_dtc)

    knn_res.append(np.mean(knn_fold_res))
    dtc_res.append(np.mean(dtc_fold_res))
    knn_f1.append(np.mean(knn_f1_fold))
    dtc_f1.append(np.mean(dtc_f1_fold))
    knn_far.append(np.mean(knn_far_fold))
    dtc_far.append(np.mean(dtc_far_fold))
    knn_mar.append(np.mean(knn_mar_fold))
    dtc_mar.append(np.mean(dtc_mar_fold))

print(f"KNN: Balanced Accuracy:, {np.mean(knn_res):0.3f}, F1:, {np.mean(knn_f1): 0.3f}, FAR:, {np.mean(knn_far):0.3f}, MAR:, {np.mean(knn_mar):0.3f}")
print(f"DTC: Balanced Accuracy:, {np.mean(dtc_res):0.3f}, F1:, {np.mean(dtc_f1): 0.3f}, FAR:, {np.mean(dtc_far):0.3f}, MAR:, {np.mean(dtc_mar):0.3f}")
