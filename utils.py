import os
import pandas as pd


def init_experiment():
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
    return X, y


def avg_importance():
    feature_importance_df = pd.read_csv("results/feature_importance_results.csv")
    averaged_feature_importance = feature_importance_df.groupby(['label', 'model', 'variable'])[
        'average_importance'].mean().reset_index()
    averaged_feature_importance['average_importance'] = averaged_feature_importance['average_importance'].round(3)
    print(averaged_feature_importance)
    averaged_feature_importance.to_csv("averaged_feature_importance_results.csv", index=False)


def avg_overfitting():
    overfitting_df = pd.read_csv("results/overfitting_results.csv")
    averaged_overfitting = overfitting_df.groupby(['label', 'model'])[
        ['avg_train_accuracy', 'avg_test_accuracy']].mean().reset_index()
    averaged_overfitting[['avg_train_accuracy', 'avg_test_accuracy']] = averaged_overfitting[
        ['avg_train_accuracy', 'avg_test_accuracy']].round(3)
    print(averaged_overfitting)
    averaged_overfitting.to_csv("averaged_overfitting_results.csv", index=False)


def avg_bias():
    bias_df = pd.read_csv("results/bias_results.csv")
    averaged_bias = bias_df.groupby(['label', 'model', 'class'])['class_accuracy'].mean().reset_index()
    averaged_bias['class_accuracy'] = averaged_bias['class_accuracy'].round(3)
    print(averaged_bias)
    averaged_bias.to_csv("averaged_bias_results.csv", index=False)


avg_importance()
avg_overfitting()
avg_bias()
