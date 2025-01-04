import sys
import warnings

import pandas as pd

sys.path.append("..")
from model import KNN
from metrics import chp_score
from utils import load_preprocess_skab, plot_results

warnings.filterwarnings("ignore", category=UserWarning)

Xy_traintest_list = load_preprocess_skab()

model = KNN(scaling=True, n_neighbors=10)

predicted_outlier, predicted_cp = [], []
true_outlier, true_cp = [], []
for X_train, X_test, y_train, y_test in Xy_traintest_list:
    model.fit(X_train)

    model.predict(
        X_test,
        window_size=5,
        plot_fig=False,
    )

    prediction = pd.Series(
        (model.scores.loc[:, "KNN_score"].values > model.ucl).astype(int),
        index=X_test.index,
    ).fillna(0)

    predicted_outlier.append(prediction)

    prediction_cp = abs(prediction.diff())
    prediction_cp.iloc[0] = prediction.iloc[0]
    predicted_cp.append(prediction_cp)

    true_outlier.append(y_test.loc[:, "anomaly"])
    true_cp.append(y_test.loc[:, "changepoint"])

plot_results(
    (true_outlier[33], predicted_outlier[33]),
    (true_cp[33], predicted_cp[33]),
)

binary = chp_score(true_outlier, predicted_outlier, metric="binary")

add = chp_score(
    true_cp,
    predicted_cp,
    metric="average_time",
    window_width="60s",
    anomaly_window_destination="righter",
)

nab = chp_score(
    true_cp,
    predicted_cp,
    metric="nab",
    window_width="60s",
    anomaly_window_destination="righter",
)