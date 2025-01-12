from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class SVM:
    def __init__(
            self,
            kernel="rbf",
            nu=0.05,
            gamma="scale",
            scaling=False,
    ):
        self.scores = None
        self.scaler = None
        self._feature_names_in = None
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.scaling = scaling
        self.svm = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)

    def fit(self, x):
        x = x.copy()

        self._feature_names_in = x.columns

        if self.scaling:
            self.scaler = StandardScaler()
            x_ = self.scaler.fit_transform(x)
        else:
            x_ = x.values

        self.svm.fit(x_)

        decision_scores = self.svm.decision_function(x_)
        self.scores = DataFrame(decision_scores, index=x.index, columns=["SVM_score"])

    def predict(self, x, window_size=1):
        x = x.copy()
        x = x.loc[:, self._feature_names_in]

        if self.scaling:
            x_ = self.scaler.transform(x)
        else:
            x_ = x.values

        decision_scores = self.svm.decision_function(x_)

        #wygładzanie wynikow oknem przesuwnym
        self.scores = DataFrame(
            decision_scores,
            index=x.index,
            columns=["SVM_score"]
        ).rolling(window_size).median()

        anomalies = self.scores < 0  #decyzja SVM ponizej 0 to anomalia

        return anomalies