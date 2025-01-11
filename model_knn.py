import numpy as np
import pandas as pd


class KNN:
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        self.scores = None
        self.ucl = None

    def _validate_data(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.select_dtypes(include=[np.number]).values
        return X.astype(float)

    def _compute_distances(self, X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:

        distances = np.zeros((X_test.shape[0], X_train.shape[0]))

        for i in range(X_test.shape[0]):
            diff = X_train - X_test[i]
            distances[i] = np.sqrt(np.sum(diff * diff, axis=1))

        return distances

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self.X_train = X_train
        self.y_train = y_train

        X_train_valid = self._validate_data(X_train)
        train_distances = self._compute_distances(X_train_valid, X_train_valid)

        sorted_distances = np.sort(train_distances, axis=1)
        train_scores = np.mean(sorted_distances[:, 1:self.n_neighbors + 1], axis=1)

        self.scores = pd.DataFrame(
            train_scores,
            index=X_train.index,
            columns=['KNN_score']
        )

        self.ucl = np.percentile(train_scores, 95)

        return self

    def predict(self, X_test: pd.DataFrame):
        if self.X_train is None:
            raise ValueError("Model must be fitted before making predictions")

        X_test_valid = self._validate_data(X_test)
        X_train_valid = self._validate_data(self.X_train)

        # Oblicz odległości między punktami testowymi a treningowymi
        distances = self._compute_distances(X_train_valid, X_test_valid)

        # Oblicz scores dla danych testowych
        sorted_distances = np.sort(distances, axis=1)
        test_scores = np.mean(sorted_distances[:, :self.n_neighbors], axis=1)

        # Zapisz scores w formacie DataFrame
        self.scores = pd.DataFrame(
            test_scores,
            index=X_test.index,
            columns=['KNN_score']
        )

        return self
