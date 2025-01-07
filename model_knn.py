import os
import numpy as np
import scipy.stats as SS
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


class KNN:
    def __init__(
            self,
            n_neighbors=5,
            scaling=False,
            p_value=0.999,
    ):
        """
        Inicjalizacja detektora anomalii opartego na KNN.

        Parameters:
        -----------
        n_neighbors : int, default=5
            Liczba sąsiadów do uwzględnienia
        scaling : bool, default=False
            Czy standaryzować dane
        p_value : float, default=0.999
            Poziom ufności dla progu wykrywania anomalii
        """
        self.n_neighbors = n_neighbors
        self.scaling = scaling
        self.p_value = p_value
        self.knn = NearestNeighbors(n_neighbors=n_neighbors)

    def _calculate_distances(self, x):
        """Obliczanie odległości do k-najbliższych sąsiadów"""
        distances, _ = self.knn.kneighbors(x)
        # Bierzemy średnią odległość do k sąsiadów
        return np.mean(distances, axis=1)

    def _calculate_ucl(self, distances):
        """Obliczanie górnej granicy kontrolnej (UCL)"""
        linspace = np.linspace(0, max(distances) * 2, 10000)
        # Zakładamy rozkład chi-kwadrat dla odległości
        c_alpha = linspace[SS.chi2.cdf(linspace, df=self.n_features) < self.p_value][-1]
        self.ucl = c_alpha

    def plot_scores(self, scores=None, ucl=None, save_fig=False, fig_name="KNN"):
        """
        Wizualizacja wyników detekcji anomalii.

        Parameters:
        -----------
        scores : pandas.DataFrame, default=None
            Wyniki (odległości) do najbliższych sąsiadów
        ucl : float, default=None
            Górna granica kontrolna
        save_fig : bool, default=False
            Czy zapisać wykres
        fig_name : str, default='KNN'
            Nazwa pliku wykresu
        """
        if scores is None:
            scores = self.scores
        if ucl is None:
            ucl = self.ucl

        plt.figure(figsize=(12, 4))
        plt.plot(scores, label="KNN distance score")
        plt.grid(True)
        plt.axhline(ucl, zorder=10, color="r", label="UCL")
        plt.ylim(0, 3 * max(scores.min().values, ucl))
        plt.xlim(scores.index.values[0], scores.index.values[-1])
        plt.title("KNN Distance Score Chart")
        plt.xlabel("Time")
        plt.ylabel("Distance Score")
        plt.legend()
        plt.tight_layout()

        if save_fig:
            self._save(name=fig_name)

    @staticmethod
    def _save(name="", fmt="png"):
        """Zapisywanie wykresu do pliku"""
        pwd = os.getcwd()
        iPath = pwd + "/pictures/"
        if not os.path.exists(iPath):
            os.mkdir(iPath)
        os.chdir(iPath)
        plt.savefig(f"{name}.{fmt}", fmt="png", dpi=150, bbox_inches="tight")
        os.chdir(pwd)

    def fit(self, x):
        """
        Trenowanie detektora na danych treningowych.

        Parameters:
        -----------
        x : pandas.DataFrame
            Zbiór treningowy
        """
        x = x.copy()

        # Usuwanie stałych kolumn
        initial_cols_number = len(x.columns)
        x = x.loc[:, (x != x.iloc[0]).any()]
        self._feature_names_in = x.columns
        if initial_cols_number > len(x.columns):
            print("Usunięto stałe kolumny")

        self.n_features = len(x.columns)

        if self.scaling:
            self.scaler = StandardScaler()
            self.scaler.fit(x)
            x_ = self.scaler.transform(x)
        else:
            x_ = x.values

        # Trenowanie modelu KNN
        self.knn.fit(x_)

        # Obliczanie odległości dla danych treningowych
        distances = self._calculate_distances(x_)

        # Obliczanie progu UCL
        self._calculate_ucl(distances)

    def predict(self, x, plot_fig=True, save_fig=False, fig_name="KNN", window_size=1):
        """
        Wykrywanie anomalii w nowych danych.

        Parameters:
        -----------
        x : pandas.DataFrame
            Dane do analizy
        plot_fig : bool, default=True
            Czy generować wykres
        save_fig : bool, default=False
            Czy zapisać wykres
        fig_name : str, default='KNN'
            Nazwa pliku wykresu
        window_size : int, default=1
            Rozmiar okna do wygładzania wyników

        Returns:
        --------
        pandas.DataFrame
            Wyniki detekcji anomalii (True/False)
        """
        x = x.copy()
        x = x.loc[:, self._feature_names_in]

        if self.scaling:
            x_ = self.scaler.transform(x)
        else:
            x_ = x.values

        # Obliczanie odległości
        distances = self._calculate_distances(x_)

        # Wygładzanie wyników oknem przesuwnym
        self.scores = DataFrame(
            distances,
            index=x.index,
            columns=["KNN_score"]
        ).rolling(window_size).median()

        # Wykrywanie anomalii
        anomalies = self.scores > self.ucl

        if plot_fig:
            self.plot_scores(save_fig=save_fig, fig_name=fig_name)

        return anomalies