import os
import numpy as np
from matplotlib import pyplot as plt
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
        """
        Inicjalizacja detektora anomalii opartego na SVM.

        Parameters:
        -----------
        kernel : str, default='rbf'
            Typ kernela używanego przez SVM.
        nu : float, default=0.05
            Parametr kontroli odsetka obserwacji zakwalifikowanych jako anomalie.
        gamma : str or float, default='scale'
            Parametr kernela SVM.
        scaling : bool, default=False
            Czy standaryzować dane.
        """
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.scaling = scaling
        self.svm = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)

    def plot_scores(self, scores=None, save_fig=False, fig_name="SVM"):
        """
        Wizualizacja wyników detekcji anomalii.

        Parameters:
        -----------
        scores : pandas.DataFrame, default=None
            Wyniki detekcji (odległości).
        save_fig : bool, default=False
            Czy zapisać wykres.
        fig_name : str, default='SVM'
            Nazwa pliku wykresu.
        """
        if scores is None:
            scores = self.scores

        plt.figure(figsize=(12, 4))
        plt.plot(scores, label="SVM decision function")
        plt.grid(True)
        plt.axhline(0, zorder=10, color="r", label="Decision Boundary")
        plt.title("SVM Decision Function")
        plt.xlabel("Time")
        plt.ylabel("Decision Score")
        plt.legend()
        plt.tight_layout()

        if save_fig:
            self._save(name=fig_name)

    @staticmethod
    def _save(name="", fmt="png"):
        """Zapisywanie wykresu do pliku."""
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
            Zbiór treningowy.
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

        # Trenowanie modelu SVM
        self.svm.fit(x_)

        # Obliczanie funkcji decyzyjnej dla danych treningowych
        decision_scores = self.svm.decision_function(x_)

        # Przechowywanie wyników
        self.scores = DataFrame(
            decision_scores,
            index=x.index,
            columns=["SVM_score"]
        )

    def predict(self, x, plot_fig=True, save_fig=False, fig_name="SVM", window_size=1):
        """
        Wykrywanie anomalii w nowych danych.

        Parameters:
        -----------
        x : pandas.DataFrame
            Dane do analizy.
        plot_fig : bool, default=True
            Czy generować wykres.
        save_fig : bool, default=False
            Czy zapisać wykres.
        fig_name : str, default='SVM'
            Nazwa pliku wykresu.
        window_size : int, default=1
            Rozmiar okna do wygładzania wyników.

        Returns:
        --------
        pandas.DataFrame
            Wyniki detekcji anomalii (True/False).
        """
        x = x.copy()
        x = x.loc[:, self._feature_names_in]

        if self.scaling:
            x_ = self.scaler.transform(x)
        else:
            x_ = x.values

        # Obliczanie funkcji decyzyjnej
        decision_scores = self.svm.decision_function(x_)

        # Wygładzanie wyników oknem przesuwnym
        self.scores = DataFrame(
            decision_scores,
            index=x.index,
            columns=["SVM_score"]
        ).rolling(window_size).median()

        # Wykrywanie anomalii
        anomalies = self.scores < 0  # Decyzja SVM poniżej zera to anomalia

        if plot_fig:
            self.plot_scores(save_fig=save_fig, fig_name=fig_name)

        return anomalies