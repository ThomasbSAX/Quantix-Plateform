
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (norm, expon, uniform, lognorm, gamma, beta)
from scipy.stats import gaussian_kde
from scipy.stats import entropy
import numpy.polynomial.polynomial as poly
from typing import Dict



class DistributionApproximator:
    """
    Classe d'approximation de distribution.
    - Nettoie et convertit les données en valeurs numériques.
    - Normalise entre [0,1].
    - Produit la densité empirique et une approximation polynomiale.
    - (Visualisation supprimée) Retourne uniquement la formule approchée.
    - Retourne la formule approchée uniquement en LaTeX.
    """

    def __init__(self, df, column, degree=6):
        self.df = df
        self.column = column
        self.degree = degree
        self.data = None
        self.coefs = None
        self.latex_formula = None

    def _prepare_data(self):
        """Convertit en numérique et normalise entre 0 et 1."""
        series = pd.to_numeric(self.df[self.column], errors="coerce").dropna()
        series = (series - series.min()) / (series.max() - series.min())
        self.data = series.to_numpy()

    def _fit_polynomial(self):
        """Ajuste un polynôme sur la densité empirique KDE."""
        kde = stats.gaussian_kde(self.data)
        xs = np.linspace(0, 1, 300)
        ys = kde(xs)
        self.coefs = poly.polyfit(xs, ys, self.degree)
        ys_fit = poly.polyval(xs, self.coefs)
        return xs, ys, ys_fit

    def _latex_formula(self):
        """Construit la formule polynomiale en LaTeX."""
        terms = [f"{c:.5f} x^{{{i}}}" for i, c in enumerate(self.coefs)]
        self.latex_formula = "f(x) = " + " + ".join(terms)
        return self.latex_formula

    def run(self):
        """Pipeline complet : préparation, ajustement, sortie (sans affichage)."""
        self._prepare_data()
        xs, ys, ys_fit = self._fit_polynomial()
        formula = self._latex_formula()

        return formula

class DistributionComparator:
    """
    Compare une colonne numérique d'un DataFrame :
    1) Avec une autre colonne
    2) Avec une distribution théorique (normale, exponentielle, etc.)

    Fournit :
    - KL Divergence
    - Jensen-Shannon Distance
    - Covariance, corrélation de Pearson (pour comparaison de deux séries)
    """

    available_distributions = {
        "normal": norm,
        "exponential": expon,
        "uniform": uniform,
        "lognormal": lognorm,
        "gamma": gamma,
        "beta": beta
    }

    def __init__(self, df: pd.DataFrame, column: str, bins: int = 100):
        self.df = df
        self.column = column
        self.bins = bins

        self.data = None
        self.hist = None
        self.kde = None
        self.bin_edges = None

    def _prepare_data(self):
        """Convertit en valeurs numériques, drop les NaN."""
        self.data = pd.to_numeric(self.df[self.column], errors="coerce").dropna().to_numpy()

    def _compute_histogram(self):
        """Construit un histogramme normalisé."""
        hist, bin_edges = np.histogram(self.data, bins=self.bins, density=True)
        epsilon = 1e-10
        self.hist = hist + epsilon
        self.bin_edges = bin_edges

    def generate_theoretical(self, dist_name: str, **params) -> np.ndarray:
        """
        Génère une densité théorique en utilisant scipy.stats.
        
        Args:
            dist_name: nom de distribution (ex: "normal", "exponential")
            params: paramètres (par ex mu, sigma pour norm)
        """
        if dist_name not in self.available_distributions:
            raise ValueError(f"Distribution '{dist_name}' non supportée.")

        dist = self.available_distributions[dist_name](**params)
        centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        return dist.pdf(centers)

    def _compute_kde(self):
        """Calcule une KDE sur les données."""
        self.kde = gaussian_kde(self.data)

    def kullback_leibler(self, other_hist: np.ndarray) -> float:
        """KL divergence entre histogrammes."""
        return entropy(self.hist, other_hist)

    def jensen_shannon(self, other_hist: np.ndarray) -> float:
        """Jensen-Shannon distance symétrisée."""
        m = 0.5 * (self.hist + other_hist)
        return 0.5 * (entropy(self.hist, m) + entropy(other_hist, m))

    def compare_with_column(self, other_col: str) -> Dict[str, float]:
        """Compare avec une autre colonne."""
        other_data = pd.to_numeric(self.df[other_col], errors="coerce").dropna().to_numpy()
        other_hist, _ = np.histogram(other_data, bins=self.bin_edges, density=True)
        other_hist += 1e-10

        return {
            "kl_divergence": self.kullback_leibler(other_hist),
            "jensen_shannon": self.jensen_shannon(other_hist),
            "covariance": np.cov(self.data, other_data)[0, 1],
            "pearson_correlation": np.corrcoef(self.data, other_data)[0, 1]
        }

    def compare_with_theoretical(self, dist_name: str, **params) -> Dict[str, float]:
        """Compare avec une distribution théorique."""
        theoretical_pdf = self.generate_theoretical(dist_name, **params)
        return {
            "kl_divergence": self.kullback_leibler(theoretical_pdf),
            "jensen_shannon": self.jensen_shannon(theoretical_pdf)
        }

    def run(self):
        """Pipeline complet de préparation."""
        self._prepare_data()
        self._compute_histogram()
        self._compute_kde()
        return self.hist, self.kde
