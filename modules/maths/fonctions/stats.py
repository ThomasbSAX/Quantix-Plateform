import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from scipy.optimize import curve_fit
from typing import Callable, Dict, List, Optional, Any, Union, Literal

class StatisticalAnalyzer:
    """
    Classe pour réaliser des analyses et tests statistiques standard
    sur les colonnes de DataFrame pandas.

    Fonctionnalités :
    - Tests de corrélation (Pearson, Spearman, Kendall)
    - Tests de normalité (Shapiro, KS)
    - Tests de comparaison de groupes (ANOVA, t-test, Mann-Whitney)
    - Test d'indépendance (Chi2, Fisher exact)
    - Régressions (linéaire et logistique)
    - R2, RMSE, etc.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _clean_numeric(self, series: pd.Series) -> np.ndarray:
        """ Nettoie et convertit une série en numérique sans NaN """
        return pd.to_numeric(series, errors='coerce').dropna().values

    def correlation_test(self, col1: str, col2: str, method='pearson') -> Dict[str, Any]:
        """ Test de corrélation entre deux colonnes """
        x = self._clean_numeric(self.df[col1])
        y = self._clean_numeric(self.df[col2])

        if method == 'pearson':
            stat, pval = stats.pearsonr(x, y)
        elif method == 'spearman':
            stat, pval = stats.spearmanr(x, y)
        elif method == 'kendall':
            stat, pval = stats.kendalltau(x, y)
        else:
            raise ValueError("Méthode non supportée : pearson, spearman, kendall")

        return {
            "test": f"{method.capitalize()} correlation",
            "columns": (col1, col2),
            "statistic": stat,
            "p_value": pval,
            "interpretation": "Corrélation significative" if pval < 0.05 else "Corrélation non significative"
        }

    def normality_test(self, col: str, method='shapiro') -> Dict[str, Any]:
        """ Test de normalité pour une colonne """
        x = self._clean_numeric(self.df[col])

        if method == 'shapiro':
            stat, pval = stats.shapiro(x)
        elif method == 'ks':
            stat, pval = stats.kstest(x, 'norm')
        else:
            raise ValueError("Test non supporté : shapiro, ks")

        return {
            "test": f"{method.upper()} normality test",
            "column": col,
            "statistic": stat,
            "p_value": pval,
            "interpretation": "Données semblent normales" if pval > 0.05 else "Non normal"
        }

    def ttest_independent(self, col: str, group_col: str) -> Dict[str, Any]:
        """ Test t de Student pour deux groupes indépendants """
        groups = self.df[group_col].dropna().unique()
        if len(groups) != 2:
            raise ValueError("ttest_independent nécessite exactement 2 groupes")

        group1 = self._clean_numeric(self.df[self.df[group_col] == groups[0]][col])
        group2 = self._clean_numeric(self.df[self.df[group_col] == groups[1]][col])
        stat, pval = stats.ttest_ind(group1, group2)

        return {
            "test": "t-test indépendant",
            "groups": [groups[0], groups[1]],
            "statistic": stat,
            "p_value": pval,
            "interpretation": "Différence significative" if pval < 0.05 else "Pas de différence significative"
        }

    def anova_oneway(self, col: str, group_col: str) -> Dict[str, Any]:
        """ ANOVA à un facteur """
        groups = [self._clean_numeric(group[col]) for name, group in self.df.groupby(group_col)]
        stat, pval = stats.f_oneway(*groups)

        return {
            "test": "ANOVA 1-way",
            "factor": group_col,
            "response": col,
            "statistic": stat,
            "p_value": pval,
            "interpretation": "Différences significatives" if pval < 0.05 else "Pas de différence détectée"
        }

    def chi2_test(self, col1: str, col2: str) -> Dict[str, Any]:
        """ Test de Chi2 pour deux variables catégorielles """
        contingency = pd.crosstab(self.df[col1], self.df[col2])
        stat, pval, dof, _ = stats.chi2_contingency(contingency)

        return {
            "test": "Chi2 d'indépendance",
            "columns": (col1, col2),
            "statistic": stat,
            "p_value": pval,
            "degrees_of_freedom": dof,
            "interpretation": "Variables associées" if pval < 0.05 else "Pas d'association détectée"
        }

    def linear_regression(self, y: str, X: List[str]) -> Dict[str, Any]:
        """ Régression linéaire (OLS) """
        X_data = sm.add_constant(self.df[X].select_dtypes(include=[np.number]))
        y_data = self._clean_numeric(self.df[y])

        model = sm.OLS(y_data, X_data).fit()
        return {
            "model": "OLS Regression",
            "dependent": y,
            "independent": X,
            "summary": model.summary().as_text()
        }

    def logistic_regression(self, y: str, X: List[str]) -> Dict[str, Any]:
        """ Régression logistique pour variable binaire """
        if set(self.df[y].unique()) - {0, 1}:
            raise ValueError("Variable cible doit être binaire (0/1)")

        X_data = sm.add_constant(self.df[X].select_dtypes(include=[np.number]))
        y_data = self.df[y].astype(int)

        model = sm.Logit(y_data, X_data).fit(disp=False)
        return {
            "model": "Logistic Regression",
            "dependent": y,
            "independent": X,
            "summary": model.summary().as_text()
        }
