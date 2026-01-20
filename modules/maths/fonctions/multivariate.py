import pandas as pd
import numpy as np


class MultivariateAnalyzer:
    """
    Classe d'analyse multivariée pour données quantitatives/catégorielles mélangées.
    Fournit ACP, ICA, AFC, Régression multivariée (sans visualisation).
    """

    def __init__(self, df, target=None):
        self.df = df.copy()
        self.target = target
        self.numeric = df.select_dtypes(include=['number'])
        self.categorical = df.select_dtypes(include=['object', 'category'])
        self.results = {}

    def run_pca(self, n_components=2, scale=True, show: bool = False):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        X = self.numeric.dropna()

        if scale:
            X = StandardScaler().fit_transform(X)

        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X)
        explained = pca.explained_variance_ratio_

        # Stocker
        self.results['pca'] = {
            'model': pca,
            'components': components,
            'explained_variance_ratio': explained
        }

        # Visualisation supprimée: retourne toujours fig=None
        return self.results['pca'], None

    def run_ica(self, n_components=2, show: bool = False):
        from sklearn.decomposition import FastICA

        X = self.numeric.dropna()
        ica = FastICA(n_components=n_components)
        comps = ica.fit_transform(X)

        self.results['ica'] = {
            'model': ica,
            'components': comps
        }

        # Visualisation supprimée: retourne toujours fig=None
        return self.results['ica'], None

    def run_afc(self, show: bool = False):
        import prince

        tables = []
        for col in self.categorical.columns:
            if self.categorical[col].nunique() <= 30:
                tables.append(col)

        if not tables:
            return None, "Pas de variables catégorielles adéquates"

        X = self.categorical[tables].dropna()
        afc = prince.CA(n_components=2, random_state=42)
        afc = afc.fit(X)

        coords = afc.row_coordinates(X)

        self.results['afc'] = {
            'model': afc,
            'coordinates': coords
        }

        # Visualisation supprimée: retourne toujours fig=None
        return self.results['afc'], None

    def multivariate_ols(self, features=None, target=None):
        import statsmodels.api as sm

        if not target and not self.target:
            raise ValueError("Cible non spécifiée")
        
        y = self.df[target or self.target].dropna()
        X = self.df[features].loc[y.index].dropna()

        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

        self.results['ols'] = model
        return model.summary()

    def generate_report(self):
        report = []

        if 'pca' in self.results:
            p = self.results['pca']
            report.append(f"ACP: {len(p['explained_variance_ratio'])} composantes, variance expl: {p['explained_variance_ratio']}")
        
        if 'ica' in self.results:
            report.append("ICA: calculées sur données numériques")
        
        if 'afc' in self.results:
            report.append("AFC: réalisée sur variables catégorielles")

        if 'ols' in self.results:
            report.append("OLS: régression multivariée disponible via .results['ols']")

        return "\n".join(report)
