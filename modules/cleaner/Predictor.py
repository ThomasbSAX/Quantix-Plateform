"""
Predictor - Système d'auto-complétion et prédiction pour données tabulaires
Utilise des modèles légers pour détecter patterns et suggérer valeurs
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

class Predictor:
    """
    Détecte les patterns dans les données et prédit les valeurs manquantes/suivantes.
    Gère les ruptures de séquence causées par suppressions de lignes.
    """
    
    def __init__(self, min_samples=5, confidence_threshold=0.7):
        """
        Args:
            min_samples: Nombre minimum de points pour détecter un pattern
            confidence_threshold: R² minimum pour considérer une prédiction fiable
        """
        self.min_samples = min_samples
        self.confidence_threshold = confidence_threshold
        self.patterns = {}
        
    def detect_pattern_type(self, values):
        """
        Détecte le type de pattern : linéaire, polynomial, constant, etc.
        
        Returns:
            dict: {'type': str, 'model': sklearn model, 'confidence': float}
        """
        if len(values) < self.min_samples:
            return {'type': 'insufficient_data', 'model': None, 'confidence': 0.0}
        
        # Préparer les données - convertir en float
        try:
            y = np.array([float(v) if v is not None else np.nan for v in values])
        except (ValueError, TypeError):
            return {'type': 'non_numeric', 'model': None, 'confidence': 0.0}
        
        X = np.arange(len(values)).reshape(-1, 1)
        
        # Retirer NaN
        mask = ~np.isnan(y)
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) < self.min_samples:
            return {'type': 'insufficient_data', 'model': None, 'confidence': 0.0}
        
        patterns_tested = []
        
        # 1. Test pattern constant (variance faible)
        if np.std(y_clean) < 0.01:
            return {
                'type': 'constant',
                'model': None,
                'confidence': 1.0,
                'value': np.mean(y_clean)
            }
        
        # 2. Test linéaire
        try:
            linear_model = LinearRegression()
            linear_model.fit(X_clean, y_clean)
            y_pred = linear_model.predict(X_clean)
            r2_linear = r2_score(y_clean, y_pred)
            patterns_tested.append({
                'type': 'linear',
                'model': linear_model,
                'confidence': max(0, r2_linear),
                'slope': linear_model.coef_[0]
            })
        except:
            pass
        
        # 3. Test polynomial degré 2
        try:
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(X_clean)
            poly_model = Ridge(alpha=1.0)
            poly_model.fit(X_poly, y_clean)
            y_pred = poly_model.predict(X_poly)
            r2_poly = r2_score(y_clean, y_pred)
            patterns_tested.append({
                'type': 'polynomial',
                'model': poly_model,
                'confidence': max(0, r2_poly),
                'degree': 2,
                'poly_features': poly_features
            })
        except:
            pass
        
        # Sélectionner le meilleur pattern
        if not patterns_tested:
            return {'type': 'none', 'model': None, 'confidence': 0.0}
        
        best_pattern = max(patterns_tested, key=lambda p: p['confidence'])
        
        return best_pattern
    
    def predict_next_value(self, values, n_ahead=1):
        """
        Prédit la ou les prochaines valeurs dans une séquence.
        
        Args:
            values: Liste de valeurs numériques
            n_ahead: Nombre de valeurs à prédire
            
        Returns:
            list: Valeurs prédites avec leur niveau de confiance
        """
        pattern = self.detect_pattern_type(values)
        
        if pattern['confidence'] < self.confidence_threshold:
            return [{
                'value': None,
                'confidence': pattern['confidence'],
                'reason': f"Confiance trop faible ({pattern['confidence']:.2f})"
            }] * n_ahead
        
        predictions = []
        current_length = len(values)
        
        for i in range(n_ahead):
            idx = current_length + i
            
            if pattern['type'] == 'constant':
                predictions.append({
                    'value': pattern['value'],
                    'confidence': pattern['confidence'],
                    'type': 'constant'
                })
                
            elif pattern['type'] == 'linear':
                X_pred = np.array([[idx]])
                value = pattern['model'].predict(X_pred)[0]
                predictions.append({
                    'value': value,
                    'confidence': pattern['confidence'],
                    'type': 'linear',
                    'slope': pattern['slope']
                })
                
            elif pattern['type'] == 'polynomial':
                X_pred = pattern['poly_features'].transform([[idx]])
                value = pattern['model'].predict(X_pred)[0]
                predictions.append({
                    'value': value,
                    'confidence': pattern['confidence'],
                    'type': 'polynomial'
                })
        
        return predictions
    
    def fill_missing_values(self, series):
        """
        Remplit les valeurs manquantes en utilisant interpolation + pattern detection.
        
        Args:
            series: pandas Series avec valeurs manquantes
            
        Returns:
            pandas Series avec valeurs prédites
        """
        result = series.copy()
        
        # Trouver les indices avec valeurs manquantes
        missing_indices = result[result.isna()].index.tolist()
        
        if not missing_indices:
            return result
        
        # Pour chaque valeur manquante
        for idx in missing_indices:
            # Prendre les N valeurs précédentes
            start_idx = max(0, idx - 10)
            context_values = result[start_idx:idx].dropna().tolist()
            
            if len(context_values) >= self.min_samples:
                predictions = self.predict_next_value(context_values, n_ahead=1)
                if predictions[0]['value'] is not None:
                    result.iloc[idx] = predictions[0]['value']
        
        return result
    
    def detect_anomalies_in_sequence(self, values, threshold_std=3):
        """
        Détecte les ruptures de logique dans une séquence.
        
        Args:
            values: Liste de valeurs
            threshold_std: Nombre d'écarts-types pour considérer une anomalie
            
        Returns:
            list: Indices des valeurs anormales
        """
        if len(values) < self.min_samples:
            return []
        
        pattern = self.detect_pattern_type(values)
        
        if pattern['confidence'] < 0.5:
            return []  # Pas assez de pattern pour détecter anomalies
        
        anomalies = []
        
        for i, val in enumerate(values):
            if np.isnan(val):
                continue
            
            # Prédire ce que devrait être cette valeur
            context = values[:i]
            if len(context) >= self.min_samples:
                predictions = self.predict_next_value(context, n_ahead=1)
                predicted = predictions[0]['value']
                
                if predicted is not None:
                    # Calculer l'écart
                    residuals = [values[j] - self.predict_next_value(values[:j], n_ahead=1)[0]['value'] 
                                for j in range(len(context)) if not np.isnan(values[j])]
                    residuals = [r for r in residuals if r is not None]
                    
                    if residuals:
                        std_residual = np.std(residuals)
                        deviation = abs(val - predicted)
                        
                        if deviation > threshold_std * std_residual:
                            anomalies.append({
                                'index': i,
                                'value': val,
                                'expected': predicted,
                                'deviation': deviation,
                                'confidence': predictions[0]['confidence']
                            })
        
        return anomalies
    
    def suggest_value_for_cell(self, df, row_idx, col_name):
        """
        Suggère une valeur pour une cellule spécifique basée sur patterns.
        
        Args:
            df: DataFrame complet
            row_idx: Index de la ligne
            col_name: Nom de la colonne
            
        Returns:
            dict: Suggestion avec confiance et raison
        """
        column_data = df[col_name].copy()
        
        # Essayer prédiction par séquence
        values_before = column_data[:row_idx].dropna().tolist()
        
        if len(values_before) >= self.min_samples:
            predictions = self.predict_next_value(values_before, n_ahead=1)
            if predictions[0]['value'] is not None and predictions[0]['confidence'] > self.confidence_threshold:
                return {
                    'value': predictions[0]['value'],
                    'confidence': predictions[0]['confidence'],
                    'method': 'sequence_prediction',
                    'pattern_type': predictions[0]['type']
                }
        
        # Fallback: moyenne de la colonne
        mean_val = column_data.mean()
        if not np.isnan(mean_val):
            return {
                'value': mean_val,
                'confidence': 0.5,
                'method': 'column_mean',
                'pattern_type': 'statistical'
            }
        
        return {
            'value': None,
            'confidence': 0.0,
            'method': 'none',
            'pattern_type': 'insufficient_data'
        }
    
    def analyze_column_patterns(self, df):
        """
        Analyse tous les patterns détectables dans chaque colonne numérique.
        
        Returns:
            dict: Résumé des patterns par colonne
        """
        results = {}
        
        for col in df.select_dtypes(include=[np.number]).columns:
            values = df[col].dropna().tolist()
            
            if len(values) >= self.min_samples:
                pattern = self.detect_pattern_type(values)
                anomalies = self.detect_anomalies_in_sequence(values)
                
                results[col] = {
                    'pattern_detected': pattern['type'],
                    'confidence': pattern['confidence'],
                    'anomalies_count': len(anomalies),
                    'anomalies': anomalies[:5],  # Limiter à 5
                    'predictable': pattern['confidence'] > self.confidence_threshold
                }
                
                # Ajouter infos spécifiques au pattern
                if pattern['type'] == 'linear':
                    results[col]['slope'] = pattern.get('slope', 0)
                    results[col]['trend'] = 'croissant' if pattern.get('slope', 0) > 0 else 'décroissant'
        
        return results


def generate_prediction_report(df, predictor=None):
    """
    Génère un rapport complet des prédictions possibles.
    """
    if predictor is None:
        predictor = Predictor()
    
    patterns = predictor.analyze_column_patterns(df)
    
    report = {
        'total_columns_analyzed': len(patterns),
        'predictable_columns': sum(1 for p in patterns.values() if p['predictable']),
        'columns_with_anomalies': sum(1 for p in patterns.values() if p['anomalies_count'] > 0),
        'patterns': patterns
    }
    
    return report
