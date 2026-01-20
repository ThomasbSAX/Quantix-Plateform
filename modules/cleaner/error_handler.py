"""
Système robuste de gestion d'erreurs pour usage scientifique
Diagnostique les erreurs et fournit des solutions claires
"""

import traceback
from typing import Optional, Any, Dict, Callable
from functools import wraps
import pandas as pd
import numpy as np


class DataQualityError(Exception):
    """Erreur liée à la qualité des données."""
    pass


class ConversionError(Exception):
    """Erreur lors de conversion d'unités ou de types."""
    pass


class ConfigurationError(Exception):
    """Erreur de configuration du système."""
    pass


class ScientificErrorHandler:
    """
    Gestionnaire d'erreurs pour analyses scientifiques.
    
    Fournit:
    - Diagnostic clair de l'erreur
    - Cause probable
    - Solution recommandée
    - Exemple de correction
    
    Usage:
        handler = ScientificErrorHandler()
        result = handler.safe_execute(function, *args, **kwargs)
    """
    
    ERROR_CATALOG = {
        # Erreurs de données
        'empty_dataframe': {
            'cause': 'Le DataFrame est vide (0 lignes)',
            'solution': 'Vérifiez que votre fichier CSV contient des données',
            'example': 'df = pd.read_csv("fichier.csv"); print(len(df))'
        },
        'missing_column': {
            'cause': 'La colonne spécifiée n\'existe pas dans le DataFrame',
            'solution': 'Vérifiez les noms de colonnes disponibles',
            'example': 'print(df.columns.tolist())'
        },
        'invalid_type': {
            'cause': 'Le type de données n\'est pas celui attendu',
            'solution': 'Convertissez les données au bon type avant traitement',
            'example': 'df["colonne"] = pd.to_numeric(df["colonne"], errors="coerce")'
        },
        'all_null': {
            'cause': 'La colonne ne contient que des valeurs nulles',
            'solution': 'Supprimez cette colonne ou remplissez les valeurs manquantes',
            'example': 'df = df.dropna(axis=1, how="all")'
        },
        
        # Erreurs de conversion d'unités
        'unknown_unit': {
            'cause': 'L\'unité spécifiée n\'est pas reconnue',
            'solution': 'Utilisez une unité standard (m, kg, s, K, etc.)',
            'example': 'convert(100, "cm", "m")  # Utilisez "cm" pas "centimetre"'
        },
        'incompatible_units': {
            'cause': 'Les unités ne sont pas de la même dimension physique',
            'solution': 'Vérifiez que vous convertissez des unités compatibles (ex: m->km, pas m->kg)',
            'example': 'convert(1000, "m", "km")  # OK\n# convert(1000, "m", "kg")  # ERREUR'
        },
        'invalid_value': {
            'cause': 'La valeur numérique est invalide (NaN, Inf, ou non numérique)',
            'solution': 'Nettoyez les données avant conversion',
            'example': 'df["col"] = pd.to_numeric(df["col"], errors="coerce")\ndf = df.dropna(subset=["col"])'
        },
        
        # Erreurs de calcul scientifique
        'division_by_zero': {
            'cause': 'Division par zéro détectée',
            'solution': 'Ajoutez une vérification ou remplacez les zéros',
            'example': 'df["ratio"] = df["a"] / df["b"].replace(0, np.nan)'
        },
        'negative_log': {
            'cause': 'Tentative de calculer le logarithme d\'un nombre négatif ou nul',
            'solution': 'Filtrez les valeurs négatives ou nulles avant le calcul',
            'example': 'df_positive = df[df["valeur"] > 0]\nresult = np.log(df_positive["valeur"])'
        },
        'negative_sqrt': {
            'cause': 'Tentative de calculer la racine carrée d\'un nombre négatif',
            'solution': 'Vérifiez les valeurs ou utilisez des nombres complexes',
            'example': 'df_positive = df[df["valeur"] >= 0]\nresult = np.sqrt(df_positive["valeur"])'
        },
        
        # Erreurs de fichiers
        'file_not_found': {
            'cause': 'Le fichier spécifié n\'existe pas',
            'solution': 'Vérifiez le chemin absolu du fichier',
            'example': 'import os; print(os.path.abspath("fichier.csv"))'
        },
        'encoding_error': {
            'cause': 'Problème d\'encodage du fichier',
            'solution': 'Spécifiez l\'encodage correct (utf-8, latin-1, cp1252)',
            'example': 'df = pd.read_csv("fichier.csv", encoding="utf-8")'
        },
        'csv_parsing_error': {
            'cause': 'Erreur de parsing du CSV (séparateur incorrect, guillemets mal fermés)',
            'solution': 'Vérifiez le séparateur et l\'option quoting',
            'example': 'df = pd.read_csv("fichier.csv", sep=";", quoting=csv.QUOTE_MINIMAL)'
        },
        
        # Erreurs de configuration
        'missing_dependency': {
            'cause': 'Bibliothèque Python requise non installée',
            'solution': 'Installez la dépendance manquante',
            'example': 'pip install pandas numpy scipy matplotlib'
        },
        'version_mismatch': {
            'cause': 'Version de bibliothèque incompatible',
            'solution': 'Mettez à jour ou downgrade la bibliothèque',
            'example': 'pip install --upgrade pandas'
        }
    }
    
    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: Si True, affiche les détails complets des erreurs
        """
        self.verbose = verbose
        self.error_log = []
    
    def diagnose_error(self, error: Exception, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Diagnostique une erreur et fournit des recommandations.
        
        Args:
            error: L'exception capturée
            context: Informations contextuelles (fonction, arguments, etc.)
        
        Returns:
            Dict contenant diagnostic, cause, solution, exemple
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Détection du type d'erreur
        diagnosis = {
            'error_type': error_type,
            'error_message': error_msg,
            'traceback': traceback.format_exc() if self.verbose else None,
            'context': context or {}
        }
        
        # Recherche dans le catalogue
        error_key = self._identify_error(error_type, error_msg, context)
        if error_key and error_key in self.ERROR_CATALOG:
            catalog_entry = self.ERROR_CATALOG[error_key]
            diagnosis.update({
                'identified_as': error_key,
                'cause': catalog_entry['cause'],
                'solution': catalog_entry['solution'],
                'example': catalog_entry['example']
            })
        else:
            diagnosis.update({
                'identified_as': 'unknown',
                'cause': 'Erreur non cataloguée',
                'solution': 'Consultez le traceback complet pour plus de détails',
                'example': 'Vérifiez les entrées et la documentation de la fonction'
            })
        
        self.error_log.append(diagnosis)
        return diagnosis
    
    def _identify_error(self, error_type: str, error_msg: str, context: Optional[Dict]) -> Optional[str]:
        """Identifie le type d'erreur à partir du message et du contexte."""
        error_msg_lower = error_msg.lower()
        
        # Erreurs de fichiers
        if 'no such file' in error_msg_lower or 'file not found' in error_msg_lower:
            return 'file_not_found'
        if 'codec' in error_msg_lower or 'decode' in error_msg_lower:
            return 'encoding_error'
        if 'parsing' in error_msg_lower or 'expected' in error_msg_lower:
            return 'csv_parsing_error'
        
        # Erreurs de données
        if error_type == 'KeyError':
            return 'missing_column'
        if 'empty' in error_msg_lower and context and 'dataframe' in str(context).lower():
            return 'empty_dataframe'
        if 'division by zero' in error_msg_lower or 'divide by zero' in error_msg_lower:
            return 'division_by_zero'
        
        # Erreurs de conversion
        if 'unit' in error_msg_lower and ('unknown' in error_msg_lower or 'not recognized' in error_msg_lower):
            return 'unknown_unit'
        if 'incompatible' in error_msg_lower and 'unit' in error_msg_lower:
            return 'incompatible_units'
        
        # Erreurs mathématiques
        if 'log' in error_msg_lower and 'invalid' in error_msg_lower:
            return 'negative_log'
        if 'sqrt' in error_msg_lower and 'invalid' in error_msg_lower:
            return 'negative_sqrt'
        
        # Erreurs de dépendances
        if 'no module named' in error_msg_lower:
            return 'missing_dependency'
        
        return None
    
    def format_diagnosis(self, diagnosis: Dict[str, Any]) -> str:
        """Formate le diagnostic en texte lisible."""
        output = [
            "="*70,
            "ERREUR DÉTECTÉE",
            "="*70,
            f"\n❌ Type: {diagnosis['error_type']}",
            f"❌ Message: {diagnosis['error_message']}\n",
        ]
        
        if diagnosis['identified_as'] != 'unknown':
            output.extend([
                f"🔍 Diagnostic: {diagnosis['identified_as']}",
                f"\n📋 Cause probable:",
                f"   {diagnosis['cause']}",
                f"\n💡 Solution recommandée:",
                f"   {diagnosis['solution']}",
                f"\n📝 Exemple de correction:",
                f"   {diagnosis['example']}\n"
            ])
        
        if diagnosis.get('context'):
            output.extend([
                "📌 Contexte:",
                f"   {diagnosis['context']}\n"
            ])
        
        if self.verbose and diagnosis.get('traceback'):
            output.extend([
                "🔧 Traceback complet:",
                diagnosis['traceback']
            ])
        
        output.append("="*70)
        return "\n".join(output)
    
    def safe_execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Exécute une fonction de manière sécurisée avec gestion d'erreurs.
        
        Args:
            func: Fonction à exécuter
            *args, **kwargs: Arguments de la fonction
        
        Returns:
            Résultat de la fonction ou None en cas d'erreur
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            context = {
                'function': func.__name__,
                'args': str(args)[:200],  # Limité pour lisibilité
                'kwargs': str(kwargs)[:200]
            }
            diagnosis = self.diagnose_error(e, context)
            print(self.format_diagnosis(diagnosis))
            return None
    
    def print_error_summary(self):
        """Affiche un résumé de toutes les erreurs rencontrées."""
        if not self.error_log:
            print("✅ Aucune erreur rencontrée")
            return
        
        print(f"\n📊 Résumé: {len(self.error_log)} erreur(s) rencontrée(s)\n")
        for i, err in enumerate(self.error_log, 1):
            print(f"{i}. {err['error_type']}: {err.get('identified_as', 'unknown')}")


def safe_scientific_operation(verbose: bool = True):
    """
    Décorateur pour sécuriser les opérations scientifiques.
    
    Usage:
        @safe_scientific_operation()
        def ma_fonction(df):
            # code...
            return result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = ScientificErrorHandler(verbose=verbose)
            return handler.safe_execute(func, *args, **kwargs)
        return wrapper
    return decorator


def validate_dataframe(df: pd.DataFrame, required_columns: Optional[list] = None) -> Dict[str, Any]:
    """
    Valide un DataFrame pour usage scientifique.
    
    Args:
        df: DataFrame à valider
        required_columns: Liste des colonnes obligatoires
    
    Returns:
        Dict avec status ('ok' ou 'error') et détails
    """
    issues = []
    
    # Vérification basique
    if df is None:
        return {'status': 'error', 'issues': ['DataFrame est None']}
    
    if len(df) == 0:
        issues.append('DataFrame vide (0 lignes)')
    
    # Colonnes requises
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            issues.append(f'Colonnes manquantes: {missing}')
    
    # Colonnes entièrement nulles
    null_cols = [col for col in df.columns if df[col].isna().all()]
    if null_cols:
        issues.append(f'Colonnes entièrement nulles: {null_cols}')
    
    # Détection de types problématiques
    for col in df.columns:
        if df[col].dtype == 'object':
            # Vérifier si c'est vraiment du texte ou des nombres mal parsés
            try:
                numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
                if numeric_count / len(df) > 0.8:
                    issues.append(f'Colonne "{col}" semble numérique mais stockée comme texte')
            except:
                pass
    
    if issues:
        return {
            'status': 'warning',
            'issues': issues,
            'recommendation': 'Corrigez ces problèmes avant analyse'
        }
    
    return {
        'status': 'ok',
        'message': 'DataFrame valide pour analyse scientifique',
        'shape': df.shape,
        'columns': df.columns.tolist()
    }


def check_numeric_validity(values: pd.Series, allow_negative: bool = True) -> Dict[str, Any]:
    """
    Vérifie la validité de valeurs numériques pour calculs scientifiques.
    
    Args:
        values: Série de valeurs à vérifier
        allow_negative: Si False, signale les valeurs négatives
    
    Returns:
        Dict avec status et détails des problèmes
    """
    issues = []
    
    # Conversion en numérique si nécessaire
    if values.dtype == 'object':
        values = pd.to_numeric(values, errors='coerce')
    
    # Vérifications
    nan_count = values.isna().sum()
    inf_count = np.isinf(values).sum()
    neg_count = (values < 0).sum() if not allow_negative else 0
    zero_count = (values == 0).sum()
    
    if nan_count > 0:
        issues.append(f'{nan_count} valeurs NaN détectées')
    
    if inf_count > 0:
        issues.append(f'{inf_count} valeurs infinies détectées')
    
    if neg_count > 0:
        issues.append(f'{neg_count} valeurs négatives détectées (non autorisées)')
    
    if zero_count > len(values) * 0.5:
        issues.append(f'{zero_count} zéros détectés ({zero_count/len(values)*100:.1f}%)')
    
    return {
        'status': 'ok' if not issues else 'warning',
        'issues': issues,
        'stats': {
            'count': len(values),
            'nan': int(nan_count),
            'inf': int(inf_count),
            'negative': int(neg_count),
            'zero': int(zero_count),
            'valid': int(len(values) - nan_count - inf_count)
        }
    }


# Exemples d'usage
if __name__ == "__main__":
    print("🧪 Tests du système de gestion d'erreurs scientifiques\n")
    
    handler = ScientificErrorHandler(verbose=True)
    
    # Test 1: DataFrame vide
    print("Test 1: DataFrame vide")
    df_empty = pd.DataFrame()
    validation = validate_dataframe(df_empty)
    print(validation)
    print()
    
    # Test 2: Colonne manquante
    print("Test 2: Colonne manquante")
    df = pd.DataFrame({'A': [1, 2, 3]})
    try:
        result = df['B']
    except Exception as e:
        diagnosis = handler.diagnose_error(e, {'dataframe_columns': df.columns.tolist()})
        print(handler.format_diagnosis(diagnosis))
    
    # Test 3: Validation de valeurs numériques
    print("Test 3: Validation numérique")
    values = pd.Series([1.5, 2.3, np.nan, np.inf, -1.2, 0])
    check = check_numeric_validity(values, allow_negative=False)
    print(check)
