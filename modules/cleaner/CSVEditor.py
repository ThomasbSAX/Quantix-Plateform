"""
CSVEditor - Éditeur CSV avancé avec fonctionnalités professionnelles
Permet toutes les opérations : ajouter/supprimer lignes/colonnes, trier, filtrer, transformations, etc.
Compatible interface web avec retour JSON et gestion d'erreurs robuste.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import shutil
import json
import warnings
from typing import Dict, List, Any, Optional, Union, Tuple, Literal
from pathlib import Path
import csv
import io


class CSVEditor:
    """
    Éditeur CSV professionnel avec fonctionnalités avancées et compatibilité web.
    """
    
    def __init__(
        self, 
        csv_path: Union[str, Path, pd.DataFrame], 
        *,
        encoding: str = "utf-8", 
        delimiter: str = ",",
        auto_detect: bool = True,
        web_mode: bool = False
    ):
        """
        Initialise l'éditeur CSV avec détection automatique et gestion d'erreurs.
        
        Args:
            csv_path: Chemin vers CSV, DataFrame, ou contenu string
            encoding: Encodage du fichier
            delimiter: Délimiteur CSV
            auto_detect: Détection automatique des paramètres
            web_mode: Mode web pour retour JSON
        """
        self.web_mode = web_mode
        self.encoding = encoding
        self.delimiter = delimiter
        self.csv_path = None
        self.metadata = {
            "created": datetime.now().isoformat(),
            "last_modified": None,
            "operations_count": 0,
            "original_shape": None
        }
        
        # Chargement intelligent des données
        if isinstance(csv_path, pd.DataFrame):
            self.df = csv_path.copy()
            self.csv_path = None
        elif isinstance(csv_path, str) and not os.path.exists(csv_path):
            # Contenu CSV en string
            if csv_path.strip().startswith('"') or ',' in csv_path or '\n' in csv_path:
                self.df = self._load_from_string(csv_path, auto_detect)
            else:
                raise FileNotFoundError(f"Fichier non trouvé: {csv_path}")
        else:
            # Fichier CSV
            self.csv_path = Path(csv_path) if csv_path else None
            self.df = self._load_csv_smart(csv_path, auto_detect)
        
        self.metadata["original_shape"] = self.df.shape
        self.history = []  # Historique pour undo/redo
        self.redo_stack = []
        self.max_history = 100
        self._save_state("initial_load")
    
    def _load_csv_smart(self, path: Union[str, Path], auto_detect: bool = True) -> pd.DataFrame:
        """Chargement intelligent avec détection automatique."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {path}")
        
        # Détection automatique de l'encodage et délimiteur
        if auto_detect:
            encoding, delimiter = self._detect_csv_parameters(path)
            self.encoding = encoding
            self.delimiter = delimiter
        
        try:
            # Tentative de chargement avec paramètres détectés
            df = pd.read_csv(
                path, 
                encoding=self.encoding, 
                delimiter=self.delimiter,
                dtype=str,  # Charger tout en string pour préserver les données
                keep_default_na=False  # Éviter la conversion automatique des "NA"
            )
            
            # Post-traitement : réconversion intelligente des types
            df = self._smart_type_inference(df)
            
            return df
            
        except Exception as e:
            # Fallbacks progressifs
            fallback_configs = [
                {"encoding": "utf-8", "delimiter": ","},
                {"encoding": "latin1", "delimiter": ","},
                {"encoding": "utf-8", "delimiter": ";"},
                {"encoding": "latin1", "delimiter": ";"},
                {"encoding": "utf-8", "delimiter": "\t"},
            ]
            
            for config in fallback_configs:
                try:
                    return pd.read_csv(path, **config, dtype=str, keep_default_na=False)
                except:
                    continue
            
            raise ValueError(f"Impossible de lire le fichier CSV: {e}")
    
    def _detect_csv_parameters(self, path: Path) -> Tuple[str, str]:
        """Détecte automatiquement l'encodage et le délimiteur."""
        # Détection d'encodage
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        detected_encoding = 'utf-8'
        
        for enc in encodings:
            try:
                with open(path, 'r', encoding=enc) as f:
                    f.read(1024)  # Test de lecture
                detected_encoding = enc
                break
            except UnicodeDecodeError:
                continue
        
        # Détection du délimiteur
        with open(path, 'r', encoding=detected_encoding) as f:
            sample = f.read(4096)
            
        sniffer = csv.Sniffer()
        try:
            delimiter = sniffer.sniff(sample).delimiter
        except:
            # Fallback: compter les délimiteurs possibles
            delim_counts = {
                ',': sample.count(','),
                ';': sample.count(';'),
                '\t': sample.count('\t'),
                '|': sample.count('|')
            }
            delimiter = max(delim_counts, key=delim_counts.get)
        
        return detected_encoding, delimiter
    
    def _load_from_string(self, content: str, auto_detect: bool = True) -> pd.DataFrame:
        """Charge un DataFrame depuis une string CSV."""
        if auto_detect:
            # Détection du délimiteur dans la string
            sniffer = csv.Sniffer()
            try:
                self.delimiter = sniffer.sniff(content[:1000]).delimiter
            except:
                # Fallback
                delim_counts = {
                    ',': content.count(','),
                    ';': content.count(';'),
                    '\t': content.count('\t')
                }
                self.delimiter = max(delim_counts, key=delim_counts.get)
        
        return pd.read_csv(
            io.StringIO(content),
            delimiter=self.delimiter,
            dtype=str,
            keep_default_na=False
        )
    
    def _smart_type_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inférence intelligente des types après chargement."""
        result = df.copy()
        
        for col in result.columns:
            # Tentative de conversion numérique
            try:
                # Nettoyer et tenter la conversion
                cleaned = result[col].str.strip().replace('', pd.NA)
                numeric = pd.to_numeric(cleaned, errors='coerce')
                
                # Si plus de 80% des valeurs sont numériques, convertir
                if numeric.notna().sum() / len(cleaned.dropna()) > 0.8:
                    result[col] = numeric
                    continue
            except:
                pass
            
            # Tentative de conversion datetime
            try:
                datetime_col = pd.to_datetime(result[col], errors='coerce', infer_datetime_format=True)
                if datetime_col.notna().sum() / len(result[col].dropna()) > 0.8:
                    result[col] = datetime_col
                    continue
            except:
                pass
            
            # Garder comme string mais nettoyer
            result[col] = result[col].astype(str).replace('nan', pd.NA)
        
        return result
    
    def _save_state(self, operation: str):
        """Sauvegarde l'état dans l'historique avec description."""
        if len(self.history) >= self.max_history:
            self.history.pop(0)
        
        self.history.append({
            "data": self.df.copy(),
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "shape": self.df.shape
        })
        
        # Clear redo stack when new operation
        self.redo_stack.clear()
        self.metadata["operations_count"] += 1
        self.metadata["last_modified"] = datetime.now().isoformat()
    
    def undo(self) -> Dict[str, Any]:
        """Annule la dernière modification avec information de retour."""
        if len(self.history) < 2:  # Garder au moins l'état initial
            return self._response(False, "Aucune opération à annuler")
        
        # Sauvegarder l'état actuel dans redo
        current_state = {
            "data": self.df.copy(),
            "operation": "undo",
            "timestamp": datetime.now().isoformat(),
            "shape": self.df.shape
        }
        self.redo_stack.append(current_state)
        
        # Restaurer l'état précédent
        previous_state = self.history.pop()
        self.df = previous_state["data"]
        
        return self._response(True, f"Annulation: {previous_state['operation']}", {
            "restored_operation": previous_state["operation"],
            "new_shape": self.df.shape
        })
    
    def redo(self) -> Dict[str, Any]:
        """Refait la dernière opération annulée."""
        if not self.redo_stack:
            return self._response(False, "Aucune opération à refaire")
        
        # Sauvegarder l'état actuel
        self._save_state("before_redo")
        
        # Restaurer depuis redo stack
        redo_state = self.redo_stack.pop()
        self.df = redo_state["data"]
        
        return self._response(True, "Opération refaite", {
            "restored_shape": self.df.shape
        })
    
    def _response(self, success: bool, message: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Formate la réponse selon le mode (web ou console)."""
        response = {
            "success": success,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "current_shape": self.df.shape if hasattr(self, 'df') else None
        }
        
        if data:
            response.update(data)
        
        if not self.web_mode:
            # Mode console : afficher directement
            status = "✅" if success else "❌"
            print(f"{status} {message}")
            if data:
                for key, value in data.items():
                    print(f"   {key}: {value}")
        
        return response
        
        self.df.to_csv(self.csv_path, index=False)
        print(f"✅ CSV sauvegardé: {self.csv_path}")
    
    # ========== GESTION DES LIGNES ==========
    
    def add_row(self, data=None, position=None):
        """
        Ajoute une nouvelle ligne.
        
        Args:
            data: Dict {colonne: valeur} ou None pour ligne vide
            position: Index où insérer (None = à la fin)
        """
        self.save_state()
        
        if data is None:
            data = {col: None for col in self.df.columns}
        
        new_row = pd.DataFrame([data])
        
        if position is None:
            self.df = pd.concat([self.df, new_row], ignore_index=True)
        else:
            self.df = pd.concat([
                self.df.iloc[:position],
                new_row,
                self.df.iloc[position:]
            ], ignore_index=True)
        
        return len(self.df) - 1 if position is None else position
    
    def delete_row(self, index):
        """
        Supprime une ligne par index.
        
        Args:
            index: Index de la ligne à supprimer
        """
        self.save_state()
        self.df = self.df.drop(index).reset_index(drop=True)
    
    def delete_rows(self, indices):
        """
        Supprime plusieurs lignes.
        
        Args:
            indices: Liste d'indices
        """
        self.save_state()
        self.df = self.df.drop(indices).reset_index(drop=True)
    
    def duplicate_row(self, index):
        """
        Duplique une ligne.
        
        Args:
            index: Index de la ligne à dupliquer
        """
        self.save_state()
        row_to_duplicate = self.df.iloc[index].copy()
        new_row = pd.DataFrame([row_to_duplicate])
        self.df = pd.concat([
            self.df.iloc[:index+1],
            new_row,
            self.df.iloc[index+1:]
        ], ignore_index=True)
        
        return index + 1
    
    def move_row(self, from_index, to_index):
        """
        Déplace une ligne d'une position à une autre.
        
        Args:
            from_index: Position actuelle
            to_index: Nouvelle position
        """
        self.save_state()
        row = self.df.iloc[from_index].copy()
        self.df = self.df.drop(from_index).reset_index(drop=True)
        
        row_df = pd.DataFrame([row])
        self.df = pd.concat([
            self.df.iloc[:to_index],
            row_df,
            self.df.iloc[to_index:]
        ], ignore_index=True)
    
    # ========== GESTION DES COLONNES ==========
    
    def add_column(self, name, default_value=None, position=None, dtype=None):
        """
        Ajoute une nouvelle colonne.
        
        Args:
            name: Nom de la colonne
            default_value: Valeur par défaut
            position: Index où insérer (None = à la fin)
            dtype: Type de données ('int', 'float', 'str', 'date')
        """
        self.save_state()
        
        # Vérifier que le nom n'existe pas
        if name in self.df.columns:
            name = f"{name}_copy"
        
        # Créer la colonne
        if position is None:
            self.df[name] = default_value
        else:
            cols = list(self.df.columns)
            cols.insert(position, name)
            self.df[name] = default_value
            self.df = self.df[cols]
        
        # Appliquer le type
        if dtype:
            if dtype == 'int':
                self.df[name] = pd.to_numeric(self.df[name], errors='coerce').fillna(0).astype(int)
            elif dtype == 'float':
                self.df[name] = pd.to_numeric(self.df[name], errors='coerce')
            elif dtype == 'date':
                self.df[name] = pd.to_datetime(self.df[name], errors='coerce')
            elif dtype == 'str':
                self.df[name] = self.df[name].astype(str)
        
        return name
    
    def delete_column(self, column):
        """
        Supprime une colonne.
        
        Args:
            column: Nom de la colonne
        """
        self.save_state()
        if column in self.df.columns:
            self.df = self.df.drop(columns=[column])
    
    def delete_columns(self, columns):
        """
        Supprime plusieurs colonnes.
        
        Args:
            columns: Liste de noms de colonnes
        """
        self.save_state()
        existing_cols = [col for col in columns if col in self.df.columns]
        if existing_cols:
            self.df = self.df.drop(columns=existing_cols)
    
    def rename_column(self, old_name, new_name):
        """
        Renomme une colonne.
        
        Args:
            old_name: Nom actuel
            new_name: Nouveau nom
        """
        self.save_state()
        if old_name in self.df.columns:
            self.df = self.df.rename(columns={old_name: new_name})
    
    def duplicate_column(self, column, new_name=None):
        """
        Duplique une colonne.
        
        Args:
            column: Nom de la colonne à dupliquer
            new_name: Nom de la nouvelle colonne (auto si None)
        """
        self.save_state()
        if column not in self.df.columns:
            raise ValueError(f"Colonne '{column}' introuvable")
        
        if new_name is None:
            new_name = f"{column}_copy"
            counter = 1
            while new_name in self.df.columns:
                new_name = f"{column}_copy{counter}"
                counter += 1
        
        self.df[new_name] = self.df[column].copy()
        return new_name
    
    def move_column(self, column, position):
        """
        Déplace une colonne à une nouvelle position.
        
        Args:
            column: Nom de la colonne
            position: Nouvelle position (0 = début)
        """
        self.save_state()
        if column not in self.df.columns:
            return
        
        cols = list(self.df.columns)
        cols.remove(column)
        cols.insert(position, column)
        self.df = self.df[cols]
    
    def reorder_columns(self, new_order):
        """
        Réorganise toutes les colonnes.
        
        Args:
            new_order: Liste avec nouvel ordre des colonnes
        """
        self.save_state()
        if set(new_order) == set(self.df.columns):
            self.df = self.df[new_order]
    
    # ========== TRI ET FILTRAGE ==========
    
    def sort_by(self, column, ascending=True):
        """
        Trie par colonne.
        
        Args:
            column: Nom de la colonne
            ascending: True pour croissant, False pour décroissant
        """
        self.save_state()
        if column in self.df.columns:
            self.df = self.df.sort_values(by=column, ascending=ascending).reset_index(drop=True)
    
    def sort_by_multiple(self, columns_order):
        """
        Trie par plusieurs colonnes.
        
        Args:
            columns_order: Liste de tuples (colonne, ascending)
        """
        self.save_state()
        columns = [col for col, _ in columns_order]
        ascending = [asc for _, asc in columns_order]
        
        if all(col in self.df.columns for col in columns):
            self.df = self.df.sort_values(by=columns, ascending=ascending).reset_index(drop=True)
    
    def filter_rows(self, condition):
        """
        Filtre les lignes selon une condition.
        
        Args:
            condition: Fonction lambda ou condition pandas
        
        Example:
            editor.filter_rows(lambda df: df['salary'] > 50000)
        """
        self.save_state()
        self.df = self.df[condition(self.df)].reset_index(drop=True)
    
    def filter_by_value(self, column, value, operator='=='):
        """
        Filtre par valeur simple.
        
        Args:
            column: Nom de la colonne
            value: Valeur à filtrer
            operator: '==', '!=', '>', '<', '>=', '<='
        """
        self.save_state()
        if column not in self.df.columns:
            return
        
        if operator == '==':
            self.df = self.df[self.df[column] == value].reset_index(drop=True)
        elif operator == '!=':
            self.df = self.df[self.df[column] != value].reset_index(drop=True)
        elif operator == '>':
            self.df = self.df[self.df[column] > value].reset_index(drop=True)
        elif operator == '<':
            self.df = self.df[self.df[column] < value].reset_index(drop=True)
        elif operator == '>=':
            self.df = self.df[self.df[column] >= value].reset_index(drop=True)
        elif operator == '<=':
            self.df = self.df[self.df[column] <= value].reset_index(drop=True)
    
    def remove_duplicates(self, columns=None, keep='first'):
        """
        Supprime les doublons.
        
        Args:
            columns: Colonnes à considérer (None = toutes)
            keep: 'first', 'last' ou False
        """
        self.save_state()
        self.df = self.df.drop_duplicates(subset=columns, keep=keep).reset_index(drop=True)
    
    # ========== OPÉRATIONS SUR CELLULES ==========
    
    def update_cell(self, row, column, value):
        """
        Met à jour une cellule.
        
        Args:
            row: Index de la ligne
            column: Nom de la colonne
            value: Nouvelle valeur
        """
        self.save_state()
        if column in self.df.columns:
            self.df.at[row, column] = value
    
    def fill_column(self, column, value):
        """
        Remplit toute une colonne avec une valeur.
        
        Args:
            column: Nom de la colonne
            value: Valeur à appliquer
        """
        self.save_state()
        if column in self.df.columns:
            self.df[column] = value
    
    def fill_range(self, column, start_row, end_row, value):
        """
        Remplit une plage de cellules.
        
        Args:
            column: Nom de la colonne
            start_row: Ligne de début
            end_row: Ligne de fin (incluse)
            value: Valeur à appliquer
        """
        self.save_state()
        if column in self.df.columns:
            self.df.loc[start_row:end_row, column] = value
    
    # ========== IMPORT / EXPORT ==========
    
    def export_to_csv(self, path, **kwargs):
        """Exporte vers CSV."""
        self.df.to_csv(path, index=False, **kwargs)
    
    def export_to_excel(self, path, sheet_name='Data'):
        """Exporte vers Excel."""
        self.df.to_excel(path, sheet_name=sheet_name, index=False)
    
    def export_to_json(self, path, orient='records'):
        """Exporte vers JSON."""
        self.df.to_json(path, orient=orient, indent=2)
    
    def import_from_csv(self, path, replace=False):
        """
        Importe depuis CSV.
        
        Args:
            path: Chemin du fichier
            replace: Si True, remplace tout. Si False, concatène
        """
        self.save_state()
        new_df = pd.read_csv(path)
        
        if replace:
            self.df = new_df
        else:
            self.df = pd.concat([self.df, new_df], ignore_index=True)
    
    # ========== STATISTIQUES ==========
    
    def get_column_stats(self, column):
        """
        Obtient les statistiques d'une colonne.
        
        Returns:
            dict avec count, mean, std, min, max, etc.
        """
        if column not in self.df.columns:
            return {}
        
        stats = {}
        col_data = self.df[column]
        
        stats['count'] = len(col_data)
        stats['missing'] = col_data.isna().sum()
        stats['unique'] = col_data.nunique()
        
        if pd.api.types.is_numeric_dtype(col_data):
            stats['mean'] = col_data.mean()
            stats['std'] = col_data.std()
            stats['min'] = col_data.min()
            stats['max'] = col_data.max()
            stats['median'] = col_data.median()
        
        return stats
    
    def get_summary(self):
        """
        Obtient un résumé complet du dataset.
        
        Returns:
            dict avec infos générales
        """
        return {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'column_names': list(self.df.columns),
            'column_types': self.df.dtypes.astype(str).to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'missing_total': self.df.isna().sum().sum()
        }
    
    # ========== NORMALISATION ==========
    
    def normalize_column(self, column, min_value=None, max_value=None):
        """
        Normalise une colonne numérique dans une plage [min, max].
        
        Args:
            column: Nom de la colonne
            min_value: Valeur minimum cible (None = utilise le min de la colonne)
            max_value: Valeur maximum cible (None = utilise le max de la colonne)
            
        Returns:
            dict avec infos sur la normalisation
        """
        self.save_state()
        
        if column not in self.df.columns:
            raise ValueError(f"Colonne '{column}' introuvable")
        
        # Vérifier si la colonne est numérique
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            # Tenter de convertir
            try:
                self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
            except:
                raise ValueError(f"Colonne '{column}' n'est pas numérique")
        
        # Sauvegarder les valeurs originales
        original_values = self.df[column].copy()
        
        # Obtenir min/max actuels
        col_min = self.df[column].min()
        col_max = self.df[column].max()
        
        # Définir la plage cible
        target_min = min_value if min_value is not None else 0
        target_max = max_value if max_value is not None else 1
        
        # Normalisation min-max
        # Formule: (x - min) / (max - min) * (target_max - target_min) + target_min
        if col_max != col_min:
            self.df[column] = ((self.df[column] - col_min) / (col_max - col_min)) * (target_max - target_min) + target_min
        else:
            # Si toutes les valeurs sont identiques
            self.df[column] = target_min
        
        # Compter les valeurs modifiées
        values_modified = (original_values != self.df[column]).sum()
        
        return {
            'column': column,
            'original_min': float(col_min),
            'original_max': float(col_max),
            'target_min': float(target_min),
            'target_max': float(target_max),
            'values_modified': int(values_modified)
        }
