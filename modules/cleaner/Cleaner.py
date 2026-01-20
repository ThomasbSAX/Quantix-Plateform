"""
Classe Cleaner - Nettoyage modulable et intelligent de CSV/DataFrames
Combine nettoyage classique et détection intelligente de +100 types de données
"""

import pandas as pd
import numpy as np
import re
import os
import json
import csv
import chardet
import io
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any, Tuple, Union
from collections import Counter
import warnings
import logging
import unicodedata
from datetime import datetime
from decimal import Decimal, InvalidOperation
warnings.filterwarnings('ignore')

# Import des modules de nettoyage
from .dropMissing import drop_missing
from .dropMissing import (
    drop_empty_rows,
    drop_empty_columns,
    smart_missing_analysis,
    detect_missing_clusters,
    detect_systematic_missing,
)
from .normalizeSpace import normalize_space
from .normalizeSpace import (
    advanced_normalize_space,
    fix_french_punctuation,
    normalize_numeric_formatting,
    normalize_dataframe_spaces,
)
from .normalizeTypo import normalize_typography
from .normalizeTypo import (
    normalize_dataframe_typography,
    detect_typography_issues,
    create_typography_rules,
    normalize_with_language_rules,
    batch_typography_analysis,
)
from .normalizeText import normalize_text_full
from .normalizeScientNot import normalize_scientific_notation
from .normalizeScientNot import to_scientific_notation
from .normalizeContext import normalize_context
from .removeDuplicates import remove_duplicates
from .detectOutliers import remove_outliers, detect_outliers
from .detectOutliers import (
    detect_multivariate_outliers,
    outlier_analysis_report,
    create_outlier_plots,
    compare_outlier_methods,
    generate_outlier_recommendations,
    quick_outlier_removal,
)
from .convertTypes import infer_and_convert_types, force_numeric
from .convertTypes import (
    analyze_column_content,
    smart_numeric_conversion,
    smart_datetime_conversion,
    smart_boolean_conversion,
    optimize_numeric_dtype,
    optimize_string_dtype,
    force_type_conversion,
    generate_type_report,
)
from .fillMissing import fill_missing
from .fillMissing import (
    smart_fill_missing,
    classic_fill_strategies,
    knn_fill_missing,
    iterative_fill_missing,
    interpolate_missing,
    advanced_missing_analysis,
    create_missing_heatmap,
)
from .trimStrings import trim_strings, remove_empty_strings
from .mean import mean
from .mediane import central_tendency
from .round import round_order
from .logic import detect_sequence_logic_break

# Utilitaires (ré-export orchestrateur)
from .log import set_log_axis
try:
    from .convertUnit import (
        build_unit_mapping,
        convert as convert_unit,
        extract_quantity as extract_unit_quantity,
        get_unit_category,
        list_all_units,
        normalize_unit as normalize_unit_token,
        is_compatible as are_units_compatible,
        convert_multiple as convert_unit_multiple,
    )
except Exception:
    build_unit_mapping = None
    convert_unit = None
    extract_unit_quantity = None
    get_unit_category = None
    list_all_units = None
    normalize_unit_token = None
    are_units_compatible = None
    convert_unit_multiple = None
from .error_handler import (
    DataQualityError,
    ConversionError,
    ConfigurationError,
    ScientificErrorHandler,
    safe_scientific_operation,
    validate_dataframe as validate_dataframe_quality,
    check_numeric_validity,
)

# Modules optionnels/avancés: import safe pour ne pas casser le Cleaner de base
try:
    from .CSVEditor import CSVEditor
except Exception:
    CSVEditor = None

try:
    from .Predictor import Predictor, generate_prediction_report
except Exception:
    Predictor = None
    generate_prediction_report = None

try:
    from .generate_pdf_report import DataQualityReport, create_quality_report
except Exception:
    DataQualityReport = None
    create_quality_report = None

try:
    from .analyse import LLMConfig, DescribeConfig, QuantixCoherenceDescriber, describe_llm
except Exception:
    LLMConfig = None
    DescribeConfig = None
    QuantixCoherenceDescriber = None
    describe_llm = None

# Bibliothèques avancées pour détection intelligente
try:
    import phonenumbers
    from phonenumbers import geocoder, carrier
    PHONENUMBERS_AVAILABLE = True
except ImportError:
    PHONENUMBERS_AVAILABLE = False

try:
    from email_validator import validate_email, EmailNotValidError
    EMAIL_VALIDATOR_AVAILABLE = True
except ImportError:
    EMAIL_VALIDATOR_AVAILABLE = False

try:
    import pint
    ureg = pint.UnitRegistry()
    PINT_AVAILABLE = True
except ImportError:
    PINT_AVAILABLE = False

try:
    from langdetect import detect, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    from dateutil import parser as date_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False

try:
    import spacy
    try:
        nlp = spacy.load("fr_core_news_md")
        SPACY_LANG = "fr"
    except:
        try:
            nlp = spacy.load("en_core_web_md")
            SPACY_LANG = "en"
        except:
            nlp = None
            SPACY_LANG = None
    SPACY_AVAILABLE = nlp is not None
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CleanerError(Exception):
    """Exception de base pour le module Cleaner."""
    pass


class EncodingError(CleanerError):
    """Erreur de détection/conversion d'encodage."""
    pass


class FileFormatError(CleanerError):
    """Erreur de format de fichier non supporté."""
    pass


class DataValidationError(CleanerError):
    """Erreur de validation des données."""
    pass


class TableStructureError(CleanerError):
    """Erreur de structure de tableau."""
    pass


class Cleaner:
    """
    Classe de nettoyage modulable et intelligent pour CSV/DataFrames.
    
    Combine nettoyage classique et détection intelligente de +100 types de données.
    
    Usage simple:
        cleaner = Cleaner()
        df_clean = cleaner.clean(df)  # Tout nettoyer
        
    Usage avec extraction intelligente:
        cleaner = Cleaner()
        df_clean, metadata = cleaner.smart_clean(df, extract_columns=True)
    """
    
    # Patterns regex ultra-optimisés pour détection intelligente
    PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'url': r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)',
        'domain': r'(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}',
        'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        'mac_address': r'(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})',
        'phone_fr': r'(?:0|\+33)[1-9](?:\s?\d{2}){4}',
        'phone_us': r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        'phone_generic': r'[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}',
        'iban': r'[A-Z]{2}\d{2}[A-Z0-9]{1,30}',
        'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        'price_eur': r'(?:\d+[.,]\d{1,2})\s?€',
        'price_usd': r'\$\s?(?:\d+[.,]\d{1,2})',
        'currency': r'(?:€|\$|£|¥|₹|₽)',
        'siret': r'\b\d{14}\b',
        'siren': r'\b\d{9}\b',
        'nir': r'\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b',
        'postal_fr': r'\b\d{5}\b',
        'postal_us': r'\b\d{5}(?:-\d{4})?\b',
        'date_iso': r'\b\d{4}-\d{2}-\d{2}\b',
        'date_fr': r'\b\d{2}[/-]\d{2}[/-]\d{4}\b',
        'time': r'\b\d{1,2}:\d{2}(?::\d{2})?\b',
        'uuid': r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b',
        # unités (détection heuristique, la conversion réelle passe par convertUnit.extract_quantity)
        # NB: on n'inclut pas de symbole monétaire ici.
        'unit_weight': r'\b\d+(?:[\s\u00A0]?\d{3})*(?:[.,]\d+)?\s?(?:ug|µg|mg|g|kg|t|lb|lbs|oz|st)\b',
        'unit_distance': r'\b\d+(?:[\s\u00A0]?\d{3})*(?:[.,]\d+)?\s?(?:mm|cm|m|km|in|ft|yd|mi|nmi)\b',
        'unit_area': r'\b\d+(?:[\s\u00A0]?\d{3})*(?:[.,]\d+)?\s?(?:mm(?:\^?2|²)|cm(?:\^?2|²)|m(?:\^?2|²)|km(?:\^?2|²)|ha|acre|ft(?:\^?2|²)|yd(?:\^?2|²))\b',
        'unit_volume': r'\b\d+(?:[\s\u00A0]?\d{3})*(?:[.,]\d+)?\s?(?:ml|cl|dl|l|m(?:\^?3|³)|cm(?:\^?3|³)|ft(?:\^?3|³)|in(?:\^?3|³)|gal|qt|pt|floz)\b',
        'unit_speed': r'\b\d+(?:[\s\u00A0]?\d{3})*(?:[.,]\d+)?\s?(?:m/s|mps|km/h|kmh|kph|mph|kt|kts|knot|ft/s|fps)\b',
        'unit_pressure': r'\b\d+(?:[\s\u00A0]?\d{3})*(?:[.,]\d+)?\s?(?:pa|kpa|mpa|bar|mbar|hpa|psi|atm|mmhg|torr)\b',
        'unit_energy': r'\b\d+(?:[\s\u00A0]?\d{3})*(?:[.,]\d+)?\s?(?:j|kj|mj|wh|kwh|cal|kcal|btu)\b',
        'unit_power': r'\b\d+(?:[\s\u00A0]?\d{3})*(?:[.,]\d+)?\s?(?:w|kw|mw|hp|ch)\b',
        'percentage': r'\b\d+(?:[.,]\d+)?%',
        'coordinates': r'[-+]?\d{1,3}\.\d+,\s*[-+]?\d{1,3}\.\d+',
    }
    
    def __init__(
        self,
        *,
        # Activation des nettoyages (par défaut: tout activé sauf les destructifs)
        drop_missing: bool = False,
        remove_duplicates: bool = False,
        trim_strings: bool = True,
        normalize_spaces: bool = True,
        fix_typo: bool = True,
        fix_scientific: bool = True,
        lowercase_strings: bool = False,
        normalize_text: bool = False,  # Désactivé (perte d'info)
        clean_context: bool = False,   # Désactivé (avancé)
        remove_outliers: bool = False, # Désactivé (suppressions)
        fill_missing_values: bool = False, # Désactivé (altère données)
        auto_convert_types: bool = False, # Désactivé (peut changer sens)
        remove_empty: bool = True,

        # Conversion d'unités (opt-in, non destructif)
        convert_units: bool = False,
        unit_mode: Literal["add", "split", "replace"] = "add",
        unit_parse_threshold: float = 0.6,
        unit_target_by_category: Optional[Dict[str, str]] = None,
        unit_target_by_column: Optional[Dict[str, str]] = None,
        unit_value_suffix: str = "__value",
        unit_unit_suffix: str = "__unit",
        unit_converted_suffix: str = "__converted",

        # Données sensibles (opt-in)
        mask_sensitive_data: bool = False,
        sensitive_patterns: Optional[Dict[str, str]] = None,
        sensitive_replacement_template: str = "[SENSITIVE_{name}]",
        
        # Paramètres de drop_missing
        missing_threshold: float = 0.3,
        missing_axis: Literal["x", "y"] = "y",
        
        # Paramètres remove_duplicates
        duplicate_subset: Optional[List[str]] = None,
        duplicate_keep: str = "first",
        
        # Paramètres outliers
        outlier_columns: Optional[List[str]] = None,
        outlier_method: Literal["iqr", "zscore"] = "iqr",
        outlier_threshold: float = 1.5,
        
        # Paramètres fill_missing
        fill_strategy: Literal["mean", "median", "mode", "forward", "backward", "constant"] = "mean",
        fill_constant: Any = 0,
        fill_columns: Optional[List[str]] = None,
        
        # Colonnes à traiter (None = toutes)
        columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None
    ):
        """
        Initialise le Cleaner avec les options de nettoyage.
        
        Args:
            drop_missing: Supprime colonnes/lignes avec trop de valeurs manquantes
            remove_duplicates: Supprime les lignes dupliquées
            trim_strings: Supprime espaces début/fin
            normalize_spaces: Normalise les espaces et la ponctuation
            fix_typo: Corrige la typographie (guillemets, tirets, etc.)
            fix_scientific: Normalise les notations scientifiques
            normalize_text: Normalisation complète - ATTENTION: perte d'info
            clean_context: Nettoyage contextuel avancé (LaTeX, HTML)
            remove_outliers: Supprime les valeurs aberrantes
            fill_missing_values: Remplit les valeurs manquantes
            auto_convert_types: Convertit automatiquement les types
            remove_empty: Remplace chaînes vides par NaN
        """
        self.drop_missing_enabled = drop_missing
        self.remove_duplicates_enabled = remove_duplicates
        self.trim_strings_enabled = trim_strings
        self.normalize_spaces_enabled = normalize_spaces
        self.fix_typo_enabled = fix_typo
        self.fix_scientific_enabled = fix_scientific
        self.lowercase_strings_enabled = lowercase_strings
        self.normalize_text_enabled = normalize_text
        self.clean_context_enabled = clean_context
        self.remove_outliers_enabled = remove_outliers
        self.fill_missing_values_enabled = fill_missing_values
        self.auto_convert_types_enabled = auto_convert_types
        self.remove_empty_enabled = remove_empty

        self.convert_units_enabled = convert_units
        self.unit_mode = unit_mode
        self.unit_parse_threshold = unit_parse_threshold
        self.unit_target_by_category = unit_target_by_category
        self.unit_target_by_column = unit_target_by_column or {}
        self.unit_value_suffix = unit_value_suffix
        self.unit_unit_suffix = unit_unit_suffix
        self.unit_converted_suffix = unit_converted_suffix

        self.mask_sensitive_data_enabled = mask_sensitive_data
        self.sensitive_patterns = sensitive_patterns
        self.sensitive_replacement_template = sensitive_replacement_template
        
        self.missing_threshold = missing_threshold
        self.missing_axis = missing_axis
        
        self.duplicate_subset = duplicate_subset
        self.duplicate_keep = duplicate_keep
        
        self.outlier_columns = outlier_columns
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        
        self.fill_strategy = fill_strategy
        self.fill_constant = fill_constant
        self.fill_columns = fill_columns
        
        self.columns = columns
        self.exclude_columns = exclude_columns or []
        
        self._stats = {
            "rows_before": 0,
            "rows_after": 0,
            "cols_before": 0,
            "cols_after": 0,
            "operations": []
        }

        # Rapport de transformations (rempli à chaque appel à clean)
        self._report: Dict[str, Any] = {}

        # Compiler les patterns regex pour smart_clean
        self.compiled_patterns = {k: re.compile(v, re.IGNORECASE) for k, v in self.PATTERNS.items()}

        # Configuration des formats supportés
        self.supported_formats = {
            '.csv': self._load_csv,
            '.tsv': self._load_tsv,
            '.txt': self._load_txt,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            '.json': self._load_json,
            '.parquet': self._load_parquet,
            '.feather': self._load_feather,
            '.pickle': self._load_pickle,
            '.pkl': self._load_pickle
        }

        # Configuration encodages
        self.common_encodings = [
            'utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1',
            'ascii', 'utf-16', 'utf-32', 'cp850', 'cp437'
        ]


    def _unique_col_name(self, df: pd.DataFrame, base: str) -> str:
        if base not in df.columns:
            return base
        i = 1
        while f"{base}_{i}" in df.columns:
            i += 1
        return f"{base}_{i}"


    def mask_sensitive_data_in_dataframe(
        self,
        df: pd.DataFrame,
        *,
        columns: Optional[List[str]] = None,
        patterns: Optional[Dict[str, str]] = None,
        replacement_template: Optional[str] = None,
    ) -> pd.DataFrame:
        """Masque des données sensibles (email/téléphone/IBAN/CB/IP/UUID) dans les colonnes texte.

        Cette opération est opt-in car elle modifie le contenu (mais pas la structure).
        """
        if df is None or df.empty:
            return df

        patterns = patterns or self.sensitive_patterns
        replacement_template = replacement_template or self.sensitive_replacement_template

        # Patterns par défaut: réutilise au maximum les regex déjà présentes dans Cleaner.PATTERNS
        if not patterns:
            patterns = {
                # On masque d'abord les motifs très structurés (évite les faux positifs)
                "email": self.PATTERNS.get("email"),
                "credit_card": self.PATTERNS.get("credit_card"),
                "iban": self.PATTERNS.get("iban"),
                "ip": self.PATTERNS.get("ip_address"),
                "uuid": self.PATTERNS.get("uuid"),
                # Téléphones: privilégier les patterns spécifiques, le pattern générique est trop large
                # NB: on prend un pattern FR un peu plus permissif que PATTERNS['phone_fr'] pour accepter '+33 6 ...'
                "phone_fr": r"(?:\+33\s?|0)[1-9](?:[\s\.-]?\d{2}){4}",
                "phone_us": self.PATTERNS.get("phone_us"),
            }
            patterns = {k: v for k, v in patterns.items() if v}

        if not patterns:
            self._stats["operations"].append("mask_sensitive_data: skipped (no patterns)")
            return df

        cols = columns or self._get_target_columns(df)
        cols = [c for c in cols if c in df.columns and df[c].dtype == "object"]
        if not cols:
            return df

        report_entries: List[Dict[str, Any]] = []

        for col in cols:
            s = df[col]
            if s.dropna().empty:
                continue
            s_str = s.astype(str)

            col_entry: Dict[str, Any] = {"column": str(col), "masked": {}}

            for name, rx in patterns.items():
                try:
                    mask = s_str.str.contains(rx, flags=re.IGNORECASE, regex=True, na=False)
                    count = int(mask.sum())
                    if count <= 0:
                        continue
                    token = replacement_template.format(name=str(name).upper())
                    # Remplacement uniquement sur les lignes concernées (évite du travail inutile)
                    replaced = s_str.where(~mask, s_str.str.replace(rx, token, flags=re.IGNORECASE, regex=True))
                    s_str = replaced
                    col_entry["masked"][str(name)] = count
                except Exception:
                    continue

            # Ne réécrit la colonne que si on a masqué quelque chose
            if col_entry["masked"]:
                df[col] = s_str
                report_entries.append(col_entry)

        if report_entries:
            total_cells = sum(sum(v for v in e["masked"].values()) for e in report_entries)
            self._stats["operations"].append(f"mask_sensitive_data: {len(report_entries)} colonnes, {total_cells} cellules")
            if isinstance(getattr(self, "_report", None), dict):
                self._report["sensitive_masking"] = report_entries

        return df


    def convert_units_in_dataframe(
        self,
        df: pd.DataFrame,
        *,
        columns: Optional[List[str]] = None,
        mode: Optional[Literal["add", "split", "replace"]] = None,
        parse_threshold: Optional[float] = None,
        target_by_category: Optional[Dict[str, str]] = None,
        target_by_column: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Détecte des valeurs type "12 km", "3.2km/h" et ajoute des colonnes numériques + unité.

        Par défaut (mode=add) c'est non destructif: la colonne originale est conservée.
        """
        if df is None or df.empty:
            return df

        if convert_unit is None or extract_unit_quantity is None or get_unit_category is None:
            self._stats["operations"].append("convert_units: skipped (convertUnit unavailable)")
            return df

        mode = mode or self.unit_mode
        threshold = float(parse_threshold if parse_threshold is not None else self.unit_parse_threshold)
        target_by_category = target_by_category or self.unit_target_by_category
        target_by_column = target_by_column or self.unit_target_by_column

        if target_by_category is None:
            # Cibles par défaut (unités SI) — pas de monnaie
            target_by_category = {
                "length": "m",
                "weight": "kg",
                "time": "s",
                "temperature": "C",
                "volume": "m3",
                "area": "m2",
                "speed": "m/s",
                "pressure": "pa",
                "energy": "j",
                "power": "w",
            }

        cols = columns or self._get_target_columns(df)
        cols = [c for c in cols if c in df.columns and df[c].dtype == "object"]

        currency_symbols = ("€", "$", "£", "¥", "₹", "₽")

        def _safe_extract(x):
            if x is None:
                return None
            try:
                s = str(x)
            except Exception:
                return None
            s = s.strip()
            if not s:
                return None
            # Exclure la monnaie
            if any(sym in s for sym in currency_symbols):
                return None
            try:
                return extract_unit_quantity(s)
            except Exception:
                return None

        converted_cols: List[Dict[str, Any]] = []

        for col in cols:
            s = df[col]
            non_null = s.dropna()
            if non_null.empty:
                continue

            sample = non_null.astype(str).head(500)
            parsed_sample = sample.apply(_safe_extract)
            hit_rate = float(parsed_sample.notna().mean()) if len(parsed_sample) else 0.0
            if hit_rate < threshold:
                continue

            # parse complet
            parsed_all = s.astype(str).where(s.notna(), None).apply(_safe_extract)

            values = parsed_all.apply(lambda t: t[0] if isinstance(t, tuple) else np.nan)
            units = parsed_all.apply(lambda t: t[1] if isinstance(t, tuple) else None)

            # catégorie (si possible)
            first_unit = next((u for u in units.dropna().tolist() if u), None)
            category = None
            try:
                if first_unit:
                    category = get_unit_category(first_unit)
            except Exception:
                category = None

            # cible
            target_unit = target_by_column.get(col)
            if not target_unit and category:
                target_unit = target_by_category.get(category)

            value_col = self._unique_col_name(df, f"{col}{self.unit_value_suffix}")
            unit_col = self._unique_col_name(df, f"{col}{self.unit_unit_suffix}")
            df[value_col] = values
            df[unit_col] = units

            converted_col = None
            converted_count = 0

            if target_unit and category:
                # Conversion cellule par cellule (unités potentiellement mélangées)
                def _conv(row):
                    v = row[0]
                    u = row[1]
                    if v is None or (isinstance(v, float) and np.isnan(v)) or not u:
                        return np.nan
                    try:
                        return float(convert_unit(float(v), str(u), str(target_unit)))
                    except Exception:
                        return np.nan

                converted_values = pd.concat([values, units], axis=1).apply(_conv, axis=1)
                converted_count = int(converted_values.notna().sum())
                converted_col = self._unique_col_name(df, f"{col}{self.unit_converted_suffix}_{str(target_unit).replace('/', '_')}")
                df[converted_col] = converted_values

                if mode == "replace":
                    df[col] = df[converted_col]
                elif mode == "split":
                    df[col] = df[value_col]

            else:
                # Pas de cible: on ne convertit pas, mais on peut splitter/remplacer
                if mode == "replace" or mode == "split":
                    df[col] = df[value_col]

            info = {
                "column": str(col),
                "hit_rate": float(hit_rate),
                "parsed_count": int(values.notna().sum()),
                "category": category,
                "target_unit": target_unit,
                "value_col": value_col,
                "unit_col": unit_col,
                "converted_col": converted_col,
                "converted_count": int(converted_count),
                "mode": mode,
            }
            converted_cols.append(info)

        if converted_cols:
            self._stats["operations"].append(f"convert_units: {len(converted_cols)} colonnes détectées")
            if isinstance(getattr(self, "_report", None), dict):
                self._report["unit_conversions"] = converted_cols

        return df
    
    def detect_encoding(self, file_path: Union[str, Path], sample_size: int = 100000) -> str:
        """
        Détecte automatiquement l'encodage d'un fichier.
        
        Args:
            file_path: Chemin vers le fichier
            sample_size: Taille de l'échantillon pour la détection
            
        Returns:
            Encodage détecté
            
        Raises:
            EncodingError: Si aucun encodage n'est détecté
        """
        try:
            with open(file_path, 'rb') as f:
                sample = f.read(sample_size)
            
            # Utiliser chardet pour la détection
            detected = chardet.detect(sample)
            
            if detected and detected['confidence'] > 0.7:
                encoding = detected['encoding']
                logger.info(f"Encodage détecté: {encoding} (confiance: {detected['confidence']:.2f})")
                return encoding
            
            # Fallback: tester les encodages courants
            for encoding in self.common_encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        f.read(1000)  # Test de lecture
                    logger.info(f"Encodage validé par test: {encoding}")
                    return encoding
                except UnicodeDecodeError:
                    continue
            
            raise EncodingError(f"Impossible de détecter l'encodage du fichier {file_path}")
            
        except Exception as e:
            raise EncodingError(f"Erreur lors de la détection d'encodage: {e}")
    
    def _load_csv(self, file_path: Union[str, Path], encoding: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Charge un fichier CSV avec détection automatique des paramètres."""
        file_path = Path(file_path)
        
        # Détecter l'encodage si non fourni
        if encoding is None:
            encoding = self.detect_encoding(file_path)
        
        # Détecter le délimiteur et autres paramètres
        with open(file_path, 'r', encoding=encoding) as f:
            sample = f.read(8192)
            
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
            delimiter = dialect.delimiter
            quotechar = dialect.quotechar
        except csv.Error:
            # Fallback: tester les délimiteurs courants
            delimiters = [',', ';', '\t', '|']
            delimiter_counts = {d: sample.count(d) for d in delimiters}
            delimiter = max(delimiter_counts, key=delimiter_counts.get)
            quotechar = '"'
        
        # Paramètres par défaut pour CSV robuste
        default_params = {
            'encoding': encoding,
            'delimiter': delimiter,
            'quotechar': quotechar,
            'skipinitialspace': True,
            'na_values': ['', 'NULL', 'null', 'None', 'N/A', 'n/a', '#N/A', 'NaN', 'nan'],
            'keep_default_na': True,
            'dtype': str,  # Charger tout en string d'abord
            'low_memory': False
        }
        
        # Fusionner avec les paramètres utilisateur
        params = {**default_params, **kwargs}
        
        try:
            df = pd.read_csv(file_path, **params)
            logger.info(f"CSV chargé avec succès: {df.shape} lignes×colonnes, encodage: {encoding}")
            return df
        except Exception as e:
            # Fallback robuste: parsing permissif (utile pour CSV sales avec colonnes irrégulières)
            fallback_params = dict(params)
            fallback_params.setdefault('engine', 'python')
            if fallback_params.get('engine') == 'python':
                fallback_params.pop('low_memory', None)
            # pandas >= 1.3
            if 'on_bad_lines' not in fallback_params:
                fallback_params['on_bad_lines'] = 'skip'

            try:
                df = pd.read_csv(file_path, **fallback_params)
                logger.warning(
                    f"CSV chargé en mode fallback (lignes invalides ignorées): {df.shape} lignes×colonnes, encodage: {encoding}"
                )
                return df
            except TypeError:
                # pandas plus ancien: error_bad_lines / warn_bad_lines
                fallback_params.pop('on_bad_lines', None)
                fallback_params['error_bad_lines'] = False
                fallback_params['warn_bad_lines'] = True
                try:
                    df = pd.read_csv(file_path, **fallback_params)
                    logger.warning(
                        f"CSV chargé en mode fallback (error_bad_lines=False): {df.shape} lignes×colonnes, encodage: {encoding}"
                    )
                    return df
                except Exception as e2:
                    raise FileFormatError(f"Erreur lors du chargement CSV: {e} | fallback: {e2}")
            except Exception as e2:
                raise FileFormatError(f"Erreur lors du chargement CSV: {e} | fallback: {e2}")
    
    def _load_tsv(self, file_path: Union[str, Path], encoding: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Charge un fichier TSV."""
        kwargs['delimiter'] = '\t'
        return self._load_csv(file_path, encoding, **kwargs)
    
    def _load_txt(self, file_path: Union[str, Path], encoding: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Charge un fichier TXT délimité."""
        return self._load_csv(file_path, encoding, **kwargs)
    
    def _load_excel(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Charge un fichier Excel avec gestion d'erreurs."""
        try:
            # Paramètres par défaut pour Excel
            default_params = {
                'dtype': str,  # Charger en string d'abord
                'na_values': ['', 'NULL', 'null', 'None', 'N/A', 'n/a', '#N/A', 'NaN', 'nan']
            }
            
            params = {**default_params, **kwargs}
            df = pd.read_excel(file_path, **params)
            logger.info(f"Excel chargé avec succès: {df.shape} lignes×colonnes")
            return df
        except Exception as e:
            raise FileFormatError(f"Erreur lors du chargement Excel: {e}")
    
    def _load_json(self, file_path: Union[str, Path], encoding: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Charge un fichier JSON avec normalisation automatique."""
        if encoding is None:
            encoding = self.detect_encoding(file_path)
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                data = json.load(f)
            
            # Normaliser selon le type de données
            if isinstance(data, list):
                df = pd.json_normalize(data)
            elif isinstance(data, dict):
                if all(isinstance(v, (list, dict)) for v in data.values()):
                    # Structure complexe
                    df = pd.json_normalize(data, errors='ignore')
                else:
                    # Structure plate
                    df = pd.DataFrame([data])
            else:
                raise ValueError("Format JSON non supporté")
            
            logger.info(f"JSON chargé et normalisé: {df.shape} lignes×colonnes")
            return df
        except Exception as e:
            raise FileFormatError(f"Erreur lors du chargement JSON: {e}")
    
    def _load_parquet(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Charge un fichier Parquet."""
        try:
            df = pd.read_parquet(file_path, **kwargs)
            logger.info(f"Parquet chargé avec succès: {df.shape} lignes×colonnes")
            return df
        except Exception as e:
            raise FileFormatError(f"Erreur lors du chargement Parquet: {e}")
    
    def _load_feather(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Charge un fichier Feather."""
        try:
            df = pd.read_feather(file_path, **kwargs)
            logger.info(f"Feather chargé avec succès: {df.shape} lignes×colonnes")
            return df
        except Exception as e:
            raise FileFormatError(f"Erreur lors du chargement Feather: {e}")
    
    def _load_pickle(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Charge un fichier Pickle avec sécurité."""
        try:
            df = pd.read_pickle(file_path, **kwargs)
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Le fichier pickle ne contient pas un DataFrame")
            logger.info(f"Pickle chargé avec succès: {df.shape} lignes×colonnes")
            return df
        except Exception as e:
            raise FileFormatError(f"Erreur lors du chargement Pickle: {e}")
    
    def load_file(self, file_path: Union[str, Path], encoding: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Charge un fichier de données avec détection automatique du format.
        
        Args:
            file_path: Chemin vers le fichier
            encoding: Encodage (détecté automatiquement si None)
            **kwargs: Paramètres spécifiques au format
            
        Returns:
            DataFrame chargé et pré-validé
            
        Raises:
            FileFormatError: Format non supporté ou erreur de chargement
            EncodingError: Problème d'encodage
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix not in self.supported_formats:
            raise FileFormatError(f"Format non supporté: {suffix}. Formats supportés: {list(self.supported_formats.keys())}")
        
        loader = self.supported_formats[suffix]
        
        try:
            # Charger le fichier
            if suffix in ['.csv', '.tsv', '.txt', '.json']:
                df = loader(file_path, encoding=encoding, **kwargs)
            else:
                df = loader(file_path, **kwargs)
            
            # Validation de base
            self._validate_dataframe(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Échec du chargement de {file_path}: {e}")
            raise
    
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Valide la structure et qualité du DataFrame.
        
        Args:
            df: DataFrame à valider
            
        Raises:
            DataValidationError: Problème de validation détecté
        """
        if df is None:
            raise DataValidationError("DataFrame est None")
        
        if df.empty:
            raise DataValidationError("DataFrame est vide")
        
        if len(df.columns) == 0:
            raise DataValidationError("DataFrame n'a aucune colonne")
        
        # Vérifier les noms de colonnes
        duplicate_cols = df.columns[df.columns.duplicated()]
        if len(duplicate_cols) > 0:
            logger.warning(f"Colonnes dupliquées détectées: {duplicate_cols.tolist()}")
        
        # Vérifier les colonnes entièrement vides
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            logger.warning(f"Colonnes entièrement vides: {empty_cols}")
        
        # Vérifier la mémoire
        memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if memory_usage_mb > 1000:  # Plus de 1GB
            logger.warning(f"DataFrame volumineux: {memory_usage_mb:.1f}MB")
    
    def fix_column_names(self, df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        """
        Corrige les noms de colonnes problématiques.
        
        Args:
            df: DataFrame à corriger
            inplace: Modifier en place
            
        Returns:
            DataFrame avec colonnes corrigées
        """
        if not inplace:
            df = df.copy()
        
        # Sauvegarder les noms originaux
        original_columns = df.columns.tolist()
        
        # Nettoyer les noms de colonnes
        new_columns = []
        for col in df.columns:
            # Convertir en string et nettoyer
            col_str = str(col).strip()
            
            # Supprimer caractères problématiques
            col_clean = re.sub(r'[^\w\s-]', '_', col_str)
            col_clean = re.sub(r'\s+', '_', col_clean)
            col_clean = re.sub(r'_+', '_', col_clean)
            col_clean = col_clean.strip('_')
            
            # Éviter les noms vides
            if not col_clean or col_clean == '_':
                col_clean = f'column_{len(new_columns)}'
            
            # Éviter les doublons
            base_name = col_clean
            counter = 1
            while col_clean in new_columns:
                col_clean = f'{base_name}_{counter}'
                counter += 1
            
            new_columns.append(col_clean)
        
        df.columns = new_columns
        
        # Logger les changements significatifs
        changed = [(old, new) for old, new in zip(original_columns, new_columns) if old != new]
        if changed and isinstance(getattr(self, "_report", None), dict):
            renames = {str(old): str(new) for old, new in changed}
            existing = self._report.get("column_renames")
            if isinstance(existing, dict):
                existing.update(renames)
            else:
                self._report["column_renames"] = renames
        if changed:
            logger.info(f"Noms de colonnes modifiés: {len(changed)} changements")
            for old, new in changed[:5]:  # Afficher les 5 premiers
                logger.info(f"  '{old}' -> '{new}'")
            if len(changed) > 5:
                logger.info(f"  ... et {len(changed) - 5} autres")
        
        self._stats["operations"].append(f"fix_column_names: {len(changed)} colonnes renommées")
        return df
    
    def fix_table_structure(self, df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        """
        Corrige les problèmes de structure de tableau.
        
        Args:
            df: DataFrame à corriger
            inplace: Modifier en place
            
        Returns:
            DataFrame avec structure corrigée
        """
        if not inplace:
            df = df.copy()
        
        issues_fixed = []
        
        # 1. Supprimer les lignes entièrement vides
        empty_rows_before = int(df.isnull().all(axis=1).sum())
        if empty_rows_before > 0:
            df = df.dropna(how='all')
            issues_fixed.append(f"Supprimé {empty_rows_before} lignes vides")
            if isinstance(getattr(self, "_report", None), dict):
                self._report.setdefault("removals", {}).setdefault("rows", []).append({
                    "reason": "all_null_row",
                    "count": empty_rows_before,
                })
        
        # 2. Supprimer les colonnes entièrement vides (sauf si c'est tout)
        if len(df.columns) > 1:
            empty_cols_before = df.columns[df.isnull().all()].tolist()
            if empty_cols_before:
                df = df.drop(columns=empty_cols_before)
                issues_fixed.append(f"Supprimé {len(empty_cols_before)} colonnes vides")
                if isinstance(getattr(self, "_report", None), dict):
                    self._report.setdefault("removals", {}).setdefault("columns", []).append({
                        "reason": "all_null_column",
                        "count": int(len(empty_cols_before)),
                        "columns": [str(c) for c in empty_cols_before[:200]],
                        "columns_truncated": bool(len(empty_cols_before) > 200),
                    })
        
        # 3. Détecter et corriger les headers malformés
        if len(df) > 1:
            # Vérifier si la première ligne pourrait être un header
            first_row = df.iloc[0]
            # On évite un faux positif: ne faire ça que si les colonnes actuelles sont génériques
            # (ex: 0/1/2, Unnamed: x, column_x), ce qui arrive quand le fichier est sans header.
            try:
                col_names = list(df.columns)
                col_str = [str(c).strip().lower() for c in col_names]
                cols_are_generic = (
                    all(isinstance(c, int) for c in col_names)
                    or all(re.fullmatch(r"(unnamed:\s*\d+|column_\d+|\d+)", s or "") for s in col_str)
                )
            except Exception:
                cols_are_generic = False

            # Heuristique "header-like": valeurs non manquantes, uniques, et majoritairement textuelles
            try:
                fr_str = first_row.astype('string')
                fr_clean = fr_str.fillna('').str.strip()
                alpha_ratio = fr_clean.str.contains(r"[A-Za-zÀ-ÿ]", regex=True).mean()
            except Exception:
                alpha_ratio = 0.0

            is_likely_header = (
                cols_are_generic
                and first_row.notna().all()
                and len(first_row.unique()) == len(first_row)
                and float(alpha_ratio) >= 0.6
            )

            if is_likely_header and not df.columns.equals(first_row):
                logger.info("Première ligne détectée comme header manqué")
                df.columns = first_row.astype(str)
                df = df.drop(df.index[0]).reset_index(drop=True)
                issues_fixed.append("Corrigé header manqué")
                if isinstance(getattr(self, "_report", None), dict):
                    self._report.setdefault("removals", {}).setdefault("rows", []).append({
                        "reason": "promote_first_row_to_header",
                        "count": 1,
                    })
        
        # 4. Détecter les colonnes qui devraient être séparées
        for col in df.select_dtypes(include=['object']).columns:
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                # Chercher des patterns de séparation constants
                separators = [';', '|', ':', ' - ', ' / ']
                for sep in separators:
                    if sample.str.contains(f'\\{sep}', regex=True).mean() > 0.8:
                        # Plus de 80% des valeurs contiennent ce séparateur
                        parts_counts = sample.str.split(sep).str.len()
                        if parts_counts.mode().iloc[0] > 1:  # Mode > 1 (généralement 2+)
                            logger.info(f"Colonne '{col}' pourrait être séparée sur '{sep}'")
                            # Note: On ne sépare pas automatiquement, juste signaler
                            break
        
        # 5. Corriger les indices dupliqués
        if df.index.duplicated().any():
            df = df.reset_index(drop=True)
            issues_fixed.append("Indices dupliqués corrigés")
        
        if issues_fixed:
            logger.info(f"Structure de tableau corrigée: {', '.join(issues_fixed)}")
            self._stats["operations"].append(f"fix_table_structure: {len(issues_fixed)} corrections")
            if isinstance(getattr(self, "_report", None), dict):
                self._report.setdefault("structure_fixes", []).extend(issues_fixed)
        
        return df
    
    def normalize_encoding(self, df: pd.DataFrame, target_encoding: str = 'utf-8', inplace: bool = True) -> pd.DataFrame:
        """
        Normalise l'encodage des données textuelles.
        
        Args:
            df: DataFrame à normaliser
            target_encoding: Encodage cible
            inplace: Modifier en place
            
        Returns:
            DataFrame avec encodage normalisé
        """
        if not inplace:
            df = df.copy()
        
        text_columns = df.select_dtypes(include=['object']).columns
        issues_fixed = 0
        
        for col in text_columns:
            # Détecter et corriger les problèmes d'encodage courants
            original_values = df[col].dropna()
            if len(original_values) == 0:
                continue
            
            # Patterns de problèmes d'encodage courants
            encoding_issues = {
                # Accents mal encodés
                r'Ã[©èêëà]': lambda x: x.replace('Ã©', 'é').replace('Ã ', 'à').replace('Ãª', 'ê').replace('Ã«', 'ë').replace('Ã¨', 'è'),
                # Caractères spéciaux mal encodés
                r'â‚¬': lambda x: x.replace('â‚¬', '€'),
                r'â€™': lambda x: x.replace('â€™', "'"),
                r'â€œ|â€\x9d': lambda x: x.replace('â€œ', '"').replace('â€\x9d', '"'),
                # BOM et caractères invisibles
                r'^\ufeff': lambda x: x.replace('\ufeff', ''),
            }
            
            for pattern, fix_func in encoding_issues.items():
                mask = df[col].str.contains(pattern, regex=True, na=False)
                if mask.any():
                    df.loc[mask, col] = df.loc[mask, col].apply(fix_func)
                    issues_fixed += mask.sum()
        
        if issues_fixed > 0:
            logger.info(f"Problèmes d'encodage corrigés: {issues_fixed} cellules")
            self._stats["operations"].append(f"normalize_encoding: {issues_fixed} corrections")
        
        return df
    
    def detect_and_fix_data_types(self, df: pd.DataFrame, inplace: bool = True) -> Dict[str, str]:
        """
        Détecte et corrige automatiquement les types de données.
        
        Args:
            df: DataFrame à analyser
            inplace: Modifier en place
            
        Returns:
            Dictionnaire des conversions effectuées
        """
        if not inplace:
            df = df.copy()
        
        conversions = {}
        
        for col in df.columns:
            original_dtype = str(df[col].dtype)
            
            # Ignorer les colonnes déjà typées correctement
            if df[col].dtype != 'object':
                continue
            
            sample = df[col].dropna().head(1000)
            if len(sample) == 0:
                continue
            
            # Tentative de conversion numérique
            try:
                # Nettoyer d'abord
                cleaned = sample.astype(str).str.strip()
                cleaned = cleaned.str.replace(',', '.', regex=False)  # Virgule décimale
                cleaned = cleaned.str.replace(r'[^\d.+-]', '', regex=True)  # Garder seulement chiffres et signes
                
                numeric_converted = pd.to_numeric(cleaned, errors='coerce')
                success_rate = numeric_converted.notna().sum() / len(sample)
                
                if success_rate > 0.8:  # Plus de 80% de succès
                    # Appliquer à toute la colonne
                    df_cleaned = df[col].astype(str).str.strip()
                    df_cleaned = df_cleaned.str.replace(',', '.', regex=False)
                    df_cleaned = df_cleaned.str.replace(r'[^\d.+-]', '', regex=True)
                    df[col] = pd.to_numeric(df_cleaned, errors='coerce')
                    
                    # Optimiser le type numérique
                    if df[col].dtype == 'float64' and df[col].notna().all():
                        if (df[col] == df[col].astype(int)).all():
                            df[col] = df[col].astype('Int64')  # Nullable integer
                            conversions[col] = f'{original_dtype} -> Int64'
                        else:
                            conversions[col] = f'{original_dtype} -> float64'
                    else:
                        conversions[col] = f'{original_dtype} -> float64'
                    continue
            except:
                pass
            
            # Tentative de conversion datetime
            try:
                datetime_converted = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
                success_rate = datetime_converted.notna().sum() / len(sample)
                
                if success_rate > 0.8:
                    df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    conversions[col] = f'{original_dtype} -> datetime64'
                    continue
            except:
                pass
            
            # Tentative de conversion booléenne
            boolean_values = {'true', 'false', '1', '0', 'yes', 'no', 'oui', 'non', 'y', 'n'}
            lower_values = set(sample.astype(str).str.lower().str.strip())
            
            if lower_values.issubset(boolean_values | {'nan', 'none', ''}):
                bool_map = {
                    'true': True, '1': True, 'yes': True, 'oui': True, 'y': True,
                    'false': False, '0': False, 'no': False, 'non': False, 'n': False
                }
                df[col] = df[col].astype(str).str.lower().str.strip().map(bool_map)
                conversions[col] = f'{original_dtype} -> boolean'
                continue
            
            # Optimiser les strings (category si peu d'uniques)
            unique_ratio = df[col].nunique() / len(df[col].dropna())
            if unique_ratio < 0.5:  # Moins de 50% de valeurs uniques
                df[col] = df[col].astype('category')
                conversions[col] = f'{original_dtype} -> category'
        
        if conversions:
            logger.info(f"Types de données optimisés: {len(conversions)} colonnes")
            for col, conversion in list(conversions.items())[:5]:
                logger.info(f"  {col}: {conversion}")
            if len(conversions) > 5:
                logger.info(f"  ... et {len(conversions) - 5} autres")
            
            self._stats["operations"].append(f"detect_and_fix_data_types: {len(conversions)} conversions")

            if isinstance(getattr(self, "_report", None), dict):
                dtype_conv = self._report.get("dtype_conversions")
                if isinstance(dtype_conv, dict):
                    dtype_conv.update({str(k): str(v) for k, v in conversions.items()})
                else:
                    self._report["dtype_conversions"] = {str(k): str(v) for k, v in conversions.items()}
        
        return conversions

    def _get_target_columns(self, df: pd.DataFrame) -> List[str]:
        """Détermine les colonnes à traiter."""
        if self.columns:
            cols = [c for c in self.columns if c in df.columns]
        else:
            cols = df.columns.tolist()
        
        return [c for c in cols if c not in self.exclude_columns]
    
    def _apply_to_series(self, series: pd.Series, func) -> pd.Series:
        """Applique une fonction de nettoyage à une série."""
        def _safe_apply(x):
            try:
                if pd.isna(x):
                    return x
            except Exception:
                pass
            return func(x)

        return series.apply(_safe_apply)
    
    def remove_missing_values(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
        axis: Optional[Literal["x", "y"]] = None
    ) -> pd.DataFrame:
        """
        Supprime les lignes/colonnes avec trop de valeurs manquantes.
        
        Args:
            df: DataFrame à nettoyer
            threshold: Seuil (0-1), None = utilise la config
            axis: 'x' ou 'y', None = utilise la config
        
        Returns:
            DataFrame nettoyé
        """
        thresh = threshold if threshold is not None else self.missing_threshold
        ax = axis if axis is not None else self.missing_axis
        
        before_shape = df.shape
        df = drop_missing(df, threshold=thresh, axis=ax)
        after_shape = df.shape
        
        self._stats["operations"].append(
            f"drop_missing: {before_shape} -> {after_shape}"
        )

        if isinstance(getattr(self, "_report", None), dict):
            self._report["drop_missing"] = {
                "threshold": float(thresh),
                "axis": str(ax),
                "shape_before": [int(before_shape[0]), int(before_shape[1])],
                "shape_after": [int(after_shape[0]), int(after_shape[1])],
                "rows_removed": int(before_shape[0] - after_shape[0]),
                "cols_removed": int(before_shape[1] - after_shape[1]),
            }
        
        return df
    
    def normalize_spaces(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise les espaces et la ponctuation.
        - Supprime espaces multiples
        - Corrige ponctuation
        - Recolle les séparateurs de milliers
        """
        cols = self._get_target_columns(df)
        
        for col in cols:
            if pd.api.types.is_string_dtype(df[col].dtype):
                df[col] = self._apply_to_series(df[col], normalize_space)
        
        self._stats["operations"].append("normalize_spaces")
        return df
    
    def fix_typography(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Corrige la typographie.
        - Guillemets " " → "
        - Apostrophes ' → '
        - Tirets – — → -
        - Points de suspension … → ...
        """
        cols = self._get_target_columns(df)
        
        for col in cols:
            if pd.api.types.is_string_dtype(df[col].dtype):
                df[col] = self._apply_to_series(df[col], normalize_typography)
        
        self._stats["operations"].append("fix_typography")
        return df

    def lowercase_all_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Met en minuscule toutes les colonnes texte (object/string).

        Objectif: standardiser les mots sans passer par normalize_text_full
        (qui enlève accents/emojis).
        """
        cols = self._get_target_columns(df)
        for col in cols:
            try:
                if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col].dtype):
                    df[col] = df[col].astype('string').str.lower()
            except Exception:
                continue
        self._stats["operations"].append("lowercase_strings")
        return df
    
    def fix_scientific_notation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise les notations scientifiques.
        - 1e7 → 10000000
        - 1×10^7 → 10000000
        - 3,2e-4 → 0.00032
        """
        cols = self._get_target_columns(df)
        
        for col in cols:
            if pd.api.types.is_string_dtype(df[col].dtype):
                df[col] = self._apply_to_series(df[col], normalize_scientific_notation)
        
        self._stats["operations"].append("fix_scientific")
        return df
    
    def normalize_text_full(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalisation complète du texte.
        ATTENTION: Perte d'information (minuscules, accents, emojis)
        
        - Supprime emojis
        - Enlève accents
        - Minuscules
        - Supprime caractères de contrôle
        """
        cols = self._get_target_columns(df)
        
        for col in cols:
            if pd.api.types.is_string_dtype(df[col].dtype):
                df[col] = self._apply_to_series(df[col], normalize_text_full)
        
        self._stats["operations"].append("normalize_text_full")
        return df
    
    def clean_context(self, df: pd.DataFrame, preserve_latex: bool = True) -> pd.DataFrame:
        """
        Nettoyage contextuel avancé.
        - Préserve formules LaTeX
        - Supprime balises HTML
        - Nettoie URLs/emails
        """
        cols = self._get_target_columns(df)
        
        for col in cols:
            if pd.api.types.is_string_dtype(df[col].dtype):
                df[col] = self._apply_to_series(
                    df[col],
                    lambda x: normalize_context(x, preserve_latex=preserve_latex)
                )
        
        self._stats["operations"].append("clean_context")
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Supprime les lignes dupliquées.
        """
        before_len = len(df)
        df = remove_duplicates(df, subset=self.duplicate_subset, keep=self.duplicate_keep)
        after_len = len(df)
        
        self._stats["operations"].append(f"remove_duplicates: {before_len} → {after_len} lignes")

        if isinstance(getattr(self, "_report", None), dict):
            self._report["deduplication"] = {
                "rows_before": int(before_len),
                "rows_after": int(after_len),
                "rows_removed": int(before_len - after_len),
                "keep": str(self.duplicate_keep),
                "subset": [str(c) for c in (self.duplicate_subset or [])],
            }
        return df
    
    def trim_all_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Supprime les espaces en début/fin de toutes les chaînes.
        """
        df = trim_strings(df, columns=self._get_target_columns(df))
        self._stats["operations"].append("trim_strings")
        return df
    
    def remove_empty_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remplace les chaînes vides par NaN.
        """
        df = remove_empty_strings(df, replace_with_nan=True)
        self._stats["operations"].append("remove_empty_strings")
        return df
    
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Supprime les lignes avec valeurs aberrantes.
        """
        cols = self.outlier_columns or [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ]
        
        before_len = len(df)
        df = remove_outliers(
            df, 
            columns=cols,
            method=self.outlier_method,
            threshold=self.outlier_threshold
        )
        after_len = len(df)
        
        self._stats["operations"].append(f"remove_outliers: {before_len} → {after_len} lignes")

        if isinstance(getattr(self, "_report", None), dict):
            self._report["outliers"] = {
                "rows_before": int(before_len),
                "rows_after": int(after_len),
                "rows_removed": int(before_len - after_len),
                "method": str(self.outlier_method),
                "threshold": float(self.outlier_threshold),
                "columns": [str(c) for c in cols],
            }
        return df
    
    def fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remplit les valeurs manquantes selon la stratégie définie.
        """
        missing_before = int(df.isna().sum().sum())
        df = fill_missing(
            df,
            strategy=self.fill_strategy,
            constant_value=self.fill_constant,
            columns=self.fill_columns
        )
        missing_after = int(df.isna().sum().sum())
        self._stats["operations"].append(f"fill_missing: strategy={self.fill_strategy}")

        if isinstance(getattr(self, "_report", None), dict):
            self._report["fill_missing"] = {
                "strategy": str(self.fill_strategy),
                "constant": self.fill_constant,
                "columns": [str(c) for c in (self.fill_columns or [])],
                "missing_before": int(missing_before),
                "missing_after": int(missing_after),
                "missing_filled": int(max(0, missing_before - missing_after)),
            }
        return df
    
    def convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convertit automatiquement les types de colonnes.
        """
        df = infer_and_convert_types(df, aggressive=False)
        self._stats["operations"].append("convert_types")
        return df
    
    def compute_statistics(self, df: pd.DataFrame, columns: List[str] = None) -> Dict[str, Any]:
        """
        Calcule des statistiques sur les colonnes numériques.
        
        Returns:
            Dictionnaire avec mean, median, mode pour chaque colonne
        """
        cols = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        stats = {}
        
        for col in cols:
            if col not in df.columns:
                continue
            
            # Tenter de convertir en numérique si ce n'est pas déjà le cas
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    data_series = pd.to_numeric(df[col], errors='coerce').dropna()
                except:
                    continue
            else:
                data_series = df[col].dropna()
            
            data = data_series.tolist()
            if len(data) == 0:
                continue
            
            try:
                stats[col] = {
                    "mean_arithmetic": mean(data, kind="arithmetic"),
                    "mean_geometric": mean(data, kind="geometric") if all(v > 0 for v in data) else None,
                    "median": central_tendency(data, kind="median"),
                    "mode": central_tendency(data, kind="mode")
                }
            except Exception as e:
                stats[col] = {"error": str(e)}
        
        return stats
    
    def detect_sequence_anomalies(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Détecte les anomalies dans une séquence (ex: EMP00001, EMP00002, ..., EMP01002).
        
        Args:
            df: DataFrame
            column: Nom de la colonne contenant la séquence
        
        Returns:
            Dictionnaire avec le statut et les détails de l'anomalie détectée
        """
        if column not in df.columns:
            return {"status": "error", "message": f"Colonne '{column}' introuvable"}
        
        # Extraire la séquence en ignorant les NaN
        sequence = df[column].dropna().tolist()
        
        if len(sequence) == 0:
            return {"status": "error", "message": "Séquence vide"}
        
        # Utiliser logic.py pour détecter les ruptures
        result = detect_sequence_logic_break(sequence, index_base=0, min_learn=3)
        
        return result
    
    def validate_sequences(self, df: pd.DataFrame, columns: List[str] = None) -> Dict[str, Any]:
        """
        Valide toutes les colonnes qui semblent être des séquences.
        
        Args:
            df: DataFrame
            columns: Liste des colonnes à valider (None = auto-détection)
        
        Returns:
            Dictionnaire {colonne: résultat_détection}
        """
        if columns is None:
            # Auto-détection : colonnes avec patterns répétitifs
            columns = []
            for col in df.columns:
                # Vérifier si la colonne contient des patterns de type ID
                sample = df[col].dropna().head(10).astype(str)
                if any(sample.str.match(r'^[A-Z]+\d+$')):
                    columns.append(col)
        
        results = {}
        for col in columns:
            results[col] = self.detect_sequence_anomalies(df, col)
        
        return results
    
    def fix_sequence(self, df: pd.DataFrame, column: str, auto_fix: bool = False) -> pd.DataFrame:
        """
        Corrige les anomalies de séquence dans une colonne.
        
        Args:
            df: DataFrame
            column: Colonne à corriger
            auto_fix: Si True, corrige automatiquement, sinon retourne le rapport
        
        Returns:
            DataFrame corrigé si auto_fix=True, sinon DataFrame original
        """
        result = self.detect_sequence_anomalies(df, column)
        
        if 'break_at' not in result:
            return df
        
        if not auto_fix:
            print(f"⚠️  Anomalie détectée dans '{column}' à l'index {result['break_at']}")
            print(f"   Attendu: {result['expected']}, Observé: {result['observed']}")
            print(f"   Utilisez auto_fix=True pour corriger automatiquement")
            return df
        
        # Correction automatique
        df = df.copy()
        sequence = df[column].tolist()
        
        # Extraire le pattern (préfixe + format nombre)
        import re
        sample = str(sequence[0])
        match = re.match(r'^([A-Z]+)(\d+)$', sample)
        
        if not match:
            print(f"❌ Impossible de déterminer le pattern pour '{column}'")
            return df
        
        prefix = match.group(1)
        num_width = len(match.group(2))
        
        # Reconstruire la séquence correcte
        for i in range(len(sequence)):
            if pd.notna(sequence[i]):
                sequence[i] = f"{prefix}{str(i+1).zfill(num_width)}"
        
        df[column] = sequence
        
        print(f"✅ Séquence '{column}' corrigée automatiquement")
        print(f"   Pattern: {prefix}{{nombre sur {num_width} chiffres}}")
        
        return df
    
    def detect_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Détecte les lignes dupliquées (complètes et partielles).
        """
        duplicates_info = {
            "complete_duplicates": [],
            "partial_duplicates": {}
        }
        
        # Doublons complets
        dup_mask = df.duplicated(keep=False)
        if dup_mask.any():
            dup_rows = df[dup_mask].index.tolist()
            duplicates_info["complete_duplicates"] = dup_rows
        
        # Doublons partiels sur colonnes clés
        key_columns = ['email', 'employee_id', 'phone'] if all(c in df.columns for c in ['email', 'employee_id', 'phone']) else []
        
        for col in key_columns:
            if col in df.columns:
                dup_mask = df[col].duplicated(keep=False)
                if dup_mask.any():
                    dup_rows = df[dup_mask].index.tolist()
                    dup_values = df.loc[dup_rows, col].value_counts()
                    duplicates_info["partial_duplicates"][col] = {
                        "rows": dup_rows,
                        "values": dup_values.to_dict()
                    }
        
        return duplicates_info
    
    def detect_invalid_formats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Détecte les valeurs avec formats invalides (email, téléphone).
        """
        import re
        
        format_issues = {}
        
        # Validation email
        if 'email' in df.columns:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            invalid_emails = []
            
            for idx, email in df['email'].items():
                if pd.notna(email) and not re.match(email_pattern, str(email)):
                    invalid_emails.append({"ligne": idx, "valeur": str(email)})
            
            if invalid_emails:
                format_issues['email'] = invalid_emails
        
        # Validation téléphone (format français)
        if 'phone' in df.columns:
            phone_pattern = r'^(\+33|0)[1-9](\d{2}){4}$'
            invalid_phones = []
            
            for idx, phone in df['phone'].items():
                if pd.notna(phone):
                    phone_clean = str(phone).replace(' ', '').replace('.', '').replace('-', '')
                    if not re.match(phone_pattern, phone_clean):
                        invalid_phones.append({"ligne": idx, "valeur": str(phone)})
            
            if invalid_phones:
                format_issues['phone'] = invalid_phones
        
        return format_issues
    
    def detect_impossible_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Détecte les valeurs impossibles ou hors limites réalistes.
        """
        impossible = {}
        
        # Âge impossible
        if 'age' in df.columns:
            invalid_age = []
            for idx, age in df['age'].items():
                if pd.notna(age):
                    try:
                        age_val = float(age)
                        if age_val < 0 or age_val > 120:
                            invalid_age.append({"ligne": idx, "valeur": age_val})
                    except:
                        pass
            
            if invalid_age:
                impossible['age'] = invalid_age
        
        # Salaire impossible
        if 'salary' in df.columns:
            invalid_salary = []
            for idx, sal in df['salary'].items():
                if pd.notna(sal):
                    try:
                        sal_val = float(sal)
                        if sal_val < 0 or sal_val > 10000000:  # 10M limite
                            invalid_salary.append({"ligne": idx, "valeur": sal_val})
                    except:
                        pass
            
            if invalid_salary:
                impossible['salary'] = invalid_salary
        
        return impossible
    
    def compute_advanced_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calcule des statistiques avancées (skewness, kurtosis, distribution).
        """
        from scipy import stats as scipy_stats
        
        advanced_stats = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].notna().sum() > 3:  # Minimum 3 valeurs
                data = df[col].dropna()
                
                advanced_stats[col] = {
                    "skewness": float(scipy_stats.skew(data)),
                    "kurtosis": float(scipy_stats.kurtosis(data)),
                    "coefficient_variation": float(data.std() / data.mean()) if data.mean() != 0 else 0,
                    "distribution_type": "normale" if abs(scipy_stats.skew(data)) < 0.5 else "asymétrique"
                }
        
        return advanced_stats
    
    def detect_business_inconsistencies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Détecte les incohérences métier (dates, âges, etc.).
        """
        inconsistencies = []
        
        # Vérifier cohérence âge/date d'embauche
        if 'age' in df.columns and 'hire_date' in df.columns:
            for idx, row in df.iterrows():
                if pd.notna(row['age']) and pd.notna(row['hire_date']):
                    try:
                        age = int(row['age'])
                        # Si âge < 16 au moment de l'embauche (incohérent)
                        if age < 16:
                            inconsistencies.append({
                                "ligne": idx,
                                "type": "age_embauche",
                                "description": f"Âge {age} trop jeune pour être embauché"
                            })
                    except:
                        pass
        
        # Vérifier cohérence salaire/poste si colonnes existent
        if 'salary' in df.columns and 'department' in df.columns:
            # Convertir salary en numeric si nécessaire
            try:
                salary_numeric = pd.to_numeric(df['salary'], errors='coerce')
                df_temp = df.copy()
                df_temp['salary_num'] = salary_numeric
                
                # Calculer médiane par département
                salary_by_dept = df_temp.groupby('department')['salary_num'].median()
                
                for idx, row in df.iterrows():
                    if pd.notna(row['salary']) and pd.notna(row['department']):
                        dept = row['department']
                        if dept in salary_by_dept:
                            try:
                                sal_val = float(row['salary'])
                                median_sal = salary_by_dept[dept]
                                if pd.notna(median_sal) and sal_val > median_sal * 3:  # 3x la médiane
                                    inconsistencies.append({
                                        "ligne": idx,
                                        "type": "salaire_excessif",
                                        "description": f"Salaire {sal_val:.0f} >> médiane département ({median_sal:.0f})"
                                    })
                            except:
                                pass
            except Exception as e:
                pass  # Ignorer les erreurs de conversion
        
        return {"inconsistencies": inconsistencies}
    
    def generate_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Génère un rapport complet des anomalies détectées dans le DataFrame.
        
        Args:
            df: DataFrame à analyser
        
        Returns:
            Dictionnaire avec toutes les anomalies détectées par type
        """
        report = {
            "total_rows": len(df),
            "total_cols": len(df.columns),
            "missing_values": {},
            "outliers": {},
            "sequence_breaks": {},
            "domain_anomalies": {},
            "duplicates": {},
            "invalid_formats": {},
            "impossible_values": {},
            "advanced_statistics": {},
            "business_inconsistencies": {},
            "summary": []
        }
        
        # 1. Valeurs manquantes par colonne
        missing = df.isnull()
        for col in df.columns:
            missing_rows = missing[missing[col]].index.tolist()
            if missing_rows:
                report["missing_values"][col] = missing_rows
                report["summary"].append(
                    f"⚠️  Colonne '{col}': {len(missing_rows)} valeurs manquantes aux lignes {missing_rows[:5]}{'...' if len(missing_rows) > 5 else ''}"
                )
        
        # 2. Outliers dans les colonnes numériques
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        
        # Ajouter aussi les colonnes convertibles en numérique
        for col in df.columns:
            if col not in numeric_cols and df[col].dtype == 'object':
                try:
                    test_convert = pd.to_numeric(df[col], errors='coerce')
                    if test_convert.notna().sum() > len(df) * 0.5:  # Au moins 50% convertibles
                        numeric_cols.append(col)
                except:
                    pass
        
        for col in numeric_cols:
            # Convertir en numérique si nécessaire
            if not pd.api.types.is_numeric_dtype(df[col]):
                col_data = pd.to_numeric(df[col], errors='coerce').dropna()
            else:
                col_data = df[col].dropna()
            
            if len(col_data) > 0:
                try:
                    # Détecter outliers avec IQR
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    
                    # Créer le masque d'outliers
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        col_numeric = pd.to_numeric(df[col], errors='coerce')
                        outlier_mask = (col_numeric < lower) | (col_numeric > upper)
                    else:
                        outlier_mask = (df[col] < lower) | (df[col] > upper)
                    
                    outlier_indices = df[outlier_mask].index.tolist()
                    
                    if outlier_indices:
                        if not pd.api.types.is_numeric_dtype(df[col]):
                            outlier_values = pd.to_numeric(df[col], errors='coerce').loc[outlier_indices].tolist()
                        else:
                            outlier_values = df.loc[outlier_indices, col].tolist()
                        
                        report["outliers"][col] = [
                            {"ligne": idx, "valeur": val, "min_attendu": lower, "max_attendu": upper}
                            for idx, val in zip(outlier_indices, outlier_values) if pd.notna(val)
                        ]
                        
                        if report["outliers"][col]:
                            report["summary"].append(
                                f"📊 Colonne '{col}': {len(report['outliers'][col])} outliers détectés (lignes {[x['ligne'] for x in report['outliers'][col][:5]]}{'...' if len(report['outliers'][col]) > 5 else ''})"
                            )
                except:
                    pass
        
        # 3. Ruptures de séquence
        sequence_cols = []
        for col in df.columns:
            sample = df[col].dropna().head(10).astype(str)
            if any(sample.str.match(r'^[A-Z]+\d+$')):
                sequence_cols.append(col)
        
        for col in sequence_cols:
            result = self.detect_sequence_anomalies(df, col)
            if 'break_at' in result:
                report["sequence_breaks"][col] = {
                    "ligne": result['break_at'],
                    "attendu": result['expected'],
                    "observe": result['observed'],
                    "regle": result['rule']
                }
                report["summary"].append(
                    f"🔢 Colonne '{col}': Rupture de séquence à la ligne {result['break_at']} (attendu: {result['expected']}, observé: {result['observed']})"
                )
        
        # 4. Anomalies de domaine (emails, etc.)
        for col in df.columns:
            if 'email' in col.lower() or 'mail' in col.lower():
                email_series = df[col].dropna().astype(str)
                if len(email_series) > 0:
                    # Extraire les domaines
                    domains = email_series.str.extract(r'@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$')[0]
                    domain_counts = domains.value_counts()
                    
                    if len(domain_counts) > 0:
                        # Le domaine principal (le plus fréquent)
                        main_domain = domain_counts.index[0]
                        main_count = domain_counts.iloc[0]
                        
                        # Trouver les domaines minoritaires (< 5% du total)
                        total_valid_emails = domains.notna().sum()
                        threshold = total_valid_emails * 0.05
                        
                        anomalous_domains = domain_counts[domain_counts < threshold].index.tolist()
                        
                        if len(anomalous_domains) > 0:
                            anomaly_rows = []
                            for domain in anomalous_domains:
                                # Trouver les indices où le domaine correspond
                                mask = domains == domain
                                rows_with_domain = domains[mask].index.tolist()
                                
                                for row in rows_with_domain:
                                    anomaly_rows.append({
                                        "ligne": row,
                                        "email": df.loc[row, col],
                                        "domaine": domain,
                                        "domaine_attendu": main_domain
                                    })
                            
                            if anomaly_rows:
                                report["domain_anomalies"][col] = anomaly_rows
                                report["summary"].append(
                                    f"📧 Colonne '{col}': {len(anomaly_rows)} emails avec domaine inhabituel (lignes {[x['ligne'] for x in anomaly_rows[:5]]}{'...' if len(anomaly_rows) > 5 else ''})"
                                )
        
        # 5. Doublons
        duplicates_result = self.detect_duplicates(df)
        report["duplicates"] = duplicates_result
        
        if duplicates_result["complete_duplicates"]:
            report["summary"].append(
                f"🔄 {len(duplicates_result['complete_duplicates'])} lignes dupliquées complètes"
            )
        
        for col, dup_info in duplicates_result["partial_duplicates"].items():
            report["summary"].append(
                f"🔄 Colonne '{col}': {len(dup_info['rows'])} doublons détectés"
            )
        
        # 6. Formats invalides
        format_issues = self.detect_invalid_formats(df)
        report["invalid_formats"] = format_issues
        
        for col, issues in format_issues.items():
            report["summary"].append(
                f"📝 Colonne '{col}': {len(issues)} formats invalides"
            )
        
        # 7. Valeurs impossibles
        impossible = self.detect_impossible_values(df)
        report["impossible_values"] = impossible
        
        for col, issues in impossible.items():
            report["summary"].append(
                f"⛔ Colonne '{col}': {len(issues)} valeurs impossibles"
            )
        
        # 8. Statistiques avancées
        try:
            advanced_stats = self.compute_advanced_statistics(df)
            report["advanced_statistics"] = advanced_stats
        except Exception as e:
            report["advanced_statistics"] = {"error": str(e)}
        
        # 9. Incohérences métier
        business_incons = self.detect_business_inconsistencies(df)
        report["business_inconsistencies"] = business_incons
        
        if business_incons.get("inconsistencies"):
            report["summary"].append(
                f"💼 {len(business_incons['inconsistencies'])} incohérences métier détectées"
            )
        
        return report
    
    def print_report(self, df: pd.DataFrame):
        """
        Génère et affiche un rapport détaillé des anomalies.
        """
        report = self.generate_report(df)
        
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 25 + "RAPPORT D'ANOMALIES" + " " * 34 + "║")
        print("╚" + "═" * 78 + "╝\n")
        
        print(f"📋 Dataset : {report['total_rows']} lignes × {report['total_cols']} colonnes\n")
        
        # Résumé
        if report["summary"]:
            print("=" * 80)
            print(" RÉSUMÉ DES ANOMALIES DÉTECTÉES")
            print("=" * 80 + "\n")
            
            for msg in report["summary"]:
                print(f"  {msg}")
            print()
        else:
            print("✅ Aucune anomalie détectée !\n")
            return
        
        # Détails valeurs manquantes
        if report["missing_values"]:
            print("=" * 80)
            print(" DÉTAILS : VALEURS MANQUANTES")
            print("=" * 80 + "\n")
            
            for col, rows in report["missing_values"].items():
                print(f"Colonne '{col}' :")
                print(f"  • {len(rows)} valeurs manquantes")
                print(f"  • Lignes concernées : {rows[:10]}{'...' if len(rows) > 10 else ''}")
                print(f"\n  ❓ Que faire ?")
                print(f"     - Supprimer les lignes : cleaner.drop_missing = True")
                print(f"     - Remplir avec moyenne : cleaner.fill_missing_values = True (strategy='mean')")
                print(f"     - Remplir avec médiane : strategy='median'")
                print(f"     - Laisser tel quel : Aucune action\n")
        
        # Détails outliers
        if report["outliers"]:
            print("=" * 80)
            print(" DÉTAILS : OUTLIERS (VALEURS ABERRANTES)")
            print("=" * 80 + "\n")
            
            for col, outliers in report["outliers"].items():
                print(f"Colonne '{col}' :")
                print(f"  • {len(outliers)} outliers détectés")
                for item in outliers[:5]:
                    print(f"    - Ligne {item['ligne']}: valeur={item['valeur']:.2f} (attendu: {item['min_attendu']:.2f} - {item['max_attendu']:.2f})")
                if len(outliers) > 5:
                    print(f"    ... et {len(outliers)-5} autres")
                
                print(f"\n  ❓ Que faire ?")
                print(f"     - Supprimer ces lignes : cleaner.remove_outliers = True")
                print(f"     - Remplacer par médiane : fill_missing après conversion en NaN")
                print(f"     - Vérifier manuellement : Inspecter les lignes")
                print(f"     - Laisser tel quel : Aucune action\n")
        
        # Détails ruptures de séquence
        if report["sequence_breaks"]:
            print("=" * 80)
            print(" DÉTAILS : RUPTURES DE SÉQUENCE")
            print("=" * 80 + "\n")
            
            for col, brk in report["sequence_breaks"].items():
                print(f"Colonne '{col}' :")
                print(f"  • Rupture détectée à la ligne {brk['ligne']}")
                print(f"  • Règle détectée : {brk['regle']}")
                print(f"  • Valeur attendue : {brk['attendu']}")
                print(f"  • Valeur observée : {brk['observe']}")
                
                print(f"\n  ❓ Que faire ?")
                print(f"     - Corriger automatiquement : cleaner.fix_sequence(df, '{col}', auto_fix=True)")
                print(f"     - Corriger manuellement : df.loc[{brk['ligne']}, '{col}'] = '{brk['attendu']}'")
                print(f"     - Ignorer : Laisser tel quel\n")
        
        # Détails anomalies de domaine
        if report["domain_anomalies"]:
            print("=" * 80)
            print(" DÉTAILS : ANOMALIES DE DOMAINE EMAIL")
            print("=" * 80 + "\n")
            
            for col, anomalies in report["domain_anomalies"].items():
                print(f"Colonne '{col}' :")
                print(f"  • {len(anomalies)} emails avec domaine inhabituel")
                for item in anomalies[:5]:
                    print(f"    - Ligne {item['ligne']}: {item['email']}")
                    print(f"      Domaine: {item['domaine']} (attendu: {item['domaine_attendu']})")
                if len(anomalies) > 5:
                    print(f"    ... et {len(anomalies)-5} autres")
                
                print(f"\n  ❓ Que faire ?")
                print(f"     - Corriger manuellement : Vérifier et corriger les domaines")
                print(f"     - Filtrer : Exclure les domaines non autorisés")
                print(f"     - Laisser tel quel : Si variation normale\n")
        
        print("=" * 80)
        print(" ACTIONS DISPONIBLES")
        print("=" * 80 + "\n")
        
        print("# Pour appliquer des corrections :")
        print("cleaner = Cleaner(")
        print("    drop_missing=True,          # Supprimer lignes avec NaN")
        print("    fill_missing_values=True,   # Remplir NaN avec moyenne/médiane")
        print("    remove_outliers=True        # Supprimer outliers")
        print(")")
        print("df_clean = cleaner.clean(df)")
        print()
        print("# Pour corriger les séquences :")
        print("df_fixed = cleaner.fix_sequence(df, 'employee_id', auto_fix=True)")
        print()
        
        print("=" * 80)
    
    def clean(self, 
              df: Optional[pd.DataFrame] = None, 
              file_path: Optional[Union[str, Path]] = None,
              encoding: Optional[str] = None,
              validate_structure: bool = True,
              fix_encoding: bool = True,
              auto_detect_types: bool = True,
              **load_kwargs) -> pd.DataFrame:
        """
        Nettoie un DataFrame ou charge et nettoie un fichier.
        
        Args:
            df: DataFrame à nettoyer (optionnel si file_path fourni)
            file_path: Chemin vers fichier à charger (optionnel si df fourni)
            encoding: Encodage pour chargement fichier
            validate_structure: Valider et corriger la structure
            fix_encoding: Corriger les problèmes d'encodage
            auto_detect_types: Détecter et optimiser les types
            **load_kwargs: Arguments pour le chargement de fichier
        
        Returns:
            DataFrame nettoyé
            
        Raises:
            ValueError: Si ni df ni file_path n'est fourni
            CleanerError: Erreur lors du nettoyage
        """
        try:
            # Charger les données si nécessaire
            if df is None:
                if file_path is None:
                    raise ValueError("Soit 'df' soit 'file_path' doit être fourni")
                logger.info(f"Chargement du fichier: {file_path}")
                df = self.load_file(file_path, encoding=encoding, **load_kwargs)
            else:
                df = df.copy()  # Ne pas modifier l'original
            
            # Validation initiale
            self._validate_dataframe(df)
            
            # Statistiques initiales
            self._stats = {
                "rows_before": len(df),
                "cols_before": len(df.columns),
                "operations": [],
                "memory_before_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
                "dtypes_before": dict(df.dtypes.astype(str)),
                "started_at": datetime.now().isoformat()
            }

            # Rapport de transformations (JSON-serializable)
            self._report = {
                "generated_at": datetime.now().isoformat(),
                "config": {
                    "validate_structure": bool(validate_structure),
                    "fix_encoding": bool(fix_encoding),
                    "auto_detect_types": bool(auto_detect_types),
                    "drop_missing": bool(self.drop_missing_enabled),
                    "remove_duplicates": bool(self.remove_duplicates_enabled),
                    "trim_strings": bool(self.trim_strings_enabled),
                    "normalize_spaces": bool(self.normalize_spaces_enabled),
                    "fix_typo": bool(self.fix_typo_enabled),
                    "fix_scientific": bool(self.fix_scientific_enabled),
                    "lowercase_strings": bool(getattr(self, "lowercase_strings_enabled", False)),
                    "normalize_text": bool(self.normalize_text_enabled),
                    "clean_context": bool(self.clean_context_enabled),
                    "remove_outliers": bool(self.remove_outliers_enabled),
                    "fill_missing_values": bool(self.fill_missing_values_enabled),
                    "auto_convert_types": bool(self.auto_convert_types_enabled),
                    "remove_empty": bool(self.remove_empty_enabled),
                    "convert_units": bool(self.convert_units_enabled),
                    "unit_mode": str(self.unit_mode),
                    "unit_parse_threshold": float(self.unit_parse_threshold),
                    "unit_target_by_category": dict(self.unit_target_by_category) if self.unit_target_by_category else None,
                    "unit_target_by_column": dict(self.unit_target_by_column) if self.unit_target_by_column else {},
                    "mask_sensitive_data": bool(getattr(self, "mask_sensitive_data_enabled", False)),
                    "missing_threshold": float(self.missing_threshold),
                    "missing_axis": str(self.missing_axis),
                    "duplicate_keep": str(self.duplicate_keep),
                    "outlier_method": str(self.outlier_method),
                    "outlier_threshold": float(self.outlier_threshold),
                    "fill_strategy": str(self.fill_strategy),
                },
                "column_renames": {},
                "structure_fixes": [],
                "dtype_conversions": {},
                "removals": {"rows": [], "columns": []},
            }
            
            logger.info(f"Début du nettoyage: {df.shape[0]} lignes × {df.shape[1]} colonnes")
            
            # Phase 1: Corrections structurelles
            if validate_structure:
                df = self.fix_column_names(df, inplace=True)
                df = self.fix_table_structure(df, inplace=True)
            
            # Phase 2: Corrections d'encodage
            if fix_encoding:
                df = self.normalize_encoding(df, inplace=True)
            
            # Phase 3: Nettoyages de base (ordre optimal)
            
            # 3a. Supprimer doublons d'abord
            if self.remove_duplicates_enabled:
                df = self.remove_duplicates(df)
            
            # 3b. Trim strings
            if self.trim_strings_enabled:
                df = self.trim_all_strings(df)
            
            # 3c. Remplacer chaînes vides
            if self.remove_empty_enabled:
                df = self.remove_empty_strings(df)

            # 3d. Masquage données sensibles (opt-in) — avant normalisations (évite de casser IP/IBAN/etc.)
            if getattr(self, "mask_sensitive_data_enabled", False):
                df = self.mask_sensitive_data_in_dataframe(df)
            
            # 3e. Normalisation texte
            if self.normalize_spaces_enabled:
                df = self.normalize_spaces(df)
            
            if self.fix_typo_enabled:
                df = self.fix_typography(df)

            if getattr(self, "lowercase_strings_enabled", False):
                df = self.lowercase_all_strings(df)
            
            if self.fix_scientific_enabled:
                df = self.fix_scientific_notation(df)
            
            if self.normalize_text_enabled:
                df = self.normalize_text_full(df)
            
            if self.clean_context_enabled:
                df = self.clean_context(df)

            # 3e. Extraction / conversion d'unités (opt-in, non destructif par défaut)
            if self.convert_units_enabled:
                df = self.convert_units_in_dataframe(df)
            
            # Phase 4: Optimisation des types
            if auto_detect_types or self.auto_convert_types_enabled:
                type_conversions = self.detect_and_fix_data_types(df, inplace=True)
                if type_conversions:
                    self._stats["type_conversions"] = type_conversions
            
            # Phase 5: Traitement des valeurs manquantes
            if self.fill_missing_values_enabled:
                df = self.fill_missing(df)
            
            # Phase 6: Suppression outliers (après fill pour éviter NaN)
            if self.remove_outliers_enabled:
                df = self.remove_outliers(df)
            
            # Phase 7: Suppression colonnes/lignes avec trop de manquants (à la fin)
            if self.drop_missing_enabled:
                df = self.remove_missing_values(df)
            
            # Validation finale
            self._validate_dataframe(df)
            
            # Statistiques finales
            self._stats.update({
                "rows_after": len(df),
                "cols_after": len(df.columns),
                "memory_after_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
                "dtypes_after": dict(df.dtypes.astype(str)),
                "completed_at": datetime.now().isoformat()
            })
            
            # Calculs dérivés
            self._stats["rows_change"] = self._stats["rows_after"] - self._stats["rows_before"]
            self._stats["cols_change"] = self._stats["cols_after"] - self._stats["cols_before"]
            self._stats["memory_reduction_mb"] = self._stats["memory_before_mb"] - self._stats["memory_after_mb"]
            self._stats["memory_reduction_percent"] = (
                self._stats["memory_reduction_mb"] / self._stats["memory_before_mb"] * 100 
                if self._stats["memory_before_mb"] > 0 else 0
            )
            
            logger.info(
                f"Nettoyage terminé: {df.shape[0]} lignes × {df.shape[1]} colonnes "
                f"(Δ lignes: {self._stats['rows_change']}, Δ colonnes: {self._stats['cols_change']}, "
                f"Réduction mémoire: {self._stats['memory_reduction_percent']:.1f}%)"
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {e}")
            raise CleanerError(f"Échec du nettoyage: {e}") from e
    
    def get_stats(self, detailed: bool = False) -> Dict[str, Any]:
        """
        Retourne les statistiques du dernier nettoyage.
        
        Args:
            detailed: Inclure les statistiques détaillées
            
        Returns:
            Dictionnaire des statistiques
        """
        stats = self._stats.copy()
        
        if detailed and "dtypes_before" in stats and "dtypes_after" in stats:
            # Analyse des changements de types
            dtypes_before = stats["dtypes_before"]
            dtypes_after = stats["dtypes_after"]
            
            type_changes = {}
            for col in set(dtypes_before.keys()) & set(dtypes_after.keys()):
                if dtypes_before[col] != dtypes_after[col]:
                    type_changes[col] = f"{dtypes_before[col]} -> {dtypes_after[col]}"
            
            stats["dtype_changes"] = type_changes
            
            # Statistiques sur les types
            stats["dtype_distribution_before"] = pd.Series(dtypes_before).value_counts().to_dict()
            stats["dtype_distribution_after"] = pd.Series(dtypes_after).value_counts().to_dict()
        
        return stats

    def get_transformation_report(self) -> Dict[str, Any]:
        """Retourne le rapport structuré du dernier nettoyage (JSON-friendly)."""
        report = getattr(self, "_report", None)
        if not isinstance(report, dict):
            report = {}

        # Ajouter un résumé utile à la fin
        try:
            report = dict(report)
            report.setdefault("summary", {})
            report["summary"].update({
                "rows_before": int(self._stats.get("rows_before", 0) or 0),
                "rows_after": int(self._stats.get("rows_after", 0) or 0),
                "cols_before": int(self._stats.get("cols_before", 0) or 0),
                "cols_after": int(self._stats.get("cols_after", 0) or 0),
                "operations": list(self._stats.get("operations", []) or []),
            })
        except Exception:
            pass

        return report
    
    def print_stats(self, detailed: bool = False):
        """
        Affiche les statistiques du dernier nettoyage.
        
        Args:
            detailed: Afficher les statistiques détaillées
        """
        stats = self.get_stats(detailed=detailed)
        
        print("=" * 80)
        print("📊 RAPPORT DE NETTOYAGE PROFESSIONNEL")
        print("=" * 80)
        
        # Informations de base
        print(f"📅 Début: {stats.get('started_at', 'N/A')}")
        print(f"🏁 Fin: {stats.get('completed_at', 'N/A')}")
        print()
        
        # Dimensions
        rows_before = stats.get('rows_before', 0)
        rows_after = stats.get('rows_after', 0)
        cols_before = stats.get('cols_before', 0)
        cols_after = stats.get('cols_after', 0)
        
        print(f"📏 Dimensions:")
        print(f"  Lignes: {rows_before:,} → {rows_after:,} ({stats.get('rows_change', 0):+,})")
        print(f"  Colonnes: {cols_before} → {cols_after} ({stats.get('cols_change', 0):+})")
        print()
        
        # Mémoire
        if 'memory_before_mb' in stats:
            print(f"💾 Mémoire:")
            print(f"  Avant: {stats['memory_before_mb']:.1f} MB")
            print(f"  Après: {stats['memory_after_mb']:.1f} MB")
            print(f"  Réduction: {stats['memory_reduction_mb']:.1f} MB ({stats['memory_reduction_percent']:.1f}%)")
            print()
        
        # Opérations
        operations = stats.get('operations', [])
        if operations:
            print(f"🔧 Opérations effectuées ({len(operations)}):")
            for i, op in enumerate(operations, 1):
                print(f"  {i:2d}. {op}")
            print()
        
        # Changements de types (si detailed)
        if detailed and 'dtype_changes' in stats:
            type_changes = stats['dtype_changes']
            if type_changes:
                print(f"🔄 Types de données modifiés ({len(type_changes)}):")
                for col, change in list(type_changes.items())[:10]:
                    print(f"  {col}: {change}")
                if len(type_changes) > 10:
                    print(f"  ... et {len(type_changes) - 10} autres")
                print()
        
        # Conversions de types personnalisées
        if 'type_conversions' in stats:
            conversions = stats['type_conversions']
            print(f"🎯 Conversions de types détectées ({len(conversions)}):")
            for col, conversion in list(conversions.items())[:10]:
                print(f"  {col}: {conversion}")
            if len(conversions) > 10:
                print(f"  ... et {len(conversions) - 10} autres")
            print()
        
        # Résumé qualité
        if rows_before > 0:
            retention_rate = (rows_after / rows_before) * 100
            print(f"✅ Qualité:")
            print(f"  Rétention des données: {retention_rate:.1f}%")
            
            if retention_rate >= 95:
                quality = "Excellente 🏆"
            elif retention_rate >= 85:
                quality = "Bonne ✅"
            elif retention_rate >= 70:
                quality = "Acceptable ⚠️"
            else:
                quality = "Attention requise 🚨"
            
            print(f"  Évaluation: {quality}")
            print()
        
        print("=" * 80)
    
    # ============= MÉTHODES SMART CLEAN =============
    
    def detect_column_type(self, series: pd.Series, sample_size: int = 1000) -> Dict[str, Any]:
        """Détecte intelligemment le type d'une colonne."""
        sample = series.dropna().astype(str).head(sample_size)
        if len(sample) == 0:
            return {'type': 'empty', 'confidence': 1.0, 'patterns': []}
        
        results = {'type': 'unknown', 'confidence': 0.0, 'patterns': [], 'stats': {}}
        pattern_matches = {}
        
        for pattern_name, pattern in self.compiled_patterns.items():
            matches = sample.str.contains(pattern, regex=True, na=False).sum()
            if matches > 0:
                match_ratio = matches / len(sample)
                pattern_matches[pattern_name] = match_ratio
                if match_ratio > 0.5:
                    results['patterns'].append({'name': pattern_name, 'ratio': match_ratio, 'count': int(matches)})
        
        if pattern_matches:
            best_pattern = max(pattern_matches.items(), key=lambda x: x[1])
            results['type'] = best_pattern[0]
            results['confidence'] = best_pattern[1]
        
        results['stats']['unique_ratio'] = float(series.nunique() / len(series))
        return results
    
    def extract_emails(self, series: pd.Series) -> pd.Series:
        """Extrait et valide les emails."""
        def extract_email(text):
            if pd.isna(text):
                return None
            match = self.compiled_patterns['email'].search(str(text))
            if match:
                email = match.group(0)
                if EMAIL_VALIDATOR_AVAILABLE:
                    try:
                        valid = validate_email(email)
                        return valid.email
                    except EmailNotValidError:
                        return None
                return email
            return None
        return series.apply(extract_email)
    
    def extract_phones(self, series: pd.Series, region: str = 'FR') -> pd.Series:
        """Extrait et normalise les numéros de téléphone."""
        def extract_phone(text):
            if pd.isna(text):
                return None
            text = str(text)
            if PHONENUMBERS_AVAILABLE:
                try:
                    parsed = phonenumbers.parse(text, region)
                    if phonenumbers.is_valid_number(parsed):
                        return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
                except:
                    pass
            for pattern_name in ['phone_fr', 'phone_us', 'phone_generic']:
                match = self.compiled_patterns[pattern_name].search(text)
                if match:
                    return match.group(0)
            return None
        return series.apply(extract_phone)
    
    def extract_units(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Extrait valeurs et unités séparément."""
        def parse_unit(text):
            if pd.isna(text):
                return None, None
            text = str(text)
            if PINT_AVAILABLE:
                try:
                    quantity = ureg.parse_expression(text)
                    return float(quantity.magnitude), str(quantity.units)
                except:
                    pass
            match = re.search(r'([\d.,]+)\s*([a-zA-Z°/]+)', text)
            if match:
                value = float(match.group(1).replace(',', '.'))
                unit = match.group(2)
                return value, unit
            return None, None
        
        parsed = series.apply(parse_unit)
        values = parsed.apply(lambda x: x[0] if x else None)
        units = parsed.apply(lambda x: x[1] if x else None)
        return values, units
    
    def smart_clean(self, df: pd.DataFrame, extract_columns: bool = True, min_confidence: float = 0.5, verbose: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Nettoyage intelligent avec extraction automatique.
        
        Args:
            df: DataFrame à nettoyer
            extract_columns: Créer des colonnes dérivées
            min_confidence: Confiance minimum pour extraction (0-1)
            verbose: Afficher les informations
            
        Returns:
            (df_clean, metadata)
        """
        df_clean = df.copy()
        metadata = {'columns_analyzed': {}, 'columns_created': [], 'stats': {}}
        
        for col in df.columns:
            if df[col].dtype not in ['object', 'string']:
                continue
            
            type_info = self.detect_column_type(df[col])
            metadata['columns_analyzed'][col] = type_info
            
            if extract_columns and type_info['confidence'] >= min_confidence:
                detected_type = type_info['type']
                
                if 'email' in detected_type:
                    new_col = f"{col}_email_extracted"
                    df_clean[new_col] = self.extract_emails(df[col])
                    metadata['columns_created'].append(new_col)
                
                elif 'phone' in detected_type:
                    new_col = f"{col}_phone_normalized"
                    df_clean[new_col] = self.extract_phones(df[col])
                    metadata['columns_created'].append(new_col)
                
                elif 'unit_' in detected_type:
                    col_value = f"{col}_value"
                    col_unit = f"{col}_unit"
                    df_clean[col_value], df_clean[col_unit] = self.extract_units(df[col])
                    metadata['columns_created'].extend([col_value, col_unit])
                
                elif 'price' in detected_type or 'currency' in detected_type:
                    new_col = f"{col}_amount"
                    df_clean[new_col] = df[col].astype(str).str.extract(r'([\d.,]+)')[0].str.replace(',', '.').astype(float)
                    metadata['columns_created'].append(new_col)
                
                elif 'date' in detected_type and DATEUTIL_AVAILABLE:
                    new_col = f"{col}_parsed"
                    df_clean[new_col] = pd.to_datetime(df[col], errors='coerce')
                    metadata['columns_created'].append(new_col)
        
        metadata['stats']['columns_original'] = len(df.columns)
        metadata['stats']['columns_final'] = len(df_clean.columns)
        metadata['stats']['columns_added'] = len(metadata['columns_created'])
        
        return df_clean, metadata


