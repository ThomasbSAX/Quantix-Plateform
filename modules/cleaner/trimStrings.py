import pandas as pd


def trim_strings(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Supprime les espaces en début/fin de chaînes.
    
    Args:
        df: DataFrame
        columns: Colonnes à traiter (None = toutes les colonnes texte)
    
    Returns:
        DataFrame avec strings nettoyées
    """
    df = df.copy()
    cols = columns if columns else df.select_dtypes(include=['object']).columns
    
    for col in cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    
    return df


def remove_empty_strings(df: pd.DataFrame, replace_with_nan: bool = True) -> pd.DataFrame:
    """
    Supprime ou remplace les chaînes vides.
    
    Args:
        df: DataFrame
        replace_with_nan: Si True, remplace par NaN, sinon supprime les lignes
    
    Returns:
        DataFrame nettoyé
    """
    df = df.copy()
    
    if replace_with_nan:
        # Remplacer chaînes vides par NaN
        df = df.replace(r'^\s*$', pd.NA, regex=True)

        # Remplacer tokens manquants textuels courants par NA (sur colonnes texte uniquement)
        missing_tokens = {"na", "n/a", "null", "none", "nan"}
        for col in df.select_dtypes(include=['object', 'string']).columns:
            try:
                s = df[col].astype('string')
                lowered = s.str.strip().str.lower()
                df[col] = s.mask(lowered.isin(list(missing_tokens)), pd.NA)
            except Exception:
                continue
    else:
        # Supprimer les lignes avec chaînes vides
        mask = (df == '').any(axis=1) | (df.isna()).any(axis=1)
        df = df[~mask]
    
    return df
