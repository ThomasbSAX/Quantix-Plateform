import pandas as pd
from typing import Optional, List


def remove_duplicates(
    df: pd.DataFrame,
    *,
    subset: Optional[List[str]] = None,
    keep: str = "first"
) -> pd.DataFrame:
    """
    Supprime les lignes dupliquées.
    
    Args:
        df: DataFrame
        subset: Colonnes à considérer (None = toutes)
        keep: 'first', 'last', ou False (supprimer tous les doublons)
    
    Returns:
        DataFrame sans doublons
    """
    return df.drop_duplicates(subset=subset, keep=keep)
