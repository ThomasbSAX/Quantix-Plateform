import re
from typing import Optional
import math


def normalize_scientific_notation(s: Optional[str]) -> Optional[str]:
    """
    Normalise les écritures pseudo-scientifiques en écriture décimale explicite.

    Gère notamment :
    - 1e7, 1E7, 1×10^7, 1x10^7, 1·10^7
    - 10e7, 10E7
    - 3,2e-4 (virgule décimale)
    - espaces parasites : "1 e 7", "1 × 10 ^ 7"

    Ne modifie que les motifs clairement scientifiques.
    """
    if s is None:
        return None
    s = str(s)

    # virgule décimale -> point pour calcul
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)

    def repl_e(match):
        token = match.group(0)
        exp_token = match.group("exp")
        # Garde-fou: éviter les cas type UUID (ex: 123e4567) ou exponents énormes
        if len(exp_token.lstrip("+-")) > 3:
            return token
        base = float(match.group("base"))
        exp = int(exp_token)
        # Float64 max exponent ~308 (au-delà, c'est rarement une vraie donnée tabulaire)
        if exp > 308 or exp < -308:
            return token
        try:
            val = base * (10 ** exp)
        except OverflowError:
            return token
        return str(int(val) if exp >= 0 and base.is_integer() else val)

    def repl_pow(match):
        token = match.group(0)
        exp_token = match.group("exp")
        if len(exp_token.lstrip("+-")) > 3:
            return token
        base = float(match.group("base"))
        exp = int(exp_token)
        if exp > 308 or exp < -308:
            return token
        try:
            val = base * (10 ** exp)
        except OverflowError:
            return token
        return str(int(val) if exp >= 0 and base.is_integer() else val)

    # cas 1 : 1e7, 1E7, 10e-3
    s = re.sub(
        r"(?P<base>\d+(\.\d+)?)\s*[eE]\s*(?P<exp>[+-]?\d+)",
        repl_e,
        s
    )

    # cas 2 : 1×10^7, 1x10^7, 1·10^7
    s = re.sub(
        r"(?P<base>\d+(\.\d+)?)\s*[×x·]\s*10\s*\^\s*(?P<exp>[+-]?\d+)",
        repl_pow,
        s
    )

    return s



def to_scientific_notation(
    s: Optional[str],
    *,
    sig: int = 3,
    decimal: str = "e"
) -> Optional[str]:
    """
    Convertit des nombres écrits en clair vers une notation scientifique standard.

    - détecte entiers et décimaux dans un texte
    - ignore les nombres déjà en notation scientifique
    - gère séparateurs de milliers (espaces, points)
    - conserve le reste du texte inchangé

    sig : nombre de chiffres significatifs
    decimal : 'e' (1.23e6) ou '×10^' (1.23×10^6)
    """
    if s is None:
        return None
    s = str(s)

    # éviter de retraiter ce qui est déjà scientifique
    sci_pat = re.compile(r"\b\d+(\.\d+)?\s*[eE]\s*[+-]?\d+\b")

    def repl(match):
        token = match.group(0)
        if sci_pat.search(token):
            return token

        # normalisation milliers + décimales
        t = token.replace("\u00A0", " ").replace("\u202F", " ")
        t = re.sub(r"(?<=\d)[ .](?=\d{3}\b)", "", t)
        t = t.replace(",", ".")
        try:
            x = float(t)
        except Exception:
            return token

        if x == 0:
            mant, exp = 0.0, 0
        else:
            exp = int(math.floor(math.log10(abs(x))))
            mant = x / (10 ** exp)

        mant_str = f"{mant:.{sig-1}f}".rstrip("0").rstrip(".")
        if decimal == "×10^":
            return f"{mant_str}×10^{exp}"
        return f"{mant_str}e{exp}"

    # nombres décimaux/entiers isolés
    num_pat = re.compile(r"\b\d+(?:[.,]\d+)?(?:[ .]\d{3})*\b")
    return num_pat.sub(repl, s)