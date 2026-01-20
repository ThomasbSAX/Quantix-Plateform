"""Module universel de conversion d'unités.

Ce module essaye de charger des définitions d'unités depuis un package Python
`data` (ex: `data/length.py`). Dans ce dépôt, le dossier `data/` contient des
fichiers tabulaires (CSV, etc.) et n'est pas un package Python.

Pour éviter que l'import de l'orchestrateur `Cleaner` casse, on fournit:
- un mode "best effort" : si un package `data` compatible existe, on l'utilise
- un fallback minimal interne couvrant les unités les plus courantes

Note importante:
- AUCUNE conversion de monnaie n'est implémentée volontairement (€, $, £, etc.),
  car les taux évoluent. Les regex ci-dessous évitent aussi d'interpréter les
  symboles monétaires comme des unités.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


# Symboles monétaires explicitement exclus de la détection d'unités
_CURRENCY_SYMBOLS = {"€", "$", "£", "¥", "₽", "₩", "₹", "₺", "₫", "₪", "₦", "₱", "฿"}


def _coerce_float(x: Any) -> float:
    """Convertit en float en gérant formats FR (virgule) et séparateurs de milliers."""
    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip()
    if not s:
        raise ValueError("Valeur numérique vide")

    # normaliser espaces insécables
    s = s.replace("\u00A0", " ")
    # enlever séparateurs de milliers courants (espaces)
    s = re.sub(r"(?<=\d)\s+(?=\d{3}(\D|$))", "", s)

    # Si on a à la fois ',' et '.', on suppose que le dernier est le séparateur décimal.
    if "," in s and "." in s:
        last_comma = s.rfind(",")
        last_dot = s.rfind(".")
        if last_comma > last_dot:
            # ',' décimal, '.' milliers
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            # '.' décimal, ',' milliers
            s = s.replace(",", "")
    else:
        # Sinon, virgule décimale possible
        if "," in s and re.search(r",\d+$", s):
            s = s.replace(",", ".")

    return float(s)


def _normalize_unit_token(raw: str) -> str:
    """Normalise une écriture d'unité (sans décider de l'unité canonique).

    - casse: lower (sauf °C/°F gérées par alias)
    - micro: µ -> u
    - exposants: ²/³ -> 2/3
    - espaces: retirés
    - 'per' et séparateurs: / conservé
    """
    s = str(raw).strip()
    if not s:
        return ""
    # supprimer ponctuation de bord
    s = s.strip("()[]{}.,;:")
    s = s.replace("µ", "u")
    s = s.replace("²", "2").replace("³", "3")
    s = s.replace("·", "*")
    s = s.replace(" ", "")
    return s


@dataclass(frozen=True)
class UnitAlias:
    canonical: str
    pattern: re.Pattern[str]


def _compile_aliases() -> List[UnitAlias]:
    """Compile une liste de patterns regex pour reconnaitre les unités.

    On préfère des patterns stricts (match complet) pour limiter les faux positifs.
    """

    def full(p: str) -> re.Pattern[str]:
        return re.compile(rf"^(?:{p})$", re.IGNORECASE)

    aliases: List[UnitAlias] = []

    # ── Longueur
    aliases += [
        UnitAlias("mm", full(r"mm|millimet(?:er|re)s?|millim[èe]tres?")),
        UnitAlias("cm", full(r"cm|centimet(?:er|re)s?|centim[èe]tres?")),
        UnitAlias("m", full(r"m|met(?:er|re)s?|m[èe]tres?")),
        UnitAlias("km", full(r"km|kilomet(?:er|re)s?|kilom[èe]tres?")),
        UnitAlias("in", full(r"in|inch(?:es)?|\"")),
        UnitAlias("ft", full(r"ft|foot|feet|'")),
        UnitAlias("yd", full(r"yd|yard(?:s)?")),
        UnitAlias("mi", full(r"mi|mile(?:s)?")),
        UnitAlias("nmi", full(r"nmi|nauticalmile(?:s)?|nmile(?:s)?")),
    ]

    # ── Masse
    aliases += [
        UnitAlias("ug", full(r"ug|microgram(?:me)?s?|microg")),
        UnitAlias("mg", full(r"mg|milligram(?:me)?s?")),
        UnitAlias("g", full(r"g|gram(?:me)?s?")),
        UnitAlias("kg", full(r"kg|kilogram(?:me)?s?|kilo(?:s)?")),
        UnitAlias("t", full(r"t|tonne(?:s)?|metricton(?:s)?")),
        UnitAlias("lb", full(r"lb|lbs|pound(?:s)?")),
        UnitAlias("oz", full(r"oz|ounce(?:s)?")),
        UnitAlias("st", full(r"st|stone(?:s)?")),
    ]

    # ── Temps
    aliases += [
        UnitAlias("ms", full(r"ms|millisecond(?:e)?s?")),
        UnitAlias("s", full(r"s|sec|secs|second(?:e)?s?")),
        UnitAlias("min", full(r"min|mins|minute(?:s)?")),
        UnitAlias("h", full(r"h|hr|hrs|hour(?:s)?|heure(?:s)?")),
        UnitAlias("day", full(r"d|day(?:s)?|jour(?:s)?")),
        UnitAlias("week", full(r"w|week(?:s)?|semaine(?:s)?")),
        UnitAlias("month", full(r"mo|month(?:s)?|mois")),
        UnitAlias("year", full(r"y|yr|yrs|year(?:s)?|an(?:s)?|ann[ée]e(?:s)?")),
    ]

    # ── Température (canonical = C/F/K)
    aliases += [
        UnitAlias("C", full(r"c|°c|degc|celsius")),
        UnitAlias("F", full(r"f|°f|degf|fahrenheit")),
        UnitAlias("K", full(r"k|kelvin")),
    ]

    # ── Volume
    aliases += [
        UnitAlias("ml", full(r"ml|milliliter(?:s)?|millilitre(?:s)?")),
        UnitAlias("cl", full(r"cl|centiliter(?:s)?|centilitre(?:s)?")),
        UnitAlias("dl", full(r"dl|deciliter(?:s)?|decilitre(?:s)?")),
        UnitAlias("l", full(r"l|lt|liter(?:s)?|litre(?:s)?")),
        UnitAlias("m3", full(r"m3|m\^3|m\*\*3|cubicm(?:eter|etre)s?|m[èe]tre(?:s)?cube(?:s)?")),
        UnitAlias("cm3", full(r"cm3|cm\^3|cubiccm|centim[èe]tre(?:s)?cube(?:s)?")),
        UnitAlias("in3", full(r"in3|in\^3|cubicinch(?:es)?")),
        UnitAlias("ft3", full(r"ft3|ft\^3|cubicfoot|cubicfeet")),
        UnitAlias("gal", full(r"gal|gallon(?:s)?")),
        UnitAlias("qt", full(r"qt|quart(?:s)?")),
        UnitAlias("pt", full(r"pt|pint(?:s)?")),
        UnitAlias("floz", full(r"floz|fl-?oz|fluidounce(?:s)?")),
    ]

    # ── Surface
    aliases += [
        UnitAlias("mm2", full(r"mm2|mm\^2|mm\*\*2")),
        UnitAlias("cm2", full(r"cm2|cm\^2|cm\*\*2")),
        UnitAlias("m2", full(r"m2|m\^2|m\*\*2|sqm|m[èe]tre(?:s)?carr[ée](?:s)?")),
        UnitAlias("km2", full(r"km2|km\^2|km\*\*2")),
        UnitAlias("ha", full(r"ha|hectare(?:s)?")),
        UnitAlias("acre", full(r"acre(?:s)?")),
        UnitAlias("ft2", full(r"ft2|ft\^2|sqft")),
        UnitAlias("yd2", full(r"yd2|yd\^2|sqyd")),
    ]

    # ── Vitesse
    aliases += [
        UnitAlias("m/s", full(r"m/s|mps|m\*s-?1")),
        UnitAlias("km/h", full(r"km/h|kmh|kph")),
        UnitAlias("mph", full(r"mph|mi/h")),
        UnitAlias("knot", full(r"knot(?:s)?|kt|kts")),
        UnitAlias("ft/s", full(r"ft/s|fps")),
    ]

    # ── Pression
    aliases += [
        UnitAlias("pa", full(r"pa|pascal(?:s)?")),
        UnitAlias("kpa", full(r"kpa")),
        UnitAlias("mpa", full(r"mpa")),
        UnitAlias("bar", full(r"bar")),
        UnitAlias("mbar", full(r"mbar|hpa")),
        UnitAlias("psi", full(r"psi")),
        UnitAlias("atm", full(r"atm|atmosphere(?:s)?")),
        UnitAlias("mmhg", full(r"mmhg|torr")),
    ]

    # ── Énergie
    aliases += [
        UnitAlias("j", full(r"j|joule(?:s)?")),
        UnitAlias("kj", full(r"kj|kilojoule(?:s)?")),
        UnitAlias("mj", full(r"mj|megajoule(?:s)?")),
        UnitAlias("wh", full(r"wh|w\*h")),
        UnitAlias("kwh", full(r"kwh")),
        UnitAlias("cal", full(r"cal")),
        UnitAlias("kcal", full(r"kcal|calorie(?:s)?")),
        UnitAlias("btu", full(r"btu")),
    ]

    # ── Puissance
    aliases += [
        UnitAlias("w", full(r"w|watt(?:s)?")),
        UnitAlias("kw", full(r"kw|kilowatt(?:s)?")),
        UnitAlias("mw", full(r"mw|megawatt(?:s)?")),
        UnitAlias("hp", full(r"hp|ch|cheval(?:s)?")),
    ]

    return aliases


_ALIASES: List[UnitAlias] = _compile_aliases()


def normalize_unit(raw_unit: str) -> Optional[str]:
    """Retourne l'unité canonique (clé de UNIT_MAPPING) ou None si inconnue.

    Exemples:
    - "mètres" -> "m"
    - "°c" -> "C"
    - "kmh" -> "km/h"
    - "m²" -> "m2"
    """
    token = _normalize_unit_token(raw_unit)
    if not token:
        return None

    # Exclure explicitement les symboles monétaires
    if token in _CURRENCY_SYMBOLS:
        return None
    if any(sym in token for sym in _CURRENCY_SYMBOLS):
        return None

    # Cas exponentiels déjà normalisés (²/³ -> 2/3)
    for alias in _ALIASES:
        if alias.pattern.match(token):
            return alias.canonical
    return None


# Regex pour extraire "valeur + unité" dans une chaîne.
# On évite les monnaies en excluant les symboles monétaires.
_QUANTITY_RE = re.compile(
    r"(?P<value>[+-]?(?:\d+(?:[ \u00A0]\d{3})*|\d*)(?:[.,]\d+)?(?:[eE][+-]?\d+)?)"
    r"\s*(?P<unit>(?![€$£¥₽₩₹₺₫₪₦₱฿])[A-Za-z°µ\"'](?:[A-Za-z0-9°µ\"'/*^._-]|²|³)*)",
    re.UNICODE,
)


def extract_quantity(text: str) -> Optional[Tuple[float, str]]:
    """Extrait la première occurrence (valeur, unité_canonique) d'une chaîne."""
    if text is None:
        return None
    s = str(text)
    m = _QUANTITY_RE.search(s)
    if not m:
        return None
    raw_value = m.group("value")
    raw_unit = m.group("unit")
    unit = normalize_unit(raw_unit)
    if not unit:
        return None
    value = _coerce_float(raw_value)
    return value, unit


def _fallback_convert_length(value: float, from_unit: str, to_unit: str) -> float:
    LENGTH_TO_METER = {
        "mm": 0.001,
        "cm": 0.01,
        "m": 1.0,
        "km": 1000.0,
        "in": 0.0254,
        "ft": 0.3048,
        "yd": 0.9144,
        "mi": 1609.344,
        "nmi": 1852.0,
    }
    return value * LENGTH_TO_METER[from_unit] / LENGTH_TO_METER[to_unit]


def _fallback_convert_weight(value: float, from_unit: str, to_unit: str) -> float:
    WEIGHT_TO_KG = {
        "ug": 1e-9,
        "mg": 1e-6,
        "g": 1e-3,
        "kg": 1.0,
        "t": 1000.0,
        "lb": 0.45359237,
        "oz": 0.028349523125,
        "st": 6.35029318,
    }
    return value * WEIGHT_TO_KG[from_unit] / WEIGHT_TO_KG[to_unit]


def _fallback_convert_time(value: float, from_unit: str, to_unit: str) -> float:
    TIME_TO_S = {
        "ms": 0.001,
        "s": 1.0,
        "min": 60.0,
        "h": 3600.0,
        "day": 86400.0,
        "week": 604800.0,
        "month": 2629800.0,  # mois moyen (30.44j) : approximation
        "year": 31557600.0,  # année moyenne (365.25j) : approximation
    }
    return value * TIME_TO_S[from_unit] / TIME_TO_S[to_unit]


def _fallback_convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    if from_unit == to_unit:
        return value
    # Convert to Celsius
    if from_unit in {"°C", "C"}:
        c = value
    elif from_unit in {"K"}:
        c = value - 273.15
    elif from_unit in {"°F", "F"}:
        c = (value - 32) * 5 / 9
    else:
        raise ValueError(f"Unité température non supportée: {from_unit}")

    # Convert from Celsius
    if to_unit in {"°C", "C"}:
        return c
    if to_unit == "K":
        return c + 273.15
    if to_unit in {"°F", "F"}:
        return c * 9 / 5 + 32
    raise ValueError(f"Unité température non supportée: {to_unit}")


def _fallback_convert_volume(value: float, from_unit: str, to_unit: str) -> float:
    # base m3
    VOLUME_TO_M3 = {
        "ml": 1e-6,
        "cl": 1e-5,
        "dl": 1e-4,
        "l": 1e-3,
        "cm3": 1e-6,
        "m3": 1.0,
        "in3": 1.6387064e-5,
        "ft3": 0.028316846592,
        "gal": 0.003785411784,  # US gallon
        "qt": 0.000946352946,   # US quart
        "pt": 0.000473176473,   # US pint
        "floz": 2.95735295625e-5,  # US fl oz
    }
    return value * VOLUME_TO_M3[from_unit] / VOLUME_TO_M3[to_unit]


def _fallback_convert_area(value: float, from_unit: str, to_unit: str) -> float:
    AREA_TO_M2 = {
        "mm2": 1e-6,
        "cm2": 1e-4,
        "m2": 1.0,
        "km2": 1e6,
        "ha": 1e4,
        "acre": 4046.8564224,
        "ft2": 0.09290304,
        "yd2": 0.83612736,
    }
    return value * AREA_TO_M2[from_unit] / AREA_TO_M2[to_unit]


def _fallback_convert_speed(value: float, from_unit: str, to_unit: str) -> float:
    SPEED_TO_MPS = {
        "m/s": 1.0,
        "km/h": 1000.0 / 3600.0,
        "mph": 1609.344 / 3600.0,
        "knot": 1852.0 / 3600.0,
        "ft/s": 0.3048,
    }
    return value * SPEED_TO_MPS[from_unit] / SPEED_TO_MPS[to_unit]


def _fallback_convert_pressure(value: float, from_unit: str, to_unit: str) -> float:
    PRESSURE_TO_PA = {
        "pa": 1.0,
        "kpa": 1e3,
        "mpa": 1e6,
        "bar": 1e5,
        "mbar": 1e2,
        "psi": 6894.757293168,
        "atm": 101325.0,
        "mmhg": 133.322387415,
    }
    return value * PRESSURE_TO_PA[from_unit] / PRESSURE_TO_PA[to_unit]


def _fallback_convert_energy(value: float, from_unit: str, to_unit: str) -> float:
    ENERGY_TO_J = {
        "j": 1.0,
        "kj": 1e3,
        "mj": 1e6,
        "wh": 3600.0,
        "kwh": 3.6e6,
        "cal": 4.184,
        "kcal": 4184.0,
        "btu": 1055.05585262,
    }
    return value * ENERGY_TO_J[from_unit] / ENERGY_TO_J[to_unit]


def _fallback_convert_power(value: float, from_unit: str, to_unit: str) -> float:
    POWER_TO_W = {
        "w": 1.0,
        "kw": 1e3,
        "mw": 1e6,
        "hp": 745.6998715822702,  # mechanical horsepower
    }
    return value * POWER_TO_W[from_unit] / POWER_TO_W[to_unit]


def _try_build_mapping_from_external_data() -> Dict[str, Tuple[str, Callable]] | None:
    try:
        from data import length, weight, time_units, temperature, speed, area, volume  # type: ignore
        from data import data as data_module  # type: ignore
        from data import energy  # type: ignore
    except Exception:
        return None

    mapping: Dict[str, Tuple[str, Callable]] = {}

    if hasattr(length, "LENGTH_TO_METER"):
        for symbol in length.LENGTH_TO_METER.keys():
            mapping[symbol] = ("length", length.convert_length)

    if hasattr(weight, "WEIGHT_TO_KILOGRAM"):
        for symbol in weight.WEIGHT_TO_KILOGRAM.keys():
            mapping[symbol] = ("weight", weight.convert_weight)

    if hasattr(time_units, "TIME_TO_SECOND"):
        for symbol in time_units.TIME_TO_SECOND.keys():
            mapping[symbol] = ("time", time_units.convert_time)

    temp_symbols = ["K", "°C", "C", "°F", "F", "°R", "°De", "°N", "°Ré", "°Rø"]
    for symbol in temp_symbols:
        mapping[symbol] = ("temperature", temperature.convert_temperature)

    if hasattr(speed, "SPEED_TO_MPS"):
        for symbol in speed.SPEED_TO_MPS.keys():
            mapping[symbol] = ("speed", speed.convert_speed)

    if hasattr(area, "AREA_TO_SQM"):
        for symbol in area.AREA_TO_SQM.keys():
            mapping[symbol] = ("area", area.convert_area)

    if hasattr(volume, "VOLUME_TO_M3"):
        for symbol in volume.VOLUME_TO_M3.keys():
            mapping[symbol] = ("volume", volume.convert_volume)

    if hasattr(data_module, "DATA_TO_BIT"):
        for symbol in data_module.DATA_TO_BIT.keys():
            mapping[symbol] = ("data", data_module.convert_data)

    if hasattr(energy, "ENERGY_TO_JOULE"):
        for symbol in energy.ENERGY_TO_JOULE.keys():
            mapping[symbol] = ("energy", energy.convert_energy)

    return mapping


def build_unit_mapping() -> Dict[str, Tuple[str, Callable]]:
    """
    Construit automatiquement le mapping des unités depuis les fichiers data/*.py
    en lisant les dictionnaires *_TO_* de chaque module.
    
    Returns:
        Dict[str, Tuple[str, Callable]]: Mapping {symbole: (catégorie, fonction_conversion)}
    """
    external = _try_build_mapping_from_external_data()
    if external:
        return external

    # Fallback minimal
    mapping: Dict[str, Tuple[str, Callable]] = {}
    for u in ["mm", "cm", "m", "km", "in", "ft", "yd", "mi", "nmi"]:
        mapping[u] = ("length", _fallback_convert_length)
    for u in ["ug", "mg", "g", "kg", "t", "lb", "oz", "st"]:
        mapping[u] = ("weight", _fallback_convert_weight)
    for u in ["ms", "s", "min", "h", "day", "week", "month", "year"]:
        mapping[u] = ("time", _fallback_convert_time)
    for u in ["K", "°C", "C", "°F", "F"]:
        mapping[u] = ("temperature", _fallback_convert_temperature)

    for u in ["ml", "cl", "dl", "l", "cm3", "m3", "in3", "ft3", "gal", "qt", "pt", "floz"]:
        mapping[u] = ("volume", _fallback_convert_volume)

    for u in ["mm2", "cm2", "m2", "km2", "ha", "acre", "ft2", "yd2"]:
        mapping[u] = ("area", _fallback_convert_area)

    for u in ["m/s", "km/h", "mph", "knot", "ft/s"]:
        mapping[u] = ("speed", _fallback_convert_speed)

    for u in ["pa", "kpa", "mpa", "bar", "mbar", "psi", "atm", "mmhg"]:
        mapping[u] = ("pressure", _fallback_convert_pressure)

    for u in ["j", "kj", "mj", "wh", "kwh", "cal", "kcal", "btu"]:
        mapping[u] = ("energy", _fallback_convert_energy)

    for u in ["w", "kw", "mw", "hp"]:
        mapping[u] = ("power", _fallback_convert_power)

    return mapping


# Construire le mapping au chargement du module
UNIT_MAPPING = build_unit_mapping()



def convert(value, from_unit, to_unit):
    """
    Fonction universelle de conversion d'unités.
    Détecte automatiquement le type d'unité et applique la bonne conversion.
    
    Args:
        value: Valeur numérique à convertir
        from_unit: Unité source (symbole)
        to_unit: Unité cible (symbole)
    
    Returns:
        Valeur convertie
    
    Raises:
        ValueError: Si les unités ne sont pas compatibles ou non reconnues
    
    Examples:
        >>> convert(1, 'km', 'm')
        1000.0
        >>> convert(100, 'kg', 'lb')
        220.46226218487758
        >>> convert(1, 'h', 'min')
        60.0
        >>> convert(0, '°C', '°F')
        32.0
    """
    # Normaliser unités si besoin (ex: "mètres", "°c", "kmh")
    from_unit_norm = normalize_unit(from_unit) if isinstance(from_unit, str) else None
    to_unit_norm = normalize_unit(to_unit) if isinstance(to_unit, str) else None

    from_unit = from_unit_norm or from_unit
    to_unit = to_unit_norm or to_unit

    # Vérifier que les deux unités existent
    if from_unit not in UNIT_MAPPING:
        raise ValueError(f"Unité source '{from_unit}' non reconnue")
    if to_unit not in UNIT_MAPPING:
        raise ValueError(f"Unité cible '{to_unit}' non reconnue")
    
    # Récupérer les catégories des unités
    from_category, from_func = UNIT_MAPPING[from_unit]
    to_category, to_func = UNIT_MAPPING[to_unit]
    
    # Vérifier que les unités sont de la même catégorie
    if from_category != to_category:
        raise ValueError(
            f"Impossible de convertir '{from_unit}' ({from_category}) "
            f"vers '{to_unit}' ({to_category}). Unités incompatibles."
        )
    
    # Effectuer la conversion
    return from_func(value, from_unit, to_unit)


def get_unit_category(unit):
    """
    Retourne la catégorie d'une unité.
    
    Args:
        unit: Symbole de l'unité
    
    Returns:
        Catégorie de l'unité (ex: 'length', 'weight', 'time')
    
    Raises:
        ValueError: Si l'unité n'est pas reconnue
    """
    if unit not in UNIT_MAPPING:
        raise ValueError(f"Unité '{unit}' non reconnue")
    return UNIT_MAPPING[unit][0]


def list_all_units():
    """
    Liste toutes les unités disponibles par catégorie.
    
    Returns:
        Dictionnaire {catégorie: [liste d'unités]}
    """
    categories = {}
    for unit, (category, _) in UNIT_MAPPING.items():
        if category not in categories:
            categories[category] = []
        categories[category].append(unit)
    return categories


def list_unit_aliases() -> Dict[str, List[str]]:
    """Expose les unités canoniques et quelques alias (debug/UX).

    Note: les alias sont gérés par regex, donc ce n'est pas exhaustif.
    """
    out: Dict[str, List[str]] = {}
    for a in _ALIASES:
        out.setdefault(a.canonical, []).append(a.pattern.pattern)
    return out


def is_compatible(unit1, unit2):
    """
    Vérifie si deux unités sont compatibles (même catégorie).
    
    Args:
        unit1: Première unité
        unit2: Deuxième unité
    
    Returns:
        True si compatibles, False sinon
    """
    try:
        cat1 = get_unit_category(unit1)
        cat2 = get_unit_category(unit2)
        return cat1 == cat2
    except ValueError:
        return False


def convert_multiple(value, from_unit, to_units):
    """
    Convertit une valeur vers plusieurs unités à la fois.
    
    Args:
        value: Valeur à convertir
        from_unit: Unité source
        to_units: Liste des unités cibles
    
    Returns:
        Dictionnaire {unité: valeur convertie}
    
    Examples:
        >>> convert_multiple(100, 'km', ['m', 'mi', 'ft'])
        {'m': 100000.0, 'mi': 62.137..., 'ft': 328083.989...}
    """
    results = {}
    for to_unit in to_units:
        try:
            results[to_unit] = convert(value, from_unit, to_unit)
        except ValueError as e:
            results[to_unit] = f"Erreur: {e}"
    return results


# Alias pour compatibilité
convertUnit = convert


__all__ = [
    "UNIT_MAPPING",
    "convert",
    "convertUnit",
    "convert_multiple",
    "extract_quantity",
    "get_unit_category",
    "is_compatible",
    "list_all_units",
    "list_unit_aliases",
    "normalize_unit",
]


