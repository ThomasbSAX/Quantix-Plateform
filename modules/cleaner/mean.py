import math
from typing import Iterable, Literal


def mean(
    values: Iterable[float],
    *,
    kind: Literal["arithmetic", "geometric", "harmonic", "quadratic"] = "arithmetic"
) -> float:
    """
    Calcule une moyenne selon le type choisi.

    arithmetic : moyenne arithmétique
    geometric  : moyenne géométrique (valeurs strictement positives)
    harmonic   : moyenne harmonique (valeurs non nulles)
    quadratic  : moyenne quadratique (RMS)
    """
    vals = list(values)
    if not vals:
        raise ValueError("ensemble vide")

    n = len(vals)

    if kind == "arithmetic":
        return sum(vals) / n

    if kind == "geometric":
        if any(v <= 0 for v in vals):
            raise ValueError("la moyenne géométrique requiert des valeurs > 0")
        return math.exp(sum(math.log(v) for v in vals) / n)

    if kind == "harmonic":
        if any(v == 0 for v in vals):
            raise ValueError("la moyenne harmonique requiert des valeurs non nulles")
        return n / sum(1.0 / v for v in vals)

    if kind == "quadratic":
        return math.sqrt(sum(v * v for v in vals) / n)

    raise ValueError(f"type de moyenne inconnu: {kind}")
