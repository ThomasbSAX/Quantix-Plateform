from typing import Iterable, Any, Literal
from collections import Counter


def central_tendency(
    values: Iterable[Any],
    *,
    kind: Literal["median", "mode"] = "median"
) -> Any:
    """
    Calcule une statistique de tendance centrale.

    median : médiane (uniquement pour données numériques)
    mode   : mode (numérique ou catégoriel)
    """
    vals = list(values)
    if not vals:
        raise ValueError("ensemble vide")

    if kind == "median":
        # vérification numérique stricte
        try:
            nums = sorted(float(v) for v in vals)
        except Exception:
            raise TypeError("la médiane nécessite des valeurs numériques")

        n = len(nums)
        mid = n // 2

        if n % 2 == 1:
            return nums[mid]
        return 0.5 * (nums[mid - 1] + nums[mid])

    if kind == "mode":
        c = Counter(vals)
        max_freq = max(c.values())
        modes = [v for v, k in c.items() if k == max_freq]

        # convention statistique : mode unique si possible
        return modes[0] if len(modes) == 1 else modes

    raise ValueError(f"statistique inconnue : {kind}")
