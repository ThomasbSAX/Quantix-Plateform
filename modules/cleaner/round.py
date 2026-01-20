import math
from typing import Literal


def round_order(
    x: float,
    *,
    order: int,
    mode: Literal["up", "down"] = "up"
) -> float:
    """
    Arrondit x au supérieur ou à l’inférieur à l’ordre 10^order.

    order = 0  -> unités
    order = 1  -> dizaines
    order = 2  -> centaines
    order = -1 -> dixièmes, etc.
    """
    if order < 0:
        factor = 10 ** (-order)
        return (
            math.ceil(x * factor) / factor
            if mode == "up"
            else math.floor(x * factor) / factor
        )

    factor = 10 ** order
    return (
        math.ceil(x / factor) * factor
        if mode == "up"
        else math.floor(x / factor) * factor
    )
