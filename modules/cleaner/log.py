def set_log_axis(ax, *, x: bool = False, y: bool = False, base: int = 10) -> None:
    """
    Met un axe en échelle logarithmique (usage scientifique standard).
    ax : objet matplotlib Axes
    x, y : activer log sur l’axe x et/ou y
    base : base du logarithme (10 par défaut)
    """
    if x:
        ax.set_xscale("log", base=base)
    if y:
        ax.set_yscale("log", base=base)
