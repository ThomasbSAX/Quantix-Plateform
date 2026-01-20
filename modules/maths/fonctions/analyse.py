"""
Core numerical analysis module
Designed for OCR → LaTeX → evaluation pipelines
Numerical, heuristic, explicit, typed
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Literal, Tuple, List, Dict


# =============================================================================
# TYPES DE BASE
# =============================================================================

Number = float
Function1D = Callable[[Number], Number]
Function2D = Callable[[Number, Number], Number]


@dataclass
class NumericValue:
    value: float
    error: Optional[float] = None


@dataclass
class LimitResult:
    kind: Literal["finite", "infinite", "divergent"]
    value: Optional[float]


@dataclass
class Extremum:
    x: float
    y: float
    kind: Literal["min", "max"]


# =============================================================================
# ÉVALUATION SÉCURISÉE
# =============================================================================

def safe_eval(f: Function1D, x: float) -> Optional[float]:
    try:
        y = f(x)
        if math.isnan(y) or math.isinf(y):
            return None
        return y
    except Exception:
        return None


# =============================================================================
# LIMITES NUMÉRIQUES
# =============================================================================

def numeric_limit(
    f: Function1D,
    x0: float,
    h: float = 1e-6,
    tol: float = 1e-6
) -> LimitResult:
    left = safe_eval(f, x0 - h)
    right = safe_eval(f, x0 + h)

    if left is None or right is None:
        return LimitResult("divergent", None)

    if abs(left - right) < tol:
        return LimitResult("finite", (left + right) / 2)

    if abs(left) > 1e8 or abs(right) > 1e8:
        return LimitResult("infinite", None)

    return LimitResult("divergent", None)


def limit_infinity(
    f: Function1D,
    direction: Literal["+inf", "-inf"] = "+inf",
    x0: float = 1e6
) -> LimitResult:
    x = x0 if direction == "+inf" else -x0
    y = safe_eval(f, x)

    if y is None:
        return LimitResult("divergent", None)

    if abs(y) < 1e6:
        return LimitResult("finite", y)

    return LimitResult("infinite", None)


# =============================================================================
# CONTINUITÉ (HEURISTIQUE)
# =============================================================================

def is_continuous(
    f: Function1D,
    x0: float,
    eps: float = 1e-6,
    delta: float = 1e-4
) -> bool:
    fx = safe_eval(f, x0)
    if fx is None:
        return False

    left = safe_eval(f, x0 - delta)
    right = safe_eval(f, x0 + delta)

    if left is None or right is None:
        return False

    return abs(left - fx) < eps and abs(right - fx) < eps


# =============================================================================
# DÉRIVÉES NUMÉRIQUES
# =============================================================================

def derivative(
    f: Function1D,
    x: float,
    h: float = 1e-6
) -> NumericValue:
    fx1 = safe_eval(f, x + h)
    fx2 = safe_eval(f, x - h)

    if fx1 is None or fx2 is None:
        raise ValueError("Derivative undefined")

    value = (fx1 - fx2) / (2 * h)
    error = abs(fx1 - 2 * safe_eval(f, x) + fx2) / h**2

    return NumericValue(value, error)


def second_derivative(
    f: Function1D,
    x: float,
    h: float = 1e-5
) -> NumericValue:
    fx1 = safe_eval(f, x + h)
    fx0 = safe_eval(f, x)
    fx2 = safe_eval(f, x - h)

    if fx1 is None or fx0 is None or fx2 is None:
        raise ValueError("Second derivative undefined")

    value = (fx1 - 2 * fx0 + fx2) / h**2
    return NumericValue(value)


# =============================================================================
# INTÉGRATION NUMÉRIQUE
# =============================================================================

def integrate_rectangles(
    f: Function1D,
    a: float,
    b: float,
    n: int = 1000
) -> float:
    h = (b - a) / n
    return sum(f(a + i * h) for i in range(n)) * h


def integrate_trapezoidal(
    f: Function1D,
    a: float,
    b: float,
    n: int = 1000
) -> float:
    h = (b - a) / n
    s = (f(a) + f(b)) / 2
    for i in range(1, n):
        s += f(a + i * h)
    return s * h


def integrate_simpson(
    f: Function1D,
    a: float,
    b: float,
    n: int = 1000
) -> float:
    if n % 2 != 0:
        raise ValueError("n must be even")

    h = (b - a) / n
    s = f(a) + f(b)

    for i in range(1, n):
        coef = 4 if i % 2 else 2
        s += coef * f(a + i * h)

    return s * h / 3


# =============================================================================
# EXTREMUMS
# =============================================================================

def find_extrema(
    f: Function1D,
    a: float,
    b: float,
    n: int = 2000
) -> List[Extremum]:
    xs = np.linspace(a, b, n)
    ys = np.array([safe_eval(f, x) for x in xs])

    extrema: List[Extremum] = []

    for i in range(1, n - 1):
        if ys[i] is None:
            continue
        if ys[i] < ys[i - 1] and ys[i] < ys[i + 1]:
            extrema.append(Extremum(xs[i], ys[i], "min"))
        elif ys[i] > ys[i - 1] and ys[i] > ys[i + 1]:
            extrema.append(Extremum(xs[i], ys[i], "max"))

    return extrema


# =============================================================================
# ZÉROS ET RACINES
# =============================================================================

def find_root_bisection(
    f: Function1D,
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Optional[float]:
    fa = safe_eval(f, a)
    fb = safe_eval(f, b)

    if fa is None or fb is None or fa * fb > 0:
        return None

    for _ in range(max_iter):
        m = (a + b) / 2
        fm = safe_eval(f, m)

        if fm is None:
            return None

        if abs(fm) < tol:
            return m

        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm

    return (a + b) / 2


# =============================================================================
# ÉQUATIONS DIFFÉRENTIELLES (NUMÉRIQUE)
# =============================================================================

def solve_ode_euler(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    x_end: float,
    n: int = 1000
) -> Tuple[List[float], List[float]]:
    h = (x_end - x0) / n
    xs = [x0]
    ys = [y0]

    x, y = x0, y0
    for _ in range(n):
        y += h * f(x, y)
        x += h
        xs.append(x)
        ys.append(y)

    return xs, ys


def solve_ode_rk4(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    x_end: float,
    n: int = 1000
) -> Tuple[List[float], List[float]]:
    h = (x_end - x0) / n
    xs = [x0]
    ys = [y0]

    x, y = x0, y0
    for _ in range(n):
        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x + h, y + k3)

        y += (k1 + 2*k2 + 2*k3 + k4) / 6
        x += h

        xs.append(x)
        ys.append(y)

    return xs, ys


# =============================================================================
# ALGÈBRE VECTORIELLE
# =============================================================================

def norm(v: List[float]) -> float:
    return math.sqrt(sum(x*x for x in v))


def normalize(v: List[float]) -> List[float]:
    n = norm(v)
    if n == 0:
        raise ValueError("Zero vector")
    return [x / n for x in v]


def dot(u: List[float], v: List[float]) -> float:
    return sum(a*b for a, b in zip(u, v))


# =============================================================================
# NOMBRES COMPLEXES (SANS CLASSES PYTHON)
# =============================================================================

def complex_module(a: float, b: float) -> float:
    return math.sqrt(a*a + b*b)


def complex_argument(a: float, b: float) -> float:
    return math.atan2(b, a)


def complex_multiply(
    a1: float, b1: float,
    a2: float, b2: float
) -> Tuple[float, float]:
    return (a1*a2 - b1*b2, a1*b2 + a2*b1)


def complex_divide(
    a1: float, b1: float,
    a2: float, b2: float
) -> Tuple[float, float]:
    denom = a2*a2 + b2*b2
    if denom == 0:
        raise ValueError("Division by zero")
    return (
        (a1*a2 + b1*b2) / denom,
        (b1*a2 - a1*b2) / denom
    )


def complex_exp(a: float, b: float) -> Tuple[float, float]:
    r = math.exp(a)
    return (r * math.cos(b), r * math.sin(b))


# =============================================================================
# SUITE: OUTILS NUMÉRIQUES PLUS SÉRIEUX + COMPILATION D'EXPRESSIONS
# (à coller sous le code précédent)
# =============================================================================

import ast
import operator as _op
from typing import Iterable, Mapping, Sequence


# =============================================================================
# EXCEPTIONS
# =============================================================================

class MathCoreError(Exception):
    pass

class DomainError(MathCoreError):
    pass

class ConvergenceError(MathCoreError):
    pass

class ParseError(MathCoreError):
    pass


# =============================================================================
# OUTILS GÉNÉRAUX
# =============================================================================

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def is_finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))

def linspace(a: float, b: float, n: int) -> List[float]:
    return np.linspace(a, b, n).tolist()

def arange(a: float, b: float, step: float) -> List[float]:
    return np.arange(a, b, step).tolist()

def midpoint(a: float, b: float) -> float:
    return (a + b) / 2


# =============================================================================
# STEP-SIZE ADAPTATIF POUR DÉRIVÉES
# =============================================================================

def _recommended_h(x: float, scale: float = 1.0) -> float:
    # h ~ eps^(1/3) * max(1,|x|) (heuristique standard pour différences centrées)
    eps = np.finfo(float).eps
    return scale * (eps ** (1/3)) * max(1.0, abs(x))

def derivative_auto(
    f: Function1D,
    x: float,
    scale: float = 1.0
) -> NumericValue:
    h = _recommended_h(x, scale)
    return derivative(f, x, h=h)

def second_derivative_auto(
    f: Function1D,
    x: float,
    scale: float = 1.0
) -> NumericValue:
    h = _recommended_h(x, scale) ** 0.5  # un peu plus grand pour d2
    return second_derivative(f, x, h=h)


# =============================================================================
# GRADIENT / HESSIEN NUMÉRIQUES
# =============================================================================

def gradient(
    f: Function2D,
    x: float,
    y: float,
    h: float = 1e-6
) -> Tuple[float, float]:
    fx1 = f(x + h, y)
    fx2 = f(x - h, y)
    fy1 = f(x, y + h)
    fy2 = f(x, y - h)
    return ((fx1 - fx2) / (2*h), (fy1 - fy2) / (2*h))

def hessian_2d(
    f: Function2D,
    x: float,
    y: float,
    h: float = 1e-4
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    f00 = f(x, y)
    fxx = (f(x + h, y) - 2*f00 + f(x - h, y)) / (h*h)
    fyy = (f(x, y + h) - 2*f00 + f(x, y - h)) / (h*h)
    fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4*h*h)
    return ((fxx, fxy), (fxy, fyy))


# =============================================================================
# INTÉGRATION: SIMPSON ADAPTATIF (PLUS SÉRIEUX)
# =============================================================================

def _simpson(f: Function1D, a: float, b: float) -> float:
    c = (a + b) / 2
    h = b - a
    return (h / 6) * (f(a) + 4*f(c) + f(b))

def integrate_simpson_adaptive(
    f: Function1D,
    a: float,
    b: float,
    tol: float = 1e-8,
    max_depth: int = 20
) -> NumericValue:
    def rec(a: float, b: float, fa: float, fb: float, fm: float, s: float, depth: int) -> Tuple[float, float]:
        c = (a + b) / 2
        l = (a + c) / 2
        r = (c + b) / 2

        fl = f(l)
        fr = f(r)

        sl = (c - a) / 6 * (fa + 4*fl + fm)
        sr = (b - c) / 6 * (fm + 4*fr + fb)
        s2 = sl + sr

        err = abs(s2 - s) / 15

        if depth <= 0 or err < tol:
            return s2, err

        left_val, left_err = rec(a, c, fa, fm, fl, sl, depth - 1)
        right_val, right_err = rec(c, b, fm, fb, fr, sr, depth - 1)
        return left_val + right_val, left_err + right_err

    fa = f(a)
    fb = f(b)
    m = (a + b) / 2
    fm = f(m)
    s = (b - a) / 6 * (fa + 4*fm + fb)

    val, err = rec(a, b, fa, fb, fm, s, max_depth)
    return NumericValue(val, err)


# =============================================================================
# RACINES: NEWTON / SÉCANTE / HYBRIDE
# =============================================================================

def find_root_newton(
    f: Function1D,
    x0: float,
    df: Optional[Function1D] = None,
    tol: float = 1e-10,
    max_iter: int = 50
) -> float:
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x

        dfx = df(x) if df is not None else derivative_auto(f, x).value
        if dfx == 0 or not is_finite(dfx):
            raise ConvergenceError("Newton: derivative zero/invalid")

        x_new = x - fx / dfx
        if not is_finite(x_new):
            raise ConvergenceError("Newton: non-finite iterate")

        if abs(x_new - x) < tol:
            return x_new
        x = x_new

    raise ConvergenceError("Newton: max_iter reached")

def find_root_secant(
    f: Function1D,
    x0: float,
    x1: float,
    tol: float = 1e-10,
    max_iter: int = 80
) -> float:
    f0 = f(x0)
    f1 = f(x1)

    for _ in range(max_iter):
        if abs(f1) < tol:
            return x1

        denom = (f1 - f0)
        if denom == 0:
            raise ConvergenceError("Secant: zero denominator")

        x2 = x1 - f1 * (x1 - x0) / denom
        if not is_finite(x2):
            raise ConvergenceError("Secant: non-finite iterate")

        if abs(x2 - x1) < tol:
            return x2

        x0, f0 = x1, f1
        x1, f1 = x2, f(x2)

    raise ConvergenceError("Secant: max_iter reached")

def find_root_hybrid(
    f: Function1D,
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 100
) -> float:
    # bisection sûre + accélération type secant si possible
    fa = safe_eval(f, a)
    fb = safe_eval(f, b)
    if fa is None or fb is None or fa * fb > 0:
        raise DomainError("Hybrid root requires bracketing (f(a)f(b) <= 0)")

    x0, x1 = a, b
    f0, f1 = fa, fb

    for _ in range(max_iter):
        # tentative sécante
        if f1 != f0:
            xs = x1 - f1 * (x1 - x0) / (f1 - f0)
        else:
            xs = (x0 + x1) / 2

        # si la sécante sort, on bissecte
        if not (min(x0, x1) <= xs <= max(x0, x1)) or not is_finite(xs):
            xs = (x0 + x1) / 2

        fs = safe_eval(f, xs)
        if fs is None:
            xs = (x0 + x1) / 2
            fs = f(xs)

        if abs(fs) < tol or abs(x1 - x0) < tol:
            return xs

        # maintien du bracket
        if f0 * fs <= 0:
            x1, f1 = xs, fs
        else:
            x0, f0 = xs, fs

    raise ConvergenceError("Hybrid: max_iter reached")


# =============================================================================
# OPTIMISATION 1D: DESCENTE + BRENT-LIKE SIMPLE
# =============================================================================

def minimize_golden_section(
    f: Function1D,
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 200
) -> Tuple[float, float]:
    # suppose f unimodale sur [a,b]
    gr = (math.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            x = (a + b) / 2
            return x, f(x)

        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)

    x = (a + b) / 2
    return x, f(x)

def maximize_golden_section(
    f: Function1D,
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 200
) -> Tuple[float, float]:
    x, y = minimize_golden_section(lambda t: -f(t), a, b, tol=tol, max_iter=max_iter)
    return x, -y

def gradient_descent_1d(
    f: Function1D,
    x0: float,
    lr: float = 1e-2,
    tol: float = 1e-10,
    max_iter: int = 2000
) -> Tuple[float, float]:
    x = x0
    for _ in range(max_iter):
        d = derivative_auto(f, x).value
        x_new = x - lr * d
        if abs(x_new - x) < tol:
            return x_new, f(x_new)
        x = x_new
    raise ConvergenceError("GD 1D: max_iter reached")


# =============================================================================
# DÉTECTION D'ASYMPTOTES (HEURISTIQUE, MAIS PROPRE)
# =============================================================================

@dataclass
class AsymptoteVertical:
    x: float
    evidence: float  # amplitude / divergence

@dataclass
class AsymptoteLine:
    a: float
    b: float

def detect_vertical_asymptotes(
    f: Function1D,
    a: float,
    b: float,
    n: int = 4000,
    jump: float = 1e4
) -> List[AsymptoteVertical]:
    xs = np.linspace(a, b, n)
    ys = []
    for x in xs:
        y = safe_eval(f, float(x))
        ys.append(y)

    out: List[AsymptoteVertical] = []
    for i in range(n - 1):
        y1, y2 = ys[i], ys[i+1]
        if y1 is None or y2 is None:
            out.append(AsymptoteVertical(float(xs[i]), float("inf")))
            continue
        if abs(y2 - y1) > jump or abs(y1) > 1e10 or abs(y2) > 1e10:
            out.append(AsymptoteVertical(float(xs[i]), abs(y2 - y1)))

    # dédoublonnage (regroupement par proximité)
    out_sorted = sorted(out, key=lambda t: t.x)
    merged: List[AsymptoteVertical] = []
    for aym in out_sorted:
        if not merged or abs(aym.x - merged[-1].x) > (b - a) / n * 10:
            merged.append(aym)
    return merged

def detect_horizontal_asymptote(
    f: Function1D,
    direction: Literal["+inf", "-inf"] = "+inf",
    X: Sequence[float] = (1e3, 3e3, 1e4, 3e4, 1e5)
) -> LimitResult:
    vals = []
    for x in X:
        xx = x if direction == "+inf" else -x
        y = safe_eval(f, xx)
        if y is None:
            return LimitResult("divergent", None)
        vals.append(y)

    # convergence simple: variance des derniers points
    tail = vals[-3:]
    if np.std(tail) < 1e-6:
        return LimitResult("finite", float(np.mean(tail)))

    if max(abs(v) for v in tail) > 1e10:
        return LimitResult("infinite", None)

    return LimitResult("divergent", None)

def detect_oblique_asymptote(
    f: Function1D,
    direction: Literal["+inf", "-inf"] = "+inf",
    X: Sequence[float] = (1e3, 3e3, 1e4, 3e4, 1e5)
) -> Optional[AsymptoteLine]:
    # modèle: f(x) ~ a x + b. Estime a,b par régression sur grands x.
    xs = np.array([x if direction == "+inf" else -x for x in X], dtype=float)
    ys = []
    for x in xs:
        y = safe_eval(f, float(x))
        if y is None:
            return None
        ys.append(y)
    ys = np.array(ys, dtype=float)

    # régression y = a x + b
    A = np.vstack([xs, np.ones_like(xs)]).T
    a, b = np.linalg.lstsq(A, ys, rcond=None)[0]

    # validité: résidu relatif faible sur la queue
    pred = a*xs + b
    resid = np.mean(np.abs(pred - ys)) / max(1.0, np.mean(np.abs(ys)))
    if resid < 1e-6 and is_finite(a) and is_finite(b):
        return AsymptoteLine(float(a), float(b))
    return None


# =============================================================================
# INTÉGRALES DOUBLES (RECTANGLE + TRAPÈZES 2D)
# =============================================================================

def integrate_double_rectangles(
    f: Function2D,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int = 200,
    ny: int = 200
) -> float:
    hx = (x_max - x_min) / nx
    hy = (y_max - y_min) / ny
    s = 0.0
    for i in range(nx):
        x = x_min + (i + 0.5) * hx
        for j in range(ny):
            y = y_min + (j + 0.5) * hy
            s += f(x, y)
    return s * hx * hy

def integrate_double_trapezoidal(
    f: Function2D,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int = 200,
    ny: int = 200
) -> float:
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    hx = (x_max - x_min) / (nx - 1)
    hy = (y_max - y_min) / (ny - 1)

    s = 0.0
    for i, x in enumerate(xs):
        wx = 0.5 if i in (0, nx - 1) else 1.0
        for j, y in enumerate(ys):
            wy = 0.5 if j in (0, ny - 1) else 1.0
            s += wx * wy * f(float(x), float(y))
    return s * hx * hy


# =============================================================================
# ANALYSE : DÉRIVÉES D'ORDRE N (N ≤ 10, NUMÉRIQUE CONTRÔLÉ)
# =============================================================================

MAX_DERIVATIVE_ORDER = 10

def derivative_n(
    f: Function1D,
    x: float,
    n: int,
    h: float = 1e-4
) -> NumericValue:
    """
    Dérivée n-ième numérique (différences finies centrées)
    ATTENTION : n ≤ 10 uniquement (au-delà instable numériquement)
    """
    if n < 0:
        raise ValueError("n must be ≥ 0")
    if n > MAX_DERIVATIVE_ORDER:
        raise ValueError(f"n must be ≤ {MAX_DERIVATIVE_ORDER}")

    if n == 0:
        return NumericValue(f(x))

    # coefficients binomiaux centrés
    coeffs = {
        1: [(-1, -1), (1, 1)],
        2: [(1, -1), (-2, 0), (1, 1)],
        3: [(-1, -2), (3, -1), (-3, 1), (1, 2)],
        4: [(1, -2), (-4, -1), (6, 0), (-4, 1), (1, 2)],
    }

    if n <= 4:
        c = coeffs[n]
        s = 0.0
        for coef, shift in c:
            fx = safe_eval(f, x + shift * h)
            if fx is None:
                raise DomainError("Derivative undefined")
            s += coef * fx
        return NumericValue(s / (h ** n))

    # n > 4 : dérivation récursive contrôlée
    def g(t):
        return derivative_n(f, t, n - 1, h).value

    return derivative_n(g, x, 1, h)


# =============================================================================
# ANALYSE : DÉVELOPPEMENTS LIMITÉS (TAYLOR)
# =============================================================================

def taylor_coefficients(
    f: Function1D,
    x0: float,
    order: int
) -> List[float]:
    """
    Coefficients a_k du développement de Taylor :
    f(x) ≈ Σ a_k (x - x0)^k
    """
    if order > MAX_DERIVATIVE_ORDER:
        raise ValueError("Order too large")

    coeffs = []
    for k in range(order + 1):
        dk = derivative_n(f, x0, k).value
        coeffs.append(dk / math.factorial(k))
    return coeffs


def taylor_eval(
    coeffs: List[float],
    x: float,
    x0: float
) -> float:
    """
    Évalue un polynôme de Taylor donné par ses coefficients
    """
    dx = x - x0
    s = 0.0
    for k, a in enumerate(coeffs):
        s += a * (dx ** k)
    return s


# =============================================================================
# ANALYSE : SÉRIES (FINIES ET INFINIES)
# =============================================================================

def partial_sum(
    term: Callable[[int], float],
    n: int
) -> float:
    """
    Somme partielle Σ_{k=0}^{n} u_k
    """
    return sum(term(k) for k in range(n + 1))


def series_converges_comparison(
    term: Callable[[int], float],
    ref: Callable[[int], float],
    n: int = 10000
) -> bool:
    """
    Test de convergence par comparaison numérique
    """
    for k in range(1, n):
        if abs(term(k)) > abs(ref(k)):
            return False
    return True


# =============================================================================
# SÉRIES ENTIÈRES : RAYON DE CONVERGENCE
# =============================================================================

def radius_of_convergence_dalembert(
    coeff: Callable[[int], float],
    n: int = 1000
) -> Optional[float]:
    """
    Rayon de convergence via critère de d'Alembert :
    R = lim |a_n / a_{n+1}|
    """
    ratios = []
    for k in range(1, n):
        a_n = coeff(k)
        a_np1 = coeff(k + 1)
        if a_np1 == 0:
            continue
        ratios.append(abs(a_n / a_np1))

    if not ratios:
        return None

    # stabilisation sur la queue
    tail = ratios[-20:]
    if np.std(tail) < 1e-6:
        return float(np.mean(tail))
    return None


def radius_of_convergence_cauchy(
    coeff: Callable[[int], float],
    n: int = 1000
) -> Optional[float]:
    """
    Rayon de convergence via critère de Cauchy :
    R = 1 / limsup |a_n|^{1/n}
    """
    values = []
    for k in range(1, n):
        a = abs(coeff(k))
        if a <= 0:
            continue
        values.append(a ** (1 / k))

    if not values:
        return None

    limsup = max(values[-20:])
    if limsup == 0:
        return float("inf")
    return 1 / limsup


def evaluate_power_series(
    coeff: Callable[[int], float],
    x: float,
    n: int = 50
) -> float:
    """
    Évalue Σ a_n x^n jusqu'à n
    """
    s = 0.0
    for k in range(n):
        s += coeff(k) * (x ** k)
    return s


# =============================================================================
# INTÉGRALES IMPROPRES
# =============================================================================

def improper_integral_infinite(
    f: Function1D,
    a: float,
    direction: Literal["+inf", "-inf"] = "+inf",
    L: float = 1e3
) -> NumericValue:
    """
    Approxime ∫_a^{+∞} f(x) dx ou ∫_{-∞}^a f(x) dx
    """
    if direction == "+inf":
        val = integrate_simpson_adaptive(f, a, L)
    else:
        val = integrate_simpson_adaptive(f, -L, a)
    return val


def improper_integral_singularity(
    f: Function1D,
    a: float,
    b: float,
    singular: float,
    eps: float = 1e-6
) -> NumericValue:
    """
    ∫_a^b f(x) dx avec singularité en singular
    """
    left = integrate_simpson_adaptive(f, a, singular - eps)
    right = integrate_simpson_adaptive(f, singular + eps, b)
    return NumericValue(left.value + right.value,
                        (left.error or 0) + (right.error or 0))


# =============================================================================
# CONVEXITÉ / CONCAVITÉ
# =============================================================================

def is_convex(
    f: Function1D,
    a: float,
    b: float,
    n: int = 1000
) -> bool:
    xs = np.linspace(a, b, n)
    for x in xs:
        d2 = derivative_n(f, float(x), 2).value
        if d2 < -1e-6:
            return False
    return True


def is_concave(
    f: Function1D,
    a: float,
    b: float,
    n: int = 1000
) -> bool:
    xs = np.linspace(a, b, n)
    for x in xs:
        d2 = derivative_n(f, float(x), 2).value
        if d2 > 1e-6:
            return False
    return True


# =============================================================================
# MONOTONIE
# =============================================================================

def is_increasing(
    f: Function1D,
    a: float,
    b: float,
    n: int = 1000
) -> bool:
    xs = np.linspace(a, b, n)
    for x in xs:
        d = derivative_auto(f, float(x)).value
        if d < -1e-6:
            return False
    return True


def is_decreasing(
    f: Function1D,
    a: float,
    b: float,
    n: int = 1000
) -> bool:
    xs = np.linspace(a, b, n)
    for x in xs:
        d = derivative_auto(f, float(x)).value
        if d > 1e-6:
            return False
    return True


# =============================================================================
# TESTS CLASSIQUES DE SÉRIES NUMÉRIQUES
# =============================================================================

def series_test_geometric(u0: float, q: float) -> bool:
    """
    Série géométrique Σ u0 q^n converge ⇔ |q| < 1
    """
    return abs(q) < 1


def series_test_harmonic(p: float) -> bool:
    """
    Série de Riemann Σ 1/n^p converge ⇔ p > 1
    """
    return p > 1


def alternating_series_test(
    term: Callable[[int], float],
    n: int = 10000
) -> bool:
    """
    Test de Leibniz (numérique)
    """
    prev = abs(term(0))
    for k in range(1, n):
        curr = abs(term(k))
        if curr > prev:
            return False
        prev = curr
    return True


# =============================================================================
# TESTS DE MAJORATION / MINORATION (SÉRIES)
# =============================================================================

def is_eventually_positive(
    term: Callable[[int], float],
    n0: int = 10,
    n: int = 1000
) -> bool:
    """
    Vérifie si u_n >= 0 à partir d'un certain rang (numériquement)
    """
    for k in range(n0, n):
        if term(k) < 0:
            return False
    return True


def is_eventually_bounded_by(
    term: Callable[[int], float],
    bound: Callable[[int], float],
    n0: int = 10,
    n: int = 1000
) -> bool:
    """
    Vérifie si |u_n| <= v_n à partir d'un certain rang
    """
    for k in range(n0, n):
        if abs(term(k)) > abs(bound(k)):
            return False
    return True


# =============================================================================
# TEST DE COMPARAISON (DIRECT)
# =============================================================================

def series_test_comparison(
    term: Callable[[int], float],
    ref: Callable[[int], float],
    ref_converges: bool,
    n0: int = 10,
    n: int = 5000
) -> Optional[bool]:
    """
    Test de comparaison :
    - si 0 ≤ u_n ≤ v_n et Σ v_n converge ⇒ Σ u_n converge
    - si u_n ≥ v_n ≥ 0 et Σ v_n diverge ⇒ Σ u_n diverge
    """
    for k in range(n0, n):
        if term(k) < 0 or ref(k) < 0:
            return None
        if term(k) > ref(k):
            return None

    return ref_converges


# =============================================================================
# TEST DE COMPARAISON PAR ÉQUIVALENCE
# =============================================================================

def series_test_equivalence(
    term: Callable[[int], float],
    ref: Callable[[int], float],
    n: int = 5000
) -> Optional[float]:
    """
    Test par équivalent :
    retourne L = lim u_n / v_n si elle existe
    """
    ratios = []
    for k in range(10, n):
        v = ref(k)
        if v == 0:
            continue
        ratios.append(term(k) / v)

    if not ratios:
        return None

    tail = ratios[-50:]
    if np.std(tail) < 1e-6:
        return float(np.mean(tail))
    return None


# =============================================================================
# TEST DE CONDENSATION (CAUCHY)
# =============================================================================

def series_test_condensation(
    term: Callable[[int], float],
    n: int = 20
) -> bool:
    """
    Test de condensation :
    Σ u_n ~ Σ 2^k u_{2^k}
    Hypothèse : u_n décroissante, positive
    """
    s = 0.0
    for k in range(1, n):
        s += (2 ** k) * term(2 ** k)
    return is_finite(s)


# =============================================================================
# TEST INTÉGRAL
# =============================================================================

def series_test_integral(
    f: Function1D,
    a: float = 1.0,
    L: float = 1e3
) -> bool:
    """
    Test intégral :
    Σ f(n) converge ⇔ ∫_a^∞ f(x) dx converge
    """
    try:
        val = improper_integral_infinite(f, a)
        return is_finite(val.value)
    except Exception:
        return False


# =============================================================================
# TEST DE LEIBNIZ (ALTERNÉ)
# =============================================================================

def series_test_leibniz(
    term: Callable[[int], float],
    n: int = 5000
) -> bool:
    """
    Série alternée :
    Σ (-1)^n a_n converge si a_n ↓ 0
    """
    prev = abs(term(0))
    for k in range(1, n):
        curr = abs(term(k))
        if curr > prev + 1e-12:
            return False
        prev = curr
    return abs(term(n - 1)) < 1e-6


# =============================================================================
# TEST DE DIRICHLET
# =============================================================================

def series_test_dirichlet(
    a: Callable[[int], float],
    b: Callable[[int], float],
    n: int = 5000
) -> bool:
    """
    Test de Dirichlet :
    - sommes partielles de a_n bornées
    - b_n décroissante vers 0
    """
    # sommes partielles de a_n
    S = 0.0
    S_max = 0.0
    for k in range(1, n):
        S += a(k)
        S_max = max(S_max, abs(S))

    if S_max > 1e6:
        return False

    prev = abs(b(1))
    for k in range(2, n):
        curr = abs(b(k))
        if curr > prev + 1e-12:
            return False
        prev = curr

    return prev < 1e-6


# =============================================================================
# TEST D'ABEL
# =============================================================================

def series_test_abel(
    a: Callable[[int], float],
    b: Callable[[int], float],
    n: int = 5000
) -> bool:
    """
    Test d'Abel :
    - Σ a_n converge
    - b_n monotone bornée
    """
    # convergence de Σ a_n (numérique)
    s = partial_sum(a, n)
    if not is_finite(s):
        return False

    prev = b(0)
    for k in range(1, n):
        curr = b(k)
        if curr > prev + 1e-12:
            return False
        prev = curr

    return True


# =============================================================================
# MAJORATION DU RESTE D'UNE SÉRIE
# =============================================================================

def remainder_bound_geometric(
    u0: float,
    q: float,
    n: int
) -> Optional[float]:
    """
    Majorant du reste d'une série géométrique :
    R_n ≤ u0 q^{n+1} / (1 - q)
    """
    if abs(q) >= 1:
        return None
    return abs(u0) * (abs(q) ** (n + 1)) / (1 - abs(q))


def remainder_bound_alternating(
    term: Callable[[int], float],
    n: int
) -> float:
    """
    Majorant du reste d'une série alternée convergente :
    |R_n| ≤ |u_{n+1}|
    """
    return abs(term(n + 1))


# =============================================================================
# MAJORATION / MINORATION DE FONCTIONS
# =============================================================================

def is_majorant_of(
    g: Function1D,
    f: Function1D,
    a: float,
    b: float,
    n: int = 1000
) -> bool:
    """
    Vérifie si g(x) ≥ f(x) sur [a,b]
    """
    xs = np.linspace(a, b, n)
    for x in xs:
        fx = safe_eval(f, float(x))
        gx = safe_eval(g, float(x))
        if fx is None or gx is None:
            return False
        if gx < fx:
            return False
    return True


def is_minorant_of(
    g: Function1D,
    f: Function1D,
    a: float,
    b: float,
    n: int = 1000
) -> bool:
    """
    Vérifie si g(x) ≤ f(x) sur [a,b]
    """
    xs = np.linspace(a, b, n)
    for x in xs:
        fx = safe_eval(f, float(x))
        gx = safe_eval(g, float(x))
        if fx is None or gx is None:
            return False
        if gx > fx:
            return False
    return True


# =============================================================================
# ENCADDREMENT (UTILISABLE POUR LIMITES / INTÉGRALES)
# =============================================================================

def sandwich_limit(
    f: Function1D,
    g: Function1D,
    h: Function1D,
    x0: float,
    eps: float = 1e-6
) -> bool:
    """
    Test du théorème des gendarmes (numérique) :
    g(x) ≤ f(x) ≤ h(x) et lim g = lim h
    """
    lg = numeric_limit(g, x0)
    lh = numeric_limit(h, x0)
    lf = numeric_limit(f, x0)

    if lg.kind == lh.kind == "finite":
        if abs(lg.value - lh.value) < eps:
            return lf.kind == "finite" and abs(lf.value - lg.value) < eps
    return False


# =============================================================================
# FIN DES TESTS D'ANALYSE
# =============================================================================
