"""
trig_formulas.py
Formules trigonométriques (réel + hyperbolique) : identités, transformations,
dérivées, primitives, DL, produits/sommes, angles multiples, méthodes.
Objectif : base complète pour moteur de calcul (site) + post-traitement OCR.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, List, Literal

Number = float


# =============================================================================
# CONSTANTES / CONVENTIONS
# =============================================================================

PI = math.pi
TAU = 2 * math.pi

def deg2rad(deg: float) -> float:
    return deg * PI / 180.0

def rad2deg(rad: float) -> float:
    return rad * 180.0 / PI

def wrap_angle_pi(x: float) -> float:
    """
    Ramène un angle dans (-pi, pi]
    """
    y = (x + PI) % (2 * PI) - PI
    return PI if y == -PI else y

def wrap_angle_2pi(x: float) -> float:
    """
    Ramène un angle dans [0, 2pi)
    """
    return x % (2 * PI)

def safe_acos(x: float) -> float:
    return math.acos(max(-1.0, min(1.0, x)))

def safe_asin(x: float) -> float:
    return math.asin(max(-1.0, min(1.0, x)))


# =============================================================================
# FONCTIONS TRIGONOMÉTRIQUES + INVERSes (wrappers)
# =============================================================================

def sin(x: float) -> float: return math.sin(x)
def cos(x: float) -> float: return math.cos(x)
def tan(x: float) -> float: return math.tan(x)

def asin(x: float) -> float: return safe_asin(x)
def acos(x: float) -> float: return safe_acos(x)
def atan(x: float) -> float: return math.atan(x)
def atan2(y: float, x: float) -> float: return math.atan2(y, x)

def cot(x: float) -> float:
    s = math.sin(x)
    if s == 0:
        raise ZeroDivisionError("cot undefined when sin(x)=0")
    return math.cos(x) / s

def sec(x: float) -> float:
    c = math.cos(x)
    if c == 0:
        raise ZeroDivisionError("sec undefined when cos(x)=0")
    return 1.0 / c

def csc(x: float) -> float:
    s = math.sin(x)
    if s == 0:
        raise ZeroDivisionError("csc undefined when sin(x)=0")
    return 1.0 / s


# =============================================================================
# HYPERBOLIQUES
# =============================================================================

def sinh(x: float) -> float: return math.sinh(x)
def cosh(x: float) -> float: return math.cosh(x)
def tanh(x: float) -> float: return math.tanh(x)

def asinh(x: float) -> float: return math.asinh(x)
def acosh(x: float) -> float:
    if x < 1:
        raise ValueError("acosh defined for x>=1")
    return math.acosh(x)
def atanh(x: float) -> float:
    if not (-1 < x < 1):
        raise ValueError("atanh defined for -1<x<1")
    return math.atanh(x)

def sech(x: float) -> float: return 1.0 / math.cosh(x)
def csch(x: float) -> float:
    s = math.sinh(x)
    if s == 0:
        raise ZeroDivisionError("csch undefined when sinh(x)=0")
    return 1.0 / s
def coth(x: float) -> float:
    s = math.sinh(x)
    if s == 0:
        raise ZeroDivisionError("coth undefined when sinh(x)=0")
    return math.cosh(x) / s


# =============================================================================
# IDENTITÉS FONDAMENTALES (TRIG)
# =============================================================================

def identity_pythagorean(x: float) -> float:
    """
    sin^2 x + cos^2 x = 1  -> retourne l'erreur numérique
    """
    return (math.sin(x)**2 + math.cos(x)**2) - 1.0

def identity_tan(x: float) -> float:
    """
    1 + tan^2 x = 1/cos^2 x (pour cos x != 0)
    """
    c = math.cos(x)
    if c == 0:
        raise ZeroDivisionError("cos(x)=0")
    return (1.0 + math.tan(x)**2) - (1.0 / (c*c))

def identity_cot(x: float) -> float:
    """
    1 + cot^2 x = 1/sin^2 x (pour sin x != 0)
    """
    s = math.sin(x)
    if s == 0:
        raise ZeroDivisionError("sin(x)=0")
    return (1.0 + cot(x)**2) - (1.0 / (s*s))


# =============================================================================
# IDENTITÉS FONDAMENTALES (HYPERBOLIQUES)
# =============================================================================

def identity_hyperbolic(x: float) -> float:
    """
    cosh^2 x - sinh^2 x = 1 -> retourne l'erreur numérique
    """
    return (math.cosh(x)**2 - math.sinh(x)**2) - 1.0

def identity_tanh(x: float) -> float:
    """
    1 - tanh^2 x = sech^2 x
    """
    return (1.0 - math.tanh(x)**2) - (sech(x)**2)


# =============================================================================
# PARITÉ / PÉRIODICITÉ
# =============================================================================

def trig_parity_checks(x: float) -> Dict[str, float]:
    """
    sin(-x)=-sin x ; cos(-x)=cos x ; tan(-x)=-tan x
    retourne les écarts
    """
    return {
        "sin": math.sin(-x) + math.sin(x),
        "cos": math.cos(-x) - math.cos(x),
        "tan": math.tan(-x) + math.tan(x),
    }

def periodize_sin_cos(x: float, k: int) -> Dict[str, float]:
    """
    sin(x+2kpi)=sin x ; cos(x+2kpi)=cos x ; tan(x+kpi)=tan x
    retourne les écarts
    """
    return {
        "sin": math.sin(x + 2*k*PI) - math.sin(x),
        "cos": math.cos(x + 2*k*PI) - math.cos(x),
        "tan": math.tan(x + k*PI) - math.tan(x),
    }


# =============================================================================
# FORMULES D'ADDITION / SOUSTRACTION
# =============================================================================

def sin_add(a: float, b: float) -> float:
    return math.sin(a)*math.cos(b) + math.cos(a)*math.sin(b)

def sin_sub(a: float, b: float) -> float:
    return math.sin(a)*math.cos(b) - math.cos(a)*math.sin(b)

def cos_add(a: float, b: float) -> float:
    return math.cos(a)*math.cos(b) - math.sin(a)*math.sin(b)

def cos_sub(a: float, b: float) -> float:
    return math.cos(a)*math.cos(b) + math.sin(a)*math.sin(b)

def tan_add(a: float, b: float) -> float:
    denom = 1.0 - math.tan(a)*math.tan(b)
    if denom == 0:
        raise ZeroDivisionError("tan(a+b) undefined")
    return (math.tan(a) + math.tan(b)) / denom

def tan_sub(a: float, b: float) -> float:
    denom = 1.0 + math.tan(a)*math.tan(b)
    if denom == 0:
        raise ZeroDivisionError("tan(a-b) undefined")
    return (math.tan(a) - math.tan(b)) / denom


# =============================================================================
# DOUBLES / TRIPLES ANGLES
# =============================================================================

def sin_double(x: float) -> float:
    return 2.0 * math.sin(x) * math.cos(x)

def cos_double(x: float) -> float:
    return math.cos(x)**2 - math.sin(x)**2

def cos_double_alt1(x: float) -> float:
    return 2.0 * math.cos(x)**2 - 1.0

def cos_double_alt2(x: float) -> float:
    return 1.0 - 2.0 * math.sin(x)**2

def tan_double(x: float) -> float:
    denom = 1.0 - math.tan(x)**2
    if denom == 0:
        raise ZeroDivisionError("tan(2x) undefined")
    return 2.0 * math.tan(x) / denom

def sin_triple(x: float) -> float:
    s = math.sin(x)
    return 3.0*s - 4.0*s**3

def cos_triple(x: float) -> float:
    c = math.cos(x)
    return 4.0*c**3 - 3.0*c

def tan_triple(x: float) -> float:
    t = math.tan(x)
    denom = 1.0 - 3.0*t*t
    if denom == 0:
        raise ZeroDivisionError("tan(3x) undefined")
    return (3.0*t - t**3) / denom


# =============================================================================
# PUISSANCES -> SOMMES (réduction)
# =============================================================================

def sin2_to_cos2x(x: float) -> float:
    """
    sin^2 x = (1 - cos 2x)/2
    """
    return (1.0 - math.cos(2*x)) / 2.0

def cos2_to_cos2x(x: float) -> float:
    """
    cos^2 x = (1 + cos 2x)/2
    """
    return (1.0 + math.cos(2*x)) / 2.0

def sincos_to_sin2x(x: float) -> float:
    """
    sin x cos x = sin 2x / 2
    """
    return math.sin(2*x) / 2.0


def sin3_reduction(x: float) -> float:
    """
    sin^3 x = (3 sin x - sin 3x)/4
    """
    return (3*math.sin(x) - math.sin(3*x)) / 4.0

def cos3_reduction(x: float) -> float:
    """
    cos^3 x = (3 cos x + cos 3x)/4
    """
    return (3*math.cos(x) + math.cos(3*x)) / 4.0

def sin4_reduction(x: float) -> float:
    """
    sin^4 x = (3 - 4 cos2x + cos4x)/8
    """
    return (3.0 - 4.0*math.cos(2*x) + math.cos(4*x)) / 8.0

def cos4_reduction(x: float) -> float:
    """
    cos^4 x = (3 + 4 cos2x + cos4x)/8
    """
    return (3.0 + 4.0*math.cos(2*x) + math.cos(4*x)) / 8.0


# =============================================================================
# PRODUIT -> SOMME
# =============================================================================

def sin_sin_to_sum(a: float, b: float) -> Tuple[float, float]:
    """
    sin a sin b = 1/2 [cos(a-b) - cos(a+b)]
    retourne (term1, term2) tels que produit = 0.5*(term1 + term2)
    """
    return (math.cos(a - b), -math.cos(a + b))

def cos_cos_to_sum(a: float, b: float) -> Tuple[float, float]:
    """
    cos a cos b = 1/2 [cos(a-b) + cos(a+b)]
    """
    return (math.cos(a - b), math.cos(a + b))

def sin_cos_to_sum(a: float, b: float) -> Tuple[float, float]:
    """
    sin a cos b = 1/2 [sin(a+b) + sin(a-b)]
    """
    return (math.sin(a + b), math.sin(a - b))


# =============================================================================
# SOMME -> PRODUIT
# =============================================================================

def sin_plus_sin(a: float, b: float) -> Tuple[float, float]:
    """
    sin a + sin b = 2 sin((a+b)/2) cos((a-b)/2)
    retourne (2 sin(...), cos(...)) pour recomposition
    """
    return (2.0 * math.sin((a + b) / 2.0), math.cos((a - b) / 2.0))

def sin_minus_sin(a: float, b: float) -> Tuple[float, float]:
    """
    sin a - sin b = 2 cos((a+b)/2) sin((a-b)/2)
    """
    return (2.0 * math.cos((a + b) / 2.0), math.sin((a - b) / 2.0))

def cos_plus_cos(a: float, b: float) -> Tuple[float, float]:
    """
    cos a + cos b = 2 cos((a+b)/2) cos((a-b)/2)
    """
    return (2.0 * math.cos((a + b) / 2.0), math.cos((a - b) / 2.0))

def cos_minus_cos(a: float, b: float) -> Tuple[float, float]:
    """
    cos a - cos b = -2 sin((a+b)/2) sin((a-b)/2)
    """
    return (-2.0 * math.sin((a + b) / 2.0), math.sin((a - b) / 2.0))


# =============================================================================
# FORMULES D'ANGLE MOITIÉ / TANGENTE DEMI-ANGLE
# =============================================================================

def sin_half_from_cos(x: float, sign: int = 1) -> float:
    """
    sin(x/2) = ± sqrt((1 - cos x)/2)
    sign = +1 ou -1 selon quadrant
    """
    if sign not in (-1, 1):
        raise ValueError("sign must be ±1")
    return sign * math.sqrt(max(0.0, (1.0 - math.cos(x)) / 2.0))

def cos_half_from_cos(x: float, sign: int = 1) -> float:
    """
    cos(x/2) = ± sqrt((1 + cos x)/2)
    """
    if sign not in (-1, 1):
        raise ValueError("sign must be ±1")
    return sign * math.sqrt(max(0.0, (1.0 + math.cos(x)) / 2.0))

def tan_half_from_sin_cos(x: float) -> float:
    """
    tan(x/2) = sin x / (1 + cos x) (si 1+cos x != 0)
    """
    denom = 1.0 + math.cos(x)
    if denom == 0:
        raise ZeroDivisionError("tan(x/2) undefined when 1+cos(x)=0")
    return math.sin(x) / denom

def tan_half_alt(x: float) -> float:
    """
    tan(x/2) = (1 - cos x) / sin x (si sin x != 0)
    """
    s = math.sin(x)
    if s == 0:
        raise ZeroDivisionError("tan(x/2) undefined when sin(x)=0")
    return (1.0 - math.cos(x)) / s


# =============================================================================
# FORMULES D'EULER (COMPLEXES) - EN RÉEL VIA COS/SIN
# =============================================================================

def euler_cos(x: float) -> Tuple[float, float]:
    """
    e^{ix} = cos x + i sin x
    retourne (Re, Im)
    """
    return (math.cos(x), math.sin(x))

def cos_from_euler(x: float) -> Tuple[float, float]:
    """
    cos x = (e^{ix} + e^{-ix})/2
    retourne (e^{ix}, e^{-ix}) en (Re,Im) non stocké ici
    """
    return (math.cos(x), math.cos(x))  # placeholder structure for symbolic use

def sin_from_euler(x: float) -> Tuple[float, float]:
    """
    sin x = (e^{ix} - e^{-ix})/(2i)
    """
    return (math.sin(x), -math.sin(x))  # placeholder structure for symbolic use


# =============================================================================
# IDENTITÉS AVEC HYPERBOLIQUES
# =============================================================================

def sinh_def(x: float) -> float:
    """
    sinh x = (e^x - e^{-x})/2
    """
    return 0.5 * (math.exp(x) - math.exp(-x))

def cosh_def(x: float) -> float:
    """
    cosh x = (e^x + e^{-x})/2
    """
    return 0.5 * (math.exp(x) + math.exp(-x))

def tanh_def(x: float) -> float:
    """
    tanh x = sinh x / cosh x
    """
    return math.tanh(x)


# =============================================================================
# ADDITION HYPERBOLIQUE
# =============================================================================

def sinh_add(a: float, b: float) -> float:
    return math.sinh(a)*math.cosh(b) + math.cosh(a)*math.sinh(b)

def cosh_add(a: float, b: float) -> float:
    return math.cosh(a)*math.cosh(b) + math.sinh(a)*math.sinh(b)

def tanh_add(a: float, b: float) -> float:
    denom = 1.0 + math.tanh(a)*math.tanh(b)
    if denom == 0:
        raise ZeroDivisionError("tanh(a+b) undefined")
    return (math.tanh(a) + math.tanh(b)) / denom


# =============================================================================
# DOUBLES ANGLES HYPERBOLIQUES
# =============================================================================

def sinh_double(x: float) -> float:
    return 2.0 * math.sinh(x) * math.cosh(x)

def cosh_double(x: float) -> float:
    return math.cosh(x)**2 + math.sinh(x)**2

def tanh_double(x: float) -> float:
    denom = 1.0 + math.tanh(x)**2
    return 2.0 * math.tanh(x) / denom


# =============================================================================
# INÉGALITÉS CLASSIQUES (ANALYSE)
# =============================================================================

def inequality_sin_x_le_x(x: float) -> bool:
    """
    Pour x >= 0 : sin x <= x
    """
    if x < 0:
        raise ValueError("Use x>=0")
    return math.sin(x) <= x + 1e-12

def inequality_x_le_tan_x(x: float) -> bool:
    """
    Pour x in (0, pi/2) : x <= tan x
    """
    if not (0 < x < PI/2):
        raise ValueError("Use x in (0, pi/2)")
    return x <= math.tan(x) + 1e-12

def inequality_cos_x_ge_1_minus_x2_over2(x: float) -> bool:
    """
    cos x >= 1 - x^2/2 pour tout réel
    """
    return math.cos(x) >= 1.0 - x*x/2.0 - 1e-12

def inequality_sin_x_ge_2x_over_pi(x: float) -> bool:
    """
    Pour x in [0, pi/2] : sin x >= 2x/pi
    """
    if not (0 <= x <= PI/2):
        raise ValueError("Use x in [0, pi/2]")
    return math.sin(x) + 1e-12 >= 2.0*x/PI


# =============================================================================
# ÉQUATIONS TRIG : RÉDUCTION / RÉSOLUTION
# =============================================================================

@dataclass
class TrigSolutionSet:
    description: str
    base: Optional[float] = None
    period: Optional[float] = None
    extra: Optional[Tuple[float, float]] = None

def solve_sin_eq_0() -> TrigSolutionSet:
    """
    sin x = 0 -> x = k pi
    """
    return TrigSolutionSet("x = k*pi", base=0.0, period=PI)

def solve_cos_eq_0() -> TrigSolutionSet:
    """
    cos x = 0 -> x = pi/2 + k pi
    """
    return TrigSolutionSet("x = pi/2 + k*pi", base=PI/2, period=PI)

def solve_tan_eq_0() -> TrigSolutionSet:
    """
    tan x = 0 -> x = k pi
    """
    return TrigSolutionSet("x = k*pi", base=0.0, period=PI)

def solve_sin_eq_a(a: float) -> TrigSolutionSet:
    """
    sin x = a -> solutions:
    x = arcsin(a) + 2kpi OR x = (pi - arcsin(a)) + 2kpi
    """
    if abs(a) > 1:
        raise ValueError("|a| must be ≤ 1")
    alpha = safe_asin(a)
    return TrigSolutionSet("x = alpha + 2kpi OR x = (pi-alpha) + 2kpi", extra=(alpha, PI - alpha), period=TAU)

def solve_cos_eq_a(a: float) -> TrigSolutionSet:
    """
    cos x = a -> solutions:
    x = ± arccos(a) + 2kpi
    """
    if abs(a) > 1:
        raise ValueError("|a| must be ≤ 1")
    alpha = safe_acos(a)
    return TrigSolutionSet("x = ±alpha + 2kpi", extra=(alpha, -alpha), period=TAU)

def solve_tan_eq_a(a: float) -> TrigSolutionSet:
    """
    tan x = a -> x = arctan(a) + k pi
    """
    alpha = math.atan(a)
    return TrigSolutionSet("x = alpha + k*pi", base=alpha, period=PI)


# =============================================================================
# DÉRIVÉES (FORMULES)
# =============================================================================

def d_sin(x: float) -> float: return math.cos(x)
def d_cos(x: float) -> float: return -math.sin(x)
def d_tan(x: float) -> float:
    c = math.cos(x)
    if c == 0:
        raise ZeroDivisionError("tan' undefined when cos(x)=0")
    return 1.0 / (c*c)

def d_cot(x: float) -> float:
    s = math.sin(x)
    if s == 0:
        raise ZeroDivisionError("cot' undefined when sin(x)=0")
    return -1.0 / (s*s)

def d_sec(x: float) -> float:
    return sec(x) * math.tan(x)

def d_csc(x: float) -> float:
    return -csc(x) * cot(x)

def d_asin(x: float) -> float:
    if abs(x) >= 1:
        raise ValueError("asin' undefined for |x|>=1")
    return 1.0 / math.sqrt(1.0 - x*x)

def d_acos(x: float) -> float:
    if abs(x) >= 1:
        raise ValueError("acos' undefined for |x|>=1")
    return -1.0 / math.sqrt(1.0 - x*x)

def d_atan(x: float) -> float:
    return 1.0 / (1.0 + x*x)

def d_sinh(x: float) -> float: return math.cosh(x)
def d_cosh(x: float) -> float: return math.sinh(x)
def d_tanh(x: float) -> float: return sech(x)**2
def d_asinh(x: float) -> float: return 1.0 / math.sqrt(1.0 + x*x)
def d_acosh(x: float) -> float:
    if x <= 1:
        raise ValueError("acosh' undefined for x<=1")
    return 1.0 / (math.sqrt(x - 1.0) * math.sqrt(x + 1.0))
def d_atanh(x: float) -> float:
    if not (-1 < x < 1):
        raise ValueError("atanh' undefined for |x|>=1")
    return 1.0 / (1.0 - x*x)


# =============================================================================
# PRIMITIVES (FORMULES)
# =============================================================================

def int_sin() -> str: return "∫ sin x dx = -cos x + C"
def int_cos() -> str: return "∫ cos x dx = sin x + C"
def int_tan() -> str: return "∫ tan x dx = -ln|cos x| + C = ln|sec x| + C"
def int_cot() -> str: return "∫ cot x dx = ln|sin x| + C"
def int_sec2() -> str: return "∫ sec^2 x dx = tan x + C"
def int_csc2() -> str: return "∫ csc^2 x dx = -cot x + C"
def int_sec_tan() -> str: return "∫ sec x tan x dx = sec x + C"
def int_csc_cot() -> str: return "∫ csc x cot x dx = -csc x + C"

def int_sinh() -> str: return "∫ sinh x dx = cosh x + C"
def int_cosh() -> str: return "∫ cosh x dx = sinh x + C"
def int_sech2() -> str: return "∫ sech^2 x dx = tanh x + C"
def int_csch2() -> str: return "∫ csch^2 x dx = -coth x + C"
def int_sech_tanh() -> str: return "∫ sech x tanh x dx = -sech x + C"


# =============================================================================
# DÉVELOPPEMENTS LIMITÉS (DL) AUTOUR DE 0
# =============================================================================

def dl_sin(x: float, order: int = 9) -> float:
    """
    sin x = x - x^3/3! + x^5/5! - ...
    order impair recommandé (<= 15)
    """
    if order < 1:
        return 0.0
    s = 0.0
    sign = 1.0
    for k in range(0, (order//2)+1):
        n = 2*k + 1
        term = (x**n) / math.factorial(n)
        s += sign * term
        sign *= -1.0
    return s

def dl_cos(x: float, order: int = 8) -> float:
    """
    cos x = 1 - x^2/2! + x^4/4! - ...
    """
    s = 0.0
    sign = 1.0
    for k in range(0, (order//2)+1):
        n = 2*k
        term = (x**n) / math.factorial(n)
        s += sign * term
        sign *= -1.0
    return s

def dl_tan(x: float, order: int = 7) -> float:
    """
    tan x = x + x^3/3 + 2x^5/15 + 17x^7/315 + ...
    (troncature fixe utile en calcul)
    """
    if order <= 1:
        return x
    # coefficients exacts connus jusqu'à 7
    s = x
    if order >= 3: s += (x**3)/3.0
    if order >= 5: s += 2.0*(x**5)/15.0
    if order >= 7: s += 17.0*(x**7)/315.0
    return s

def dl_sinh(x: float, order: int = 9) -> float:
    """
    sinh x = x + x^3/3! + x^5/5! + ...
    """
    s = 0.0
    for k in range(0, (order//2)+1):
        n = 2*k + 1
        s += (x**n) / math.factorial(n)
    return s

def dl_cosh(x: float, order: int = 8) -> float:
    """
    cosh x = 1 + x^2/2! + x^4/4! + ...
    """
    s = 0.0
    for k in range(0, (order//2)+1):
        n = 2*k
        s += (x**n) / math.factorial(n)
    return s

def dl_tanh(x: float, order: int = 7) -> float:
    """
    tanh x = x - x^3/3 + 2x^5/15 - 17x^7/315 + ...
    """
    if order <= 1:
        return x
    s = x
    if order >= 3: s -= (x**3)/3.0
    if order >= 5: s += 2.0*(x**5)/15.0
    if order >= 7: s -= 17.0*(x**7)/315.0
    return s


# =============================================================================
# TABLES D'ANGLES REMARQUABLES (RAD)
# =============================================================================

@dataclass(frozen=True)
class RemarkableAngle:
    label: str
    rad: float
    sin: float
    cos: float
    tan: Optional[float]

REMARKABLE_ANGLES: List[RemarkableAngle] = [
    RemarkableAngle("0", 0.0, 0.0, 1.0, 0.0),
    RemarkableAngle("pi/6", PI/6, 0.5, math.sqrt(3)/2, 1/math.sqrt(3)),
    RemarkableAngle("pi/4", PI/4, math.sqrt(2)/2, math.sqrt(2)/2, 1.0),
    RemarkableAngle("pi/3", PI/3, math.sqrt(3)/2, 0.5, math.sqrt(3)),
    RemarkableAngle("pi/2", PI/2, 1.0, 0.0, None),
    RemarkableAngle("pi", PI, 0.0, -1.0, 0.0),
    RemarkableAngle("3pi/2", 3*PI/2, -1.0, 0.0, None),
    RemarkableAngle("2pi", 2*PI, 0.0, 1.0, 0.0),
]


# =============================================================================
# EXPORT D'UN DICTIONNAIRE "FORMULES" POUR TON SITE
# =============================================================================

Formula = Callable[..., float]

FORMULAS_NUMERIC: Dict[str, Formula] = {
    # base
    "sin": sin, "cos": cos, "tan": tan,
    "cot": cot, "sec": sec, "csc": csc,
    # addition
    "sin_add": sin_add, "cos_add": cos_add, "tan_add": tan_add,
    "sin_sub": sin_sub, "cos_sub": cos_sub, "tan_sub": tan_sub,
    # double/triple
    "sin_double": sin_double, "cos_double": cos_double, "tan_double": tan_double,
    "sin_triple": sin_triple, "cos_triple": cos_triple, "tan_triple": tan_triple,
    # reductions
    "sin2": sin2_to_cos2x, "cos2": cos2_to_cos2x, "sincos": sincos_to_sin2x,
    "sin3": sin3_reduction, "cos3": cos3_reduction,
    "sin4": sin4_reduction, "cos4": cos4_reduction,
    # hyperbolic
    "sinh": sinh, "cosh": cosh, "tanh": tanh,
    "sech": sech, "csch": csch, "coth": coth,
    "sinh_add": sinh_add, "cosh_add": cosh_add, "tanh_add": tanh_add,
    "sinh_double": sinh_double, "cosh_double": cosh_double, "tanh_double": tanh_double,
}

FORMULAS_TEXT: Dict[str, str] = {
    # primitives
    "int_sin": int_sin(),
    "int_cos": int_cos(),
    "int_tan": int_tan(),
    "int_cot": int_cot(),
    "int_sec2": int_sec2(),
    "int_csc2": int_csc2(),
    "int_sec_tan": int_sec_tan(),
    "int_csc_cot": int_csc_cot(),
    "int_sinh": int_sinh(),
    "int_cosh": int_cosh(),
    "int_sech2": int_sech2(),
    "int_csch2": int_csch2(),
    "int_sech_tanh": int_sech_tanh(),
}
