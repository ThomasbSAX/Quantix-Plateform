"""
Module: linear_algebra_core
Noyau propre de calcul matriciel (listes Python)
Conçu pour être :
- déterministe
- pédagogique
- interopérable OCR → calcul
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import math
import cmath

Matrix = List[List[float]]
Vector = List[float]


# =============================================================================
# OUTILS DE BASE
# =============================================================================

def shape(A: Matrix) -> Tuple[int, int]:
    if not A or not A[0]:
        raise ValueError("Matrice vide")
    return len(A), len(A[0])


def zeros(n: int, m: int) -> Matrix:
    return [[0.0 for _ in range(m)] for _ in range(n)]


def identity(n: int) -> Matrix:
    I = zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I


def copy(A: Matrix) -> Matrix:
    return [row[:] for row in A]


def transpose(A: Matrix) -> Matrix:
    n, m = shape(A)
    T = zeros(m, n)
    for i in range(n):
        for j in range(m):
            T[j][i] = A[i][j]
    return T


# =============================================================================
# OPÉRATIONS ARITHMÉTIQUES
# =============================================================================

def add(A: Matrix, B: Matrix) -> Matrix:
    n, m = shape(A)
    if shape(B) != (n, m):
        raise ValueError("Dimensions incompatibles")
    C = zeros(n, m)
    for i in range(n):
        for j in range(m):
            C[i][j] = A[i][j] + B[i][j]
    return C


def sub(A: Matrix, B: Matrix) -> Matrix:
    n, m = shape(A)
    if shape(B) != (n, m):
        raise ValueError("Dimensions incompatibles")
    C = zeros(n, m)
    for i in range(n):
        for j in range(m):
            C[i][j] = A[i][j] - B[i][j]
    return C


def scalar_mul(A: Matrix, λ: float) -> Matrix:
    n, m = shape(A)
    B = zeros(n, m)
    for i in range(n):
        for j in range(m):
            B[i][j] = λ * A[i][j]
    return B


def matmul(A: Matrix, B: Matrix) -> Matrix:
    n, k = shape(A)
    k2, m = shape(B)
    if k != k2:
        raise ValueError("Produit matriciel impossible")
    C = zeros(n, m)
    for i in range(n):
        for j in range(m):
            s = 0.0
            for t in range(k):
                s += A[i][t] * B[t][j]
            C[i][j] = s
    return C


# =============================================================================
# PROPRIÉTÉS STRUCTURELLES
# =============================================================================

def is_square(A: Matrix) -> bool:
    n, m = shape(A)
    return n == m


def trace(A: Matrix) -> float:
    if not is_square(A):
        raise ValueError("Trace définie uniquement pour matrices carrées")
    return sum(A[i][i] for i in range(len(A)))


def is_symmetric(A: Matrix, tol: float = 1e-10) -> bool:
    if not is_square(A):
        return False
    AT = transpose(A)
    n, m = shape(A)
    for i in range(n):
        for j in range(m):
            if abs(A[i][j] - AT[i][j]) > tol:
                return False
    return True


# =============================================================================
# DÉTERMINANT (GAUSS)
# =============================================================================

def determinant(A: Matrix, tol: float = 1e-12) -> float:
    if not is_square(A):
        raise ValueError("Déterminant: matrice non carrée")

    n = len(A)
    M = copy(A)
    det = 1.0

    for i in range(n):
        pivot = i
        for k in range(i, n):
            if abs(M[k][i]) > abs(M[pivot][i]):
                pivot = k

        if abs(M[pivot][i]) < tol:
            return 0.0

        if pivot != i:
            M[i], M[pivot] = M[pivot], M[i]
            det *= -1.0

        pivot_val = M[i][i]
        det *= pivot_val

        for k in range(i + 1, n):
            factor = M[k][i] / pivot_val
            for j in range(i, n):
                M[k][j] -= factor * M[i][j]

    return det


# =============================================================================
# INVERSE (GAUSS–JORDAN)
# =============================================================================

def inverse(A: Matrix, tol: float = 1e-12) -> Matrix:
    if not is_square(A):
        raise ValueError("Inverse: matrice non carrée")

    n = len(A)
    M = [A[i][:] + identity(n)[i] for i in range(n)]

    for i in range(n):
        pivot = i
        for k in range(i, n):
            if abs(M[k][i]) > abs(M[pivot][i]):
                pivot = k

        if abs(M[pivot][i]) < tol:
            raise ValueError("Matrice non inversible")

        M[i], M[pivot] = M[pivot], M[i]

        piv = M[i][i]
        for j in range(2 * n):
            M[i][j] /= piv

        for k in range(n):
            if k != i:
                factor = M[k][i]
                for j in range(2 * n):
                    M[k][j] -= factor * M[i][j]

    return [row[n:] for row in M]


# =============================================================================
# PUISSANCES DE MATRICES
# =============================================================================

def matrix_power(A: Matrix, k: int) -> Matrix:
    if not is_square(A):
        raise ValueError("Puissance: matrice non carrée")

    if k == 0:
        return identity(len(A))
    if k < 0:
        return matrix_power(inverse(A), -k)

    R = identity(len(A))
    B = copy(A)

    while k > 0:
        if k % 2 == 1:
            R = matmul(R, B)
        B = matmul(B, B)
        k //= 2

    return R

# =============================================================================
# OUTILS NUMÉRIQUES
# =============================================================================

def _is_close(a: float, b: float, tol: float = 1e-10) -> bool:
    return abs(a - b) <= tol

def _vector_norm2(x: Vector) -> float:
    return math.sqrt(sum(t*t for t in x))

def _dot(u: Vector, v: Vector) -> float:
    if len(u) != len(v):
        raise ValueError("Produit scalaire: dimensions incompatibles")
    return sum(ui*vi for ui, vi in zip(u, v))

def _axpy(a: float, x: Vector, y: Vector) -> Vector:
    if len(x) != len(y):
        raise ValueError("axpy: dimensions incompatibles")
    return [a*xi + yi for xi, yi in zip(x, y)]

def _scale(x: Vector, a: float) -> Vector:
    return [a*xi for xi in x]

def _subvec(x: Vector, y: Vector) -> Vector:
    if len(x) != len(y):
        raise ValueError("Soustraction vecteurs: dimensions incompatibles")
    return [xi - yi for xi, yi in zip(x, y)]

def _matvec(A: Matrix, x: Vector) -> Vector:
    n, m = shape(A)
    if len(x) != m:
        raise ValueError("Produit matrice-vecteur: dimensions incompatibles")
    out = [0.0]*n
    for i in range(n):
        s = 0.0
        row = A[i]
        for j in range(m):
            s += row[j]*x[j]
        out[i] = s
    return out


# =============================================================================
# PRIMITIFS CORE (copiés ici pour module standalone si besoin)
# =============================================================================

def shape(A: Matrix) -> Tuple[int, int]:
    if not A or not A[0]:
        raise ValueError("Matrice vide")
    return len(A), len(A[0])

def zeros(n: int, m: int) -> Matrix:
    return [[0.0 for _ in range(m)] for _ in range(n)]

def identity(n: int) -> Matrix:
    I = zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I

def copy(A: Matrix) -> Matrix:
    return [row[:] for row in A]

def transpose(A: Matrix) -> Matrix:
    n, m = shape(A)
    T = zeros(m, n)
    for i in range(n):
        for j in range(m):
            T[j][i] = A[i][j]
    return T

def matmul(A: Matrix, B: Matrix) -> Matrix:
    n, k = shape(A)
    k2, m = shape(B)
    if k != k2:
        raise ValueError("Produit matriciel impossible")
    C = zeros(n, m)
    for i in range(n):
        for j in range(m):
            s = 0.0
            for t in range(k):
                s += A[i][t]*B[t][j]
            C[i][j] = s
    return C


# =============================================================================
# ÉLIMINATION DE GAUSS: rang, solve, inverse partiel
# =============================================================================

def rref(A: Matrix, tol: float = 1e-12) -> Tuple[Matrix, List[int]]:
    """
    Forme échelonnée réduite (RREF) + pivots colonnes.
    Retourne (R, pivots).
    """
    M = copy(A)
    n, m = shape(M)
    pivots: List[int] = []
    r = 0

    for c in range(m):
        if r >= n:
            break

        pivot = r
        for i in range(r, n):
            if abs(M[i][c]) > abs(M[pivot][c]):
                pivot = i

        if abs(M[pivot][c]) < tol:
            continue

        M[r], M[pivot] = M[pivot], M[r]
        piv = M[r][c]

        for j in range(c, m):
            M[r][j] /= piv

        for i in range(n):
            if i != r:
                factor = M[i][c]
                if abs(factor) > tol:
                    for j in range(c, m):
                        M[i][j] -= factor * M[r][j]

        pivots.append(c)
        r += 1

    for i in range(n):
        for j in range(m):
            if abs(M[i][j]) < tol:
                M[i][j] = 0.0

    return M, pivots


def rank(A: Matrix, tol: float = 1e-12) -> int:
    R, pivots = rref(A, tol=tol)
    return len(pivots)


def solve_gauss(A: Matrix, b: Vector, tol: float = 1e-12) -> Vector:
    """
    Résout Ax=b si solution unique. (Gauss pivot partiel)
    """
    n, m = shape(A)
    if n != m:
        raise ValueError("solve_gauss: matrice non carrée")
    if len(b) != n:
        raise ValueError("solve_gauss: dimension de b invalide")

    M = [A[i][:] + [float(b[i])] for i in range(n)]

    for i in range(n):
        pivot = i
        for k in range(i, n):
            if abs(M[k][i]) > abs(M[pivot][i]):
                pivot = k

        if abs(M[pivot][i]) < tol:
            raise ValueError("Système singulier ou non unique")

        M[i], M[pivot] = M[pivot], M[i]

        piv = M[i][i]
        for j in range(i, n+1):
            M[i][j] /= piv

        for k in range(i+1, n):
            factor = M[k][i]
            if abs(factor) > tol:
                for j in range(i, n+1):
                    M[k][j] -= factor * M[i][j]

    x = [0.0]*n
    for i in range(n-1, -1, -1):
        s = M[i][n]
        for j in range(i+1, n):
            s -= M[i][j]*x[j]
        x[i] = s

    return x


def solve_least_squares(A: Matrix, b: Vector, tol: float = 1e-12) -> Vector:
    """
    Moindres carrés via équations normales: x = (A^T A)^{-1} A^T b
    Attention: numériquement moins stable que QR, mais acceptable pour MVP.
    """
    AT = transpose(A)
    ATA = matmul(AT, A)
    ATb = _matvec(AT, b)
    return solve_gauss(ATA, ATb, tol=tol)


# =============================================================================
# NOYAU / IMAGE (bases)
# =============================================================================

def nullspace_basis(A: Matrix, tol: float = 1e-12) -> List[Vector]:
    """
    Base du noyau de A (solutions de Ax=0) via RREF.
    Retourne une liste de vecteurs (base).
    """
    R, piv = rref(A, tol=tol)
    n, m = shape(R)
    piv_set = set(piv)
    free_cols = [j for j in range(m) if j not in piv_set]

    if not free_cols:
        return []

    basis: List[Vector] = []
    for fc in free_cols:
        x = [0.0]*m
        x[fc] = 1.0
        for i, pc in enumerate(piv):
            x[pc] = -R[i][fc]
        basis.append(x)

    return basis


def column_space_basis(A: Matrix, tol: float = 1e-12) -> List[Vector]:
    """
    Base de l'image (espace colonne) via pivots colonnes de RREF.
    Retourne les colonnes de A correspondantes.
    """
    R, piv = rref(A, tol=tol)
    n, m = shape(A)
    basis: List[Vector] = []
    for j in piv:
        col = [A[i][j] for i in range(n)]
        basis.append(col)
    return basis


# =============================================================================
# ORTHONORMALISATION / QR (Gram-Schmidt)
# =============================================================================

def gram_schmidt_columns(A: Matrix, tol: float = 1e-12) -> Tuple[Matrix, List[int]]:
    """
    Orthonormalise les colonnes de A.
    Retourne Q (n×r) et indices des colonnes gardées.
    """
    n, m = shape(A)
    Qcols: List[Vector] = []
    kept: List[int] = []

    for j in range(m):
        v = [A[i][j] for i in range(n)]
        for q in Qcols:
            v = _subvec(v, _scale(q, _dot(q, v)))
        nv = _vector_norm2(v)
        if nv > tol:
            q = _scale(v, 1.0/nv)
            Qcols.append(q)
            kept.append(j)

    r = len(Qcols)
    Q = zeros(n, r)
    for j in range(r):
        for i in range(n):
            Q[i][j] = Qcols[j][i]
    return Q, kept


def qr_decomposition(A: Matrix, tol: float = 1e-12) -> Tuple[Matrix, Matrix]:
    """
    QR (Gram-Schmidt modifié implicite) : A ≈ Q R.
    Q: n×r orthonormale, R: r×m.
    """
    n, m = shape(A)
    Q, _ = gram_schmidt_columns(A, tol=tol)
    QT = transpose(Q)
    R = matmul(QT, A)
    return Q, R


# =============================================================================
# PROJECTEURS
# =============================================================================

def projector_onto_columns(A: Matrix, tol: float = 1e-12) -> Matrix:
    """
    Projecteur orthogonal sur Im(A):
    P = A (A^T A)^{-1} A^T
    """
    AT = transpose(A)
    ATA = matmul(AT, A)
    ATA_inv = inverse_via_solve(ATA, tol=tol)
    temp = matmul(A, ATA_inv)
    return matmul(temp, AT)


def projector_onto_vector(v: Vector, tol: float = 1e-12) -> Matrix:
    """
    Projecteur sur la direction v :
    P = (v v^T) / (v^T v)
    """
    denom = _dot(v, v)
    if abs(denom) < tol:
        raise ValueError("projector_onto_vector: vecteur nul")
    n = len(v)
    P = zeros(n, n)
    for i in range(n):
        for j in range(n):
            P[i][j] = (v[i]*v[j]) / denom
    return P


# =============================================================================
# INVERSE "VIA SOLVE" (utile ici pour éviter dépendance au module précédent)
# =============================================================================

def inverse_via_solve(A: Matrix, tol: float = 1e-12) -> Matrix:
    """
    Inverse de A via résolution des n systèmes Ax=e_i.
    Stable et simple (pivot partiel).
    """
    n, m = shape(A)
    if n != m:
        raise ValueError("inverse: matrice non carrée")
    Inv = zeros(n, n)
    for j in range(n):
        e = [0.0]*n
        e[j] = 1.0
        col = solve_gauss(A, e, tol=tol)
        for i in range(n):
            Inv[i][j] = col[i]
    return Inv


# =============================================================================
# NORMES MATRICIELLES + CONDITIONNEMENT
# =============================================================================

def norm_fro(A: Matrix) -> float:
    n, m = shape(A)
    s = 0.0
    for i in range(n):
        for j in range(m):
            s += A[i][j]*A[i][j]
    return math.sqrt(s)


def norm_1(A: Matrix) -> float:
    n, m = shape(A)
    best = 0.0
    for j in range(m):
        s = 0.0
        for i in range(n):
            s += abs(A[i][j])
        if s > best:
            best = s
    return best


def norm_inf(A: Matrix) -> float:
    n, m = shape(A)
    best = 0.0
    for i in range(n):
        s = 0.0
        for j in range(m):
            s += abs(A[i][j])
        if s > best:
            best = s
    return best


def cond_number(A: Matrix, p: str = "2", tol: float = 1e-12) -> float:
    """
    Conditionnement approximatif:
    - "1": ||A||1 * ||A^{-1}||1
    - "inf": idem
    - "fro": idem
    - "2": approximation via power iteration sur A^T A (spectrale)
    """
    if p not in {"1", "inf", "fro", "2"}:
        raise ValueError("p doit être dans {'1','inf','fro','2'}")

    Ainv = inverse_via_solve(A, tol=tol)

    if p == "1":
        return norm_1(A) * norm_1(Ainv)
    if p == "inf":
        return norm_inf(A) * norm_inf(Ainv)
    if p == "fro":
        return norm_fro(A) * norm_fro(Ainv)

    # p == "2": approx ||A||2 ≈ sqrt(lambda_max(A^T A))
    AT = transpose(A)
    ATA = matmul(AT, A)
    lam = _power_iteration_max_eigen(ATA, tol=1e-10, iters=200)
    ATAi = matmul(transpose(Ainv), Ainv)  # (A^{-1})^T A^{-1}
    lam_inv = _power_iteration_max_eigen(ATAi, tol=1e-10, iters=200)
    return math.sqrt(lam) * math.sqrt(lam_inv)


def _power_iteration_max_eigen(A: Matrix, tol: float = 1e-10, iters: int = 200) -> float:
    """
    Retourne une approximation de la plus grande valeur propre de A (supposée symétrique PSD).
    """
    n, m = shape(A)
    if n != m:
        raise ValueError("power_iteration: matrice non carrée")

    x = [1.0/math.sqrt(n)]*n
    last = 0.0

    for _ in range(iters):
        y = _matvec(A, x)
        ny = _vector_norm2(y)
        if ny < tol:
            return 0.0
        x = _scale(y, 1.0/ny)
        Ax = _matvec(A, x)
        val = _dot(x, Ax)
        if abs(val - last) < tol:
            return float(val)
        last = val

    return float(last)


# =============================================================================
# TESTS DE PROPRIÉTÉS (tolérance)
# =============================================================================

def matrices_equal(A: Matrix, B: Matrix, tol: float = 1e-10) -> bool:
    n, m = shape(A)
    if shape(B) != (n, m):
        return False
    for i in range(n):
        for j in range(m):
            if abs(A[i][j] - B[i][j]) > tol:
                return False
    return True


def is_orthogonal(A: Matrix, tol: float = 1e-10) -> bool:
    n, m = shape(A)
    if n != m:
        return False
    AT = transpose(A)
    P = matmul(A, AT)
    return matrices_equal(P, identity(n), tol=tol)


def is_idempotent(A: Matrix, tol: float = 1e-10) -> bool:
    if not (shape(A)[0] == shape(A)[1]):
        return False
    A2 = matmul(A, A)
    return matrices_equal(A2, A, tol=tol)


def is_involutory(A: Matrix, tol: float = 1e-10) -> bool:
    if not (shape(A)[0] == shape(A)[1]):
        return False
    A2 = matmul(A, A)
    return matrices_equal(A2, identity(len(A)), tol=tol)


def is_nilpotent(A: Matrix, max_k: int = 20, tol: float = 1e-12) -> Tuple[bool, Optional[int]]:
    """
    Test naïf: A^k ~ 0 pour un k <= max_k.
    Attention: coûteux O(k n^3).
    """
    n, m = shape(A)
    if n != m:
        return False, None

    P = copy(A)
    for k in range(1, max_k+1):
        if _is_zero_matrix(P, tol=tol):
            return True, k
        P = matmul(P, A)
    return False, None


def _is_zero_matrix(A: Matrix, tol: float = 1e-12) -> bool:
    n, m = shape(A)
    for i in range(n):
        for j in range(m):
            if abs(A[i][j]) > tol:
                return False
    return True


# --- core minimal attendu (à importer idéalement) ---
def shape(A: Matrix) -> Tuple[int, int]:
    if not A or not A[0]:
        raise ValueError("Matrice vide")
    return len(A), len(A[0])

def zeros(n: int, m: int) -> Matrix:
    return [[0.0 for _ in range(m)] for _ in range(n)]

def identity(n: int) -> Matrix:
    I = zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I

def copy(A: Matrix) -> Matrix:
    return [row[:] for row in A]

def transpose(A: Matrix) -> Matrix:
    n, m = shape(A)
    T = zeros(m, n)
    for i in range(n):
        for j in range(m):
            T[j][i] = A[i][j]
    return T

def matmul(A: Matrix, B: Matrix) -> Matrix:
    n, k = shape(A)
    k2, m = shape(B)
    if k != k2:
        raise ValueError("Produit matriciel impossible")
    C = zeros(n, m)
    for i in range(n):
        for j in range(m):
            s = 0.0
            for t in range(k):
                s += A[i][t]*B[t][j]
            C[i][j] = s
    return C

def matvec(A: Matrix, x: Vector) -> Vector:
    n, m = shape(A)
    if len(x) != m:
        raise ValueError("Produit matrice-vecteur: dimensions incompatibles")
    out = [0.0]*n
    for i in range(n):
        s = 0.0
        row = A[i]
        for j in range(m):
            s += row[j]*x[j]
        out[i] = s
    return out

# =============================================================================
# OUTILS
# =============================================================================

def _dot(u: Vector, v: Vector) -> float:
    if len(u) != len(v):
        raise ValueError("dot: dimensions incompatibles")
    return sum(ui*vi for ui, vi in zip(u, v))

def _norm2(x: Vector) -> float:
    return math.sqrt(sum(t*t for t in x))

def _scale(x: Vector, a: float) -> Vector:
    return [a*xi for xi in x]

def _sub(x: Vector, y: Vector) -> Vector:
    if len(x) != len(y):
        raise ValueError("sub: dimensions incompatibles")
    return [xi-yi for xi, yi in zip(x, y)]

def _is_upper_triangular(A: Matrix, tol: float = 1e-10) -> bool:
    n, m = shape(A)
    if n != m:
        return False
    for i in range(n):
        for j in range(i):
            if abs(A[i][j]) > tol:
                return False
    return True

def _trace(A: Matrix) -> float:
    n, m = shape(A)
    if n != m:
        raise ValueError("trace: matrice non carrée")
    return sum(A[i][i] for i in range(n))

# =============================================================================
# CARACTÉRISTIQUE: 2x2 / 3x3
# =============================================================================

def eigvals_2x2(A: Matrix) -> List[complex]:
    n, m = shape(A)
    if (n, m) != (2, 2):
        raise ValueError("eigvals_2x2: attendu 2x2")
    a, b = A[0][0], A[0][1]
    c, d = A[1][0], A[1][1]
    tr = a + d
    det = a*d - b*c
    disc = tr*tr - 4.0*det
    if disc >= 0:
        r = math.sqrt(disc)
        return [0.5*(tr + r), 0.5*(tr - r)]
    r = math.sqrt(-disc)
    return [complex(0.5*tr, 0.5*r), complex(0.5*tr, -0.5*r)]

def det_3x3(A: Matrix) -> float:
    n, m = shape(A)
    if (n, m) != (3, 3):
        raise ValueError("det_3x3: attendu 3x3")
    a,b,c = A[0]
    d,e,f = A[1]
    g,h,i = A[2]
    return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)

def eigvals_3x3_real(A: Matrix, iters: int = 200, tol: float = 1e-10) -> List[complex]:
    """
    Approx via QR itératif non-shifté (OK pour matrices “gentilles”).
    Pour cas difficiles: fallback numpy recommandé.
    """
    n, m = shape(A)
    if (n, m) != (3, 3):
        raise ValueError("eigvals_3x3_real: attendu 3x3")
    T = copy(A)
    for _ in range(iters):
        Q, R = qr_householder(T)
        T = matmul(R, Q)
        if _is_upper_triangular(T, tol=tol):
            break
    # Extraction: diag + éventuel bloc 2x2
    vals: List[complex] = []
    i = 0
    while i < 3:
        if i < 2 and abs(T[i+1][i]) > tol:
            B = [[T[i][i], T[i][i+1]],
                 [T[i+1][i], T[i+1][i+1]]]
            vals.extend(eigvals_2x2(B))
            i += 2
        else:
            vals.append(T[i][i])
            i += 1
    return vals

# =============================================================================
# QR (Householder) – stable, utile pour spectral
# =============================================================================

def qr_householder(A: Matrix, tol: float = 1e-12) -> Tuple[Matrix, Matrix]:
    """
    QR Householder: A = Q R, Q orthogonale, R triangulaire sup.
    """
    n, m = shape(A)
    R = copy(A)
    Q = identity(n)

    for k in range(min(n, m)):
        # construire le vecteur x = R[k:, k]
        x = [R[i][k] for i in range(k, n)]
        nx = _norm2(x)
        if nx < tol:
            continue
        sign = 1.0 if x[0] >= 0 else -1.0
        u1 = x[0] + sign*nx
        v = x[:]
        v[0] = u1
        nv = _norm2(v)
        if nv < tol:
            continue
        v = _scale(v, 1.0/nv)

        # appliquer H = I - 2 v v^T sur R (à gauche) pour lignes k..n-1
        for j in range(k, m):
            col = [R[i][j] for i in range(k, n)]
            proj = 2.0*_dot(v, col)
            col2 = _sub(col, _scale(v, proj))
            for i in range(k, n):
                R[i][j] = col2[i-k]

        # accumuler Q = Q H^T = Q H (H symétrique)
        for j in range(n):
            col = [Q[i][j] for i in range(k, n)]
            proj = 2.0*_dot(v, col)
            col2 = _sub(col, _scale(v, proj))
            for i in range(k, n):
                Q[i][j] = col2[i-k]

    # Q construit en appliquant Householder sur colonnes de Q (à gauche), on a Q = H_p...H_1
    # Ici il correspond déjà au bon Q (orthogonal). R est triangulaire sup.
    return Q, R

# =============================================================================
# ITÉRATION QR (générale)
# =============================================================================

def eigvals_qr(A: Matrix, iters: int = 500, tol: float = 1e-10, shift: bool = True) -> List[complex]:
    """
    Valeurs propres via QR itératif avec shift (Wilkinson simplifié sur 2x2 bas droite).
    Retourne des valeurs (complexes possibles via blocs 2x2).
    """
    n, m = shape(A)
    if n != m:
        raise ValueError("eigvals_qr: matrice non carrée")

    T = copy(A)

    def wilkinson_mu(T: Matrix) -> float:
        # shift basé sur bloc 2x2 (n-2,n-1)
        a = T[n-2][n-2]
        b = T[n-2][n-1]
        c = T[n-1][n-2]
        d = T[n-1][n-1]
        tr = a + d
        det = a*d - b*c
        disc = tr*tr - 4.0*det
        if disc < 0:
            return d
        r = math.sqrt(disc)
        lam1 = 0.5*(tr + r)
        lam2 = 0.5*(tr - r)
        return lam1 if abs(lam1 - d) < abs(lam2 - d) else lam2

    for _ in range(iters):
        if _is_upper_triangular(T, tol=tol):
            break
        mu = wilkinson_mu(T) if (shift and n >= 2) else 0.0
        if shift:
            for i in range(n):
                T[i][i] -= mu
        Q, R = qr_householder(T)
        T = matmul(R, Q)
        if shift:
            for i in range(n):
                T[i][i] += mu

    # extraction par blocs 1x1 / 2x2
    vals: List[complex] = []
    i = 0
    while i < n:
        if i < n-1 and abs(T[i+1][i]) > tol:
            B = [[T[i][i], T[i][i+1]],
                 [T[i+1][i], T[i+1][i+1]]]
            vals.extend(eigvals_2x2(B))
            i += 2
        else:
            vals.append(T[i][i])
            i += 1
    return vals

def spectral_radius(A: Matrix, **kwargs) -> float:
    vals = eigvals_qr(A, **kwargs)
    return max(abs(v) for v in vals)

# =============================================================================
# DIAGNOSTICS DIAGONALISATION (numérique)
# =============================================================================

def diagonalization_diagnostics(
    A: Matrix,
    tol: float = 1e-8,
    qr_iters: int = 500
) -> Dict[str, Any]:
    """
    Heuristique: calcule approximativement les valeurs propres, puis estime multiplicité géométrique
    via rang(A - λI). Pour l’algèbre exacte, il faut rationnels/symbolique.
    """
    n, m = shape(A)
    if n != m:
        raise ValueError("diagonalization_diagnostics: matrice non carrée")

    eigs = eigvals_qr(A, iters=qr_iters, tol=tol, shift=True)

    # regrouper valeurs proches
    groups: List[List[complex]] = []
    for lam in eigs:
        placed = False
        for g in groups:
            if abs(lam - g[0]) <= 10*tol:
                g.append(lam)
                placed = True
                break
        if not placed:
            groups.append([lam])

    # rang via RREF (simple, local)
    def rref_rank(M: Matrix, t: float) -> int:
        X = copy(M)
        r = 0
        N, P = shape(X)
        for c in range(P):
            if r >= N:
                break
            piv = r
            for i in range(r, N):
                if abs(X[i][c]) > abs(X[piv][c]):
                    piv = i
            if abs(X[piv][c]) < t:
                continue
            X[r], X[piv] = X[piv], X[r]
            pv = X[r][c]
            for j in range(c, P):
                X[r][j] /= pv
            for i in range(N):
                if i != r:
                    f = X[i][c]
                    if abs(f) > t:
                        for j in range(c, P):
                            X[i][j] -= f*X[r][j]
            r += 1
        return r

    info = []
    for g in groups:
        lam = sum(g)/len(g)
        mult_alg = len(g)
        # matrice M = A - lam I (prendre partie réelle si imag très petite)
        if abs(lam.imag) <= tol:
            lam_eff = float(lam.real)
            M = copy(A)
            for i in range(n):
                M[i][i] -= lam_eff
            r = rref_rank(M, t=tol)
            mult_geo = n - r
            info.append({
                "lambda": lam_eff,
                "mult_alg": mult_alg,
                "mult_geo_est": mult_geo,
                "defect_est": mult_alg - mult_geo
            })
        else:
            info.append({
                "lambda": lam,
                "mult_alg": mult_alg,
                "mult_geo_est": None,
                "defect_est": None
            })

    diagonalisable_est = all(
        (d["defect_est"] is None) or (d["defect_est"] <= 0)
        for d in info
    )

    return {
        "eigs_est": eigs,
        "groups": info,
        "diagonalisable_est": diagonalisable_est
    }

# =============================================================================
# FALLBACK NUMPY (optionnel)
# =============================================================================

def eig_numpy(A: Matrix) -> Optional[Tuple[List[complex], Matrix]]:
    try:
        import numpy as np
        w, v = np.linalg.eig(np.array(A, dtype=float))
        vals = [complex(x) for x in w.tolist()]
        vecs = v.tolist()
        return vals, vecs
    except Exception:
        return None


def matrix_exponential_series(A: Matrix, order: int = 20) -> Matrix:
    """
    exp(A) ≈ sum_{k=0}^order A^k / k!
    Stable seulement si ||A|| petit.
    """
    n, m = shape(A)
    if n != m:
        raise ValueError("exp: matrice non carrée")

    result = identity(n)
    term = identity(n)

    for k in range(1, order + 1):
        term = matmul(term, A)
        coef = 1.0 / math.factorial(k)
        result = add(result, scalar_mul(term, coef))

    return result

def matrix_exponential_diagonalizable(
    P: Matrix,
    D: Matrix,
    Pinv: Matrix
) -> Matrix:
    """
    Si A = P D P^{-1} avec D diagonale,
    alors exp(A) = P exp(D) P^{-1}.
    """
    n, _ = shape(D)
    expD = zeros(n, n)
    for i in range(n):
        expD[i][i] = math.exp(D[i][i])
    return matmul(P, matmul(expD, Pinv))

def is_positive_definite(A: Matrix, tol: float = 1e-10) -> bool:
    if not is_symmetric(A, tol=tol):
        return False
    try:
        _ = cholesky(A, tol=tol)
        return True
    except Exception:
        return False

def cholesky(A: Matrix, tol: float = 1e-12) -> Matrix:
    n, m = shape(A)
    if n != m:
        raise ValueError("Cholesky: matrice non carrée")

    L = zeros(n, n)
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                if val <= tol:
                    raise ValueError("Matrice non définie positive")
                L[i][j] = math.sqrt(val)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def spectral_decomposition_symmetric(A: Matrix):
    """
    A = Q Λ Q^T, A symétrique réelle
    """
    import numpy as np
    w, Q = np.linalg.eigh(np.array(A, dtype=float))
    Λ = [[0.0]*len(w) for _ in range(len(w))]
    for i, v in enumerate(w):
        Λ[i][i] = float(v)
    return Q.tolist(), Λ


def eigenspace(A: Matrix, lam: float, tol: float = 1e-8) -> List[Vector]:
    """
    Base de Ker(A - λI)
    """
    n, _ = shape(A)
    M = copy(A)
    for i in range(n):
        M[i][i] -= lam
    return nullspace_basis(M, tol=tol)


def dunford_decomposition(A: Matrix, tol: float = 1e-8):
    """
    Retourne (D, N) avec A ≈ D + N, [D,N]=0
    Heuristique numérique.
    """
    n, _ = shape(A)
    eigs = eigvals_qr(A, tol=tol)

    # Moyenne par classes proches
    lambdas = []
    for lam in eigs:
        if abs(lam.imag) < tol:
            lambdas.append(lam.real)

    D = zeros(n, n)
    for lam in lambdas:
        E = eigenspace(A, lam, tol=tol)
        for v in E:
            P = projector_onto_vector(v)
            D = add(D, scalar_mul(P, lam))

    N = sub(A, D)
    return D, N

def companion_matrix(coeffs: List[float]) -> Matrix:
    """
    Polynôme : x^n + a_{n-1} x^{n-1} + ... + a0
    """
    n = len(coeffs)
    C = zeros(n, n)
    for i in range(1, n):
        C[i][i-1] = 1.0
    for j in range(n):
        C[0][j] = -coeffs[j]
    return C

def pseudoinverse(A: Matrix):
    """
    Moore–Penrose via NumPy SVD (stable)
    """
    import numpy as np
    U, s, Vt = np.linalg.svd(np.array(A, dtype=float), full_matrices=False)
    s_inv = np.diag([1/x if x > 1e-12 else 0.0 for x in s])
    return (Vt.T @ s_inv @ U.T).tolist()


def matrix_exponential_symmetric(A: Matrix):
    Q, Λ = spectral_decomposition_symmetric(A)
    n = len(Λ)
    expΛ = zeros(n, n)
    for i in range(n):
        expΛ[i][i] = math.exp(Λ[i][i])
    QT = transpose(Q)
    return matmul(Q, matmul(expΛ, QT))


def is_diagonalizable(A: Matrix, tol: float = 1e-8) -> bool:
    info = diagonalization_diagnostics(A, tol=tol)
    return info["diagonalisable_est"]

def is_invertible(A: Matrix, tol: float = 1e-12) -> bool:
    return abs(determinant(A)) > tol

def minimal_polynomial_degree_est(A: Matrix, max_k: int = 10, tol: float = 1e-10) -> Optional[int]:
    n, _ = shape(A)
    P = identity(n)
    for k in range(1, max_k + 1):
        P = matmul(P, A)
        if _is_zero_matrix(P, tol=tol):
            return k
    return None

def est_inversible(
    A: Matrix,
    method: str = "auto",
    tol: float = 1e-12
) -> Dict[str, object]:
    """
    Teste l'inversibilité selon différentes méthodes.

    method ∈ {
        "auto",
        "det",
        "rank",
        "gauss",
        "diag_dom",
        "spd",
        "cond"
    }
    """
    if method == "auto":
        return _est_inversible_auto(A, tol)
    if method == "det":
        return _est_inversible_det(A, tol)
    if method == "rank":
        return _est_inversible_rank(A, tol)
    if method == "gauss":
        return _est_inversible_gauss(A, tol)
    if method == "diag_dom":
        return _est_inversible_diag_dom(A)
    if method == "spd":
        return _est_inversible_spd(A, tol)
    if method == "cond":
        return _est_inversible_cond(A, tol)

    raise ValueError("Méthode inconnue")


def _est_inversible_det(A: Matrix, tol: float) -> Dict[str, object]:
    try:
        d = determinant(A)
        return {
            "inversible": abs(d) > tol,
            "method": "det",
            "determinant": d
        }
    except Exception as e:
        return {"inversible": False, "method": "det", "error": str(e)}


def _est_inversible_rank(A: Matrix, tol: float) -> Dict[str, object]:
    try:
        n, m = shape(A)
        if n != m:
            return {"inversible": False, "method": "rank", "reason": "non carrée"}
        r = rank(A, tol=tol)
        return {
            "inversible": r == n,
            "method": "rank",
            "rank": r
        }
    except Exception as e:
        return {"inversible": False, "method": "rank", "error": str(e)}



def _est_inversible_gauss(A: Matrix, tol: float) -> Dict[str, object]:
    try:
        _ = inverse_via_solve(A, tol=tol)
        return {"inversible": True, "method": "gauss"}
    except Exception as e:
        return {"inversible": False, "method": "gauss", "error": str(e)}


def _est_inversible_diag_dom(A: Matrix) -> Dict[str, object]:
    n, m = shape(A)
    if n != m:
        return {"inversible": False, "method": "diag_dom", "reason": "non carrée"}

    for i in range(n):
        diag = abs(A[i][i])
        off = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag <= off:
            return {
                "inversible": False,
                "method": "diag_dom",
                "row": i
            }
    return {"inversible": True, "method": "diag_dom"}


def _est_inversible_spd(A: Matrix, tol: float) -> Dict[str, object]:
    try:
        if not is_symmetric(A, tol=tol):
            return {"inversible": False, "method": "spd", "reason": "non symétrique"}
        _ = cholesky(A, tol=tol)
        return {"inversible": True, "method": "spd"}
    except Exception as e:
        return {"inversible": False, "method": "spd", "error": str(e)}



def _est_inversible_cond(A: Matrix, tol: float) -> Dict[str, object]:
    try:
        c = cond_number(A, p="2", tol=tol)
        return {
            "inversible": math.isfinite(c),
            "method": "cond",
            "condition_number": c
        }
    except Exception as e:
        return {"inversible": False, "method": "cond", "error": str(e)}


def _est_inversible_auto(A: Matrix, tol: float) -> Dict[str, object]:
    n, m = shape(A)
    if n != m:
        return {"inversible": False, "method": "auto", "reason": "non carrée"}

    # tests rapides suffisants
    r = _est_inversible_diag_dom(A)
    if r["inversible"]:
        return r

    r = _est_inversible_spd(A, tol)
    if r["inversible"]:
        return r

    # tests généraux
    r = _est_inversible_gauss(A, tol)
    if r["inversible"]:
        return r

    return {"inversible": False, "method": "auto"}


def is_upper_triangular(A: Matrix, tol: float = 1e-12) -> bool:
    n, m = shape(A)
    if n != m:
        return False
    for i in range(n):
        for j in range(i):
            if abs(A[i][j]) > tol:
                return False
    return True


def is_lower_triangular(A: Matrix, tol: float = 1e-12) -> bool:
    n, m = shape(A)
    if n != m:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j]) > tol:
                return False
    return True

def is_diagonal(A: Matrix, tol: float = 1e-12) -> bool:
    return is_upper_triangular(A, tol) and is_lower_triangular(A, tol)


def is_scalar_matrix(A: Matrix, tol: float = 1e-12) -> bool:
    if not is_diagonal(A, tol):
        return False
    n, _ = shape(A)
    λ = A[0][0]
    for i in range(n):
        if abs(A[i][i] - λ) > tol:
            return False
    return True

def est_diagonalisable(
    A: Matrix,
    tol: float = 1e-8,
    qr_iters: int = 500
) -> Dict[str, object]:
    """
    Test heuristique de diagonalisabilité.
    Basé sur QR + calcul des dimensions de noyaux.
    """
    n, m = shape(A)
    if n != m:
        return {
            "diagonalisable": False,
            "reason": "non carrée"
        }

    eigs = eigvals_qr(A, iters=qr_iters, tol=tol, shift=True)

    # regrouper valeurs propres proches
    groups: List[List[complex]] = []
    for λ in eigs:
        placed = False
        for g in groups:
            if abs(λ - g[0]) <= 10 * tol:
                g.append(λ)
                placed = True
                break
        if not placed:
            groups.append([λ])

    diagnostics = []
    diagonalisable = True

    for g in groups:
        λ = sum(g) / len(g)
        mult_alg = len(g)

        if abs(λ.imag) > tol:
            diagnostics.append({
                "lambda": λ,
                "mult_alg": mult_alg,
                "note": "complexe → test géométrique ignoré"
            })
            continue

        λr = float(λ.real)
        M = copy(A)
        for i in range(n):
            M[i][i] -= λr

        r = rank(M, tol=tol)
        mult_geo = n - r

        if mult_geo < mult_alg:
            diagonalisable = False

        diagnostics.append({
            "lambda": λr,
            "mult_alg": mult_alg,
            "mult_geo": mult_geo,
            "defect": mult_alg - mult_geo
        })

    return {
        "diagonalisable": diagonalisable,
        "details": diagnostics
    }


def est_diagonalisable(
    A: Matrix,
    tol: float = 1e-8,
    qr_iters: int = 500
) -> Dict[str, object]:
    """
    Test heuristique de diagonalisabilité.
    Basé sur QR + calcul des dimensions de noyaux.
    """
    n, m = shape(A)
    if n != m:
        return {
            "diagonalisable": False,
            "reason": "non carrée"
        }

    eigs = eigvals_qr(A, iters=qr_iters, tol=tol, shift=True)

    # regrouper valeurs propres proches
    groups: List[List[complex]] = []
    for λ in eigs:
        placed = False
        for g in groups:
            if abs(λ - g[0]) <= 10 * tol:
                g.append(λ)
                placed = True
                break
        if not placed:
            groups.append([λ])

    diagnostics = []
    diagonalisable = True

    for g in groups:
        λ = sum(g) / len(g)
        mult_alg = len(g)

        if abs(λ.imag) > tol:
            diagnostics.append({
                "lambda": λ,
                "mult_alg": mult_alg,
                "note": "complexe → test géométrique ignoré"
            })
            continue

        λr = float(λ.real)
        M = copy(A)
        for i in range(n):
            M[i][i] -= λr

        r = rank(M, tol=tol)
        mult_geo = n - r

        if mult_geo < mult_alg:
            diagonalisable = False

        diagnostics.append({
            "lambda": λr,
            "mult_alg": mult_alg,
            "mult_geo": mult_geo,
            "defect": mult_alg - mult_geo
        })

    return {
        "diagonalisable": diagonalisable,
        "details": diagnostics
    }

def diagonalisable_if_triangular_distinct(A: Matrix, tol: float = 1e-12) -> bool:
    if not (is_upper_triangular(A, tol) or is_lower_triangular(A, tol)):
        return False
    diag = [A[i][i] for i in range(len(A))]
    return len(set(round(d, 8) for d in diag)) == len(diag)


def _est_inversible_auto(A: Matrix, tol: float = 1e-12) -> Dict[str, object]:
    n, m = shape(A)
    if n != m:
        return {
            "inversible": False,
            "reason": "non carrée"
        }

    # Critères suffisants rapides
    if _est_inversible_diag_dom(A)["inversible"]:
        return {
            "inversible": True,
            "method": "diag_dom"
        }

    if is_symmetric(A, tol):
        try:
            cholesky(A, tol=tol)
            return {
                "inversible": True,
                "method": "spd"
            }
        except Exception:
            pass

    # Critère général
    try:
        inverse_via_solve(A, tol=tol)
        return {
            "inversible": True,
            "method": "gauss"
        }
    except Exception:
        return {
            "inversible": False,
            "method": "auto"
        }

"""
Module: dunford_expm_luxe
But:
- Décomposition de Jordan "numérique-heuristique" (approx) à partir des espaces propres généralisés.
- Exponentielle de matrice via Dunford/Jordan : exp(A) = P exp(J) P^{-1}.

Hypothèses/limites (assumées, car c'est du numérique sans symbolique):
- Les valeurs propres proches sont regroupées (tol).
- Les chaînes de Jordan sont reconstruites via noyaux de (A-λI)^k.
- Stabilité: acceptable sur matrices "raisonnables" ; pour cas pathologiques, fallback numpy recommandé.

Dépendances: aucune (pur Python).
"""

MatrixC = List[List[complex]]
VectorC = List[complex]


# =============================================================================
# CORE COMPLEXE (minimal, autonome)
# =============================================================================

def shape(A: MatrixC) -> Tuple[int, int]:
    if not A or not A[0]:
        raise ValueError("Matrice vide")
    return len(A), len(A[0])


def zeros(n: int, m: int) -> MatrixC:
    return [[0.0 + 0.0j for _ in range(m)] for _ in range(n)]


def identity(n: int) -> MatrixC:
    I = zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0 + 0.0j
    return I


def copy(A: MatrixC) -> MatrixC:
    return [row[:] for row in A]


def transpose(A: MatrixC) -> MatrixC:
    n, m = shape(A)
    T = zeros(m, n)
    for i in range(n):
        for j in range(m):
            T[j][i] = A[i][j]
    return T


def add(A: MatrixC, B: MatrixC) -> MatrixC:
    n, m = shape(A)
    if shape(B) != (n, m):
        raise ValueError("add: dimensions incompatibles")
    C = zeros(n, m)
    for i in range(n):
        for j in range(m):
            C[i][j] = A[i][j] + B[i][j]
    return C


def sub(A: MatrixC, B: MatrixC) -> MatrixC:
    n, m = shape(A)
    if shape(B) != (n, m):
        raise ValueError("sub: dimensions incompatibles")
    C = zeros(n, m)
    for i in range(n):
        for j in range(m):
            C[i][j] = A[i][j] - B[i][j]
    return C


def scalar_mul(A: MatrixC, a: complex) -> MatrixC:
    n, m = shape(A)
    B = zeros(n, m)
    for i in range(n):
        for j in range(m):
            B[i][j] = a * A[i][j]
    return B


def matmul(A: MatrixC, B: MatrixC) -> MatrixC:
    n, k = shape(A)
    k2, m = shape(B)
    if k != k2:
        raise ValueError("matmul: produit impossible")
    C = zeros(n, m)
    for i in range(n):
        for j in range(m):
            s = 0.0 + 0.0j
            for t in range(k):
                s += A[i][t] * B[t][j]
            C[i][j] = s
    return C


def matvec(A: MatrixC, x: VectorC) -> VectorC:
    n, m = shape(A)
    if len(x) != m:
        raise ValueError("matvec: dimensions incompatibles")
    out: VectorC = [0.0 + 0.0j] * n
    for i in range(n):
        s = 0.0 + 0.0j
        row = A[i]
        for j in range(m):
            s += row[j] * x[j]
        out[i] = s
    return out


def norm2(x: VectorC) -> float:
    return math.sqrt(sum((abs(z) ** 2) for z in x))


def dot(u: VectorC, v: VectorC) -> complex:
    if len(u) != len(v):
        raise ValueError("dot: dimensions incompatibles")
    return sum(ui * vi for ui, vi in zip(u, v))


def is_square(A: MatrixC) -> bool:
    n, m = shape(A)
    return n == m


def to_complex(A: List[List[float]] | List[List[complex]]) -> MatrixC:
    return [[complex(a) for a in row] for row in A]


def _zero_small(A: MatrixC, tol: float) -> None:
    n, m = shape(A)
    for i in range(n):
        for j in range(m):
            if abs(A[i][j]) < tol:
                A[i][j] = 0.0 + 0.0j


# =============================================================================
# RREF / RANG / NOYAU (complexe)
# =============================================================================

def rref(A: MatrixC, tol: float = 1e-12) -> Tuple[MatrixC, List[int]]:
    M = copy(A)
    n, m = shape(M)
    pivots: List[int] = []
    r = 0
    for c in range(m):
        if r >= n:
            break
        pivot = r
        for i in range(r, n):
            if abs(M[i][c]) > abs(M[pivot][c]):
                pivot = i
        if abs(M[pivot][c]) < tol:
            continue
        M[r], M[pivot] = M[pivot], M[r]
        pv = M[r][c]
        for j in range(c, m):
            M[r][j] /= pv
        for i in range(n):
            if i != r:
                f = M[i][c]
                if abs(f) > tol:
                    for j in range(c, m):
                        M[i][j] -= f * M[r][j]
        pivots.append(c)
        r += 1
    _zero_small(M, tol)
    return M, pivots


def rank(A: MatrixC, tol: float = 1e-12) -> int:
    _, piv = rref(A, tol=tol)
    return len(piv)


def nullspace_basis(A: MatrixC, tol: float = 1e-12) -> List[VectorC]:
    R, piv = rref(A, tol=tol)
    n, m = shape(R)
    piv_set = set(piv)
    free_cols = [j for j in range(m) if j not in piv_set]
    if not free_cols:
        return []
    basis: List[VectorC] = []
    for fc in free_cols:
        x: VectorC = [0.0 + 0.0j] * m
        x[fc] = 1.0 + 0.0j
        for i, pc in enumerate(piv):
            x[pc] = -R[i][fc]
        basis.append(x)
    return basis


# =============================================================================
# SOLVE (Gauss pivot partiel, complexe)
# =============================================================================

def solve_gauss(A: MatrixC, b: VectorC, tol: float = 1e-12) -> VectorC:
    n, m = shape(A)
    if n != m:
        raise ValueError("solve_gauss: matrice non carrée")
    if len(b) != n:
        raise ValueError("solve_gauss: dimension de b invalide")

    M: MatrixC = [A[i][:] + [complex(b[i])] for i in range(n)]

    for i in range(n):
        pivot = i
        for k in range(i, n):
            if abs(M[k][i]) > abs(M[pivot][i]):
                pivot = k
        if abs(M[pivot][i]) < tol:
            raise ValueError("solve_gauss: système singulier ou non unique")
        M[i], M[pivot] = M[pivot], M[i]
        pv = M[i][i]
        for j in range(i, n + 1):
            M[i][j] /= pv
        for k in range(i + 1, n):
            f = M[k][i]
            if abs(f) > tol:
                for j in range(i, n + 1):
                    M[k][j] -= f * M[i][j]

    x: VectorC = [0.0 + 0.0j] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s
    return x


def inverse_via_solve(A: MatrixC, tol: float = 1e-12) -> MatrixC:
    n, m = shape(A)
    if n != m:
        raise ValueError("inverse: matrice non carrée")
    Inv = zeros(n, n)
    for j in range(n):
        e: VectorC = [0.0 + 0.0j] * n
        e[j] = 1.0 + 0.0j
        col = solve_gauss(A, e, tol=tol)
        for i in range(n):
            Inv[i][j] = col[i]
    return Inv


# =============================================================================
# QR Householder + itération QR => valeurs propres (complexes possibles)
# (version réelle/complexe unifiée)
# =============================================================================

def _is_upper_triangular(A: MatrixC, tol: float = 1e-10) -> bool:
    n, m = shape(A)
    if n != m:
        return False
    for i in range(n):
        for j in range(i):
            if abs(A[i][j]) > tol:
                return False
    return True


def qr_householder(A: MatrixC, tol: float = 1e-12) -> Tuple[MatrixC, MatrixC]:
    n, m = shape(A)
    R = copy(A)
    Q = identity(n)

    for k in range(min(n, m)):
        x = [R[i][k] for i in range(k, n)]
        nx = norm2(x)
        if nx < tol:
            continue
        sign = 1.0 if (abs(x[0]) < tol or (x[0].real >= 0)) else -1.0
        u1 = x[0] + complex(sign * nx, 0.0)
        v = x[:]
        v[0] = u1
        nv = norm2(v)
        if nv < tol:
            continue
        v = [vi / nv for vi in v]

        for j in range(k, m):
            col = [R[i][j] for i in range(k, n)]
            proj = 2.0 * dot(v, col)
            col2 = [col[t] - v[t] * proj for t in range(len(col))]
            for i in range(k, n):
                R[i][j] = col2[i - k]

        for j in range(n):
            col = [Q[i][j] for i in range(k, n)]
            proj = 2.0 * dot(v, col)
            col2 = [col[t] - v[t] * proj for t in range(len(col))]
            for i in range(k, n):
                Q[i][j] = col2[i - k]

    _zero_small(R, tol)
    _zero_small(Q, tol)
    return Q, R


def eigvals_2x2(B: MatrixC) -> List[complex]:
    a, b = B[0][0], B[0][1]
    c, d = B[1][0], B[1][1]
    tr = a + d
    det = a * d - b * c
    disc = tr * tr - 4.0 * det
    r = cmath.sqrt(disc)
    return [0.5 * (tr + r), 0.5 * (tr - r)]


def eigvals_qr(A: MatrixC, iters: int = 700, tol: float = 1e-10, shift: bool = True) -> List[complex]:
    n, m = shape(A)
    if n != m:
        raise ValueError("eigvals_qr: matrice non carrée")

    T = copy(A)

    def wilkinson_mu(T: MatrixC) -> complex:
        a = T[n - 2][n - 2]
        b = T[n - 2][n - 1]
        c = T[n - 1][n - 2]
        d = T[n - 1][n - 1]
        tr = a + d
        det = a * d - b * c
        disc = tr * tr - 4.0 * det
        r = cmath.sqrt(disc)
        lam1 = 0.5 * (tr + r)
        lam2 = 0.5 * (tr - r)
        return lam1 if abs(lam1 - d) < abs(lam2 - d) else lam2

    for _ in range(iters):
        if _is_upper_triangular(T, tol=tol):
            break
        mu = wilkinson_mu(T) if (shift and n >= 2) else 0.0 + 0.0j
        if shift:
            for i in range(n):
                T[i][i] -= mu
        Q, R = qr_householder(T)
        T = matmul(R, Q)
        if shift:
            for i in range(n):
                T[i][i] += mu

    vals: List[complex] = []
    i = 0
    while i < n:
        if i < n - 1 and abs(T[i + 1][i]) > tol:
            B = [[T[i][i], T[i][i + 1]], [T[i + 1][i], T[i + 1][i + 1]]]
            vals.extend(eigvals_2x2(B))
            i += 2
        else:
            vals.append(T[i][i])
            i += 1
    return vals


# =============================================================================
# Jordan "numérique": espaces propres généralisés + chaînes
# =============================================================================

def _mat_pow(A: MatrixC, k: int) -> MatrixC:
    if k < 0:
        raise ValueError("_mat_pow: k<0")
    n, m = shape(A)
    if n != m:
        raise ValueError("_mat_pow: matrice non carrée")
    R = identity(n)
    B = copy(A)
    while k > 0:
        if k & 1:
            R = matmul(R, B)
        B = matmul(B, B)
        k >>= 1
    return R


def _A_minus_lambdaI(A: MatrixC, lam: complex) -> MatrixC:
    n, m = shape(A)
    if n != m:
        raise ValueError("_A_minus_lambdaI: non carrée")
    M = copy(A)
    for i in range(n):
        M[i][i] -= lam
    return M


def _basis_reduce(vs: List[VectorC], tol: float = 1e-12) -> List[VectorC]:
    if not vs:
        return []
    m = len(vs[0])
    M = [v[:] for v in vs]  # (k x m)
    R, piv = rref(M, tol=tol)
    kept_rows: List[int] = []
    r = 0
    for i in range(len(R)):
        if any(abs(z) > tol for z in R[i]):
            kept_rows.append(i)
            r += 1
    return [vs[i] for i in kept_rows]


def _span_contains(basis: List[VectorC], v: VectorC, tol: float = 1e-10) -> bool:
    if not basis:
        return False
    M = [b[:] for b in basis] + [v[:]]
    return rank(M, tol=tol) == rank([b[:] for b in basis], tol=tol)


def _extend_basis(basis: List[VectorC], candidates: List[VectorC], tol: float = 1e-10) -> List[VectorC]:
    out = basis[:]
    for v in candidates:
        if not out:
            out.append(v)
            continue
        M = [b[:] for b in out]
        r0 = rank(M, tol=tol)
        r1 = rank(M + [v[:]], tol=tol)
        if r1 > r0:
            out.append(v)
    return out


def jordan_decomposition_numeric(
    A_in: List[List[float]] | List[List[complex]],
    tol: float = 1e-8,
    qr_iters: int = 800
) -> Dict[str, Any]:
    """
    Retourne une décomposition approchée A ≈ P J P^{-1},
    où J est une matrice de Jordan (bloc-diagonale) reconstruite numériquement.

    Sortie:
    {
      "P": MatrixC, "J": MatrixC, "Pinv": MatrixC,
      "eigs": List[complex],
      "groups": [{"lambda": lam, "mult_alg": k, "dims": [d1,d2,...], "blocks": [s1,s2,...]}, ...],
      "ok": bool
    }
    """
    A = to_complex(A_in)
    n, m = shape(A)
    if n != m:
        raise ValueError("jordan_decomposition_numeric: matrice non carrée")

    eigs = eigvals_qr(A, iters=qr_iters, tol=tol, shift=True)

    groups: List[List[complex]] = []
    for lam in eigs:
        placed = False
        for g in groups:
            if abs(lam - g[0]) <= 10 * tol:
                g.append(lam)
                placed = True
                break
        if not placed:
            groups.append([lam])

    P_cols: List[VectorC] = []
    J = zeros(n, n)
    cursor = 0
    group_info: List[Dict[str, Any]] = []

    for g in groups:
        lam = sum(g) / len(g)
        mult_alg = len(g)

        M1 = _A_minus_lambdaI(A, lam)
        dims: List[int] = []
        kernels: List[List[VectorC]] = []

        prev_dim = 0
        for k in range(1, n + 1):
            Mk = _mat_pow(M1, k)
            ker = nullspace_basis(Mk, tol=tol)
            ker = _basis_reduce(ker, tol=tol)
            dk = len(ker)
            dims.append(dk)
            kernels.append(ker)
            if dk == prev_dim and dk >= mult_alg:
                break
            prev_dim = dk

        # Dimensions: d_k = dim ker(M^k). Les tailles de blocs se lisent via incréments.
        # Nombre de blocs de taille >= k : b_k = d_k - d_{k-1}.
        d0 = 0
        b_ge: List[int] = []
        for dk in dims:
            b_ge.append(dk - d0)
            d0 = dk

        # Reconstruire multiset des tailles de blocs:
        # nb blocs taille exactement k = b_ge[k] - b_ge[k+1]
        # (indexation 1..K)
        K = len(b_ge)
        blocks: List[int] = []
        for k in range(1, K + 1):
            b_k = b_ge[k - 1]
            b_kp1 = b_ge[k] if k < K else 0
            nb_exact = b_k - b_kp1
            for _ in range(max(0, nb_exact)):
                blocks.append(k)
        blocks.sort(reverse=True)

        # Construction de chaînes: pour un bloc de taille s,
        # on prend v_s dans ker(M^s)\ker(M^{s-1}), puis v_{s-1}=M v_s, ..., v_1=M^{s-1} v_s.
        # Heuristique: sélectionner des générateurs indépendants.
        chains: List[List[VectorC]] = []
        used_basis: List[VectorC] = []

        for s in blocks:
            ker_s = kernels[min(s, len(kernels)) - 1]
            ker_sm1 = kernels[s - 2] if s >= 2 and (s - 2) < len(kernels) else []
            generators: List[VectorC] = []
            for v in ker_s:
                if not _span_contains(ker_sm1, v, tol=10 * tol):
                    generators.append(v)
            # choisir un générateur qui augmente l'indépendance globale
            chosen: Optional[VectorC] = None
            for v in generators:
                if not used_basis:
                    chosen = v
                    break
                r0 = rank([u[:] for u in used_basis], tol=10 * tol)
                r1 = rank([u[:] for u in used_basis] + [v[:]], tol=10 * tol)
                if r1 > r0:
                    chosen = v
                    break
            if chosen is None and ker_s:
                # fallback: tenter n'importe quel v indépendant globalement
                for v in ker_s:
                    r0 = rank([u[:] for u in used_basis], tol=10 * tol) if used_basis else 0
                    r1 = rank(([u[:] for u in used_basis] if used_basis else []) + [v[:]], tol=10 * tol)
                    if r1 > r0:
                        chosen = v
                        break
            if chosen is None:
                # échec local: on sort mais on garde ce qu'on a
                break

            chain_top = chosen
            chain: List[VectorC] = [chain_top]
            for _ in range(s - 1):
                chain.append(matvec(M1, chain[-1]))
            # chain = [v_s, v_{s-1}, ..., v_1] ; on veut colonnes (v_1,...,v_s)
            chain_cols = list(reversed(chain))

            # ajouter au global si augmente le rang
            new_used = _extend_basis(used_basis, chain_cols, tol=10 * tol)
            if len(new_used) == len(used_basis):
                # si la chaîne n'apporte rien, on la saute
                continue
            used_basis = new_used
            chains.append(chain_cols)

        # Injecter chaînes dans P et construire bloc Jordan correspondant dans J
        # Si pas assez de colonnes (pathologie numérique), on fait au mieux.
        inserted = 0
        for chain_cols in chains:
            s = len(chain_cols)
            if cursor + s > n:
                break
            # colonnes P
            for col in chain_cols:
                P_cols.append(col)
            # bloc Jordan: lam sur diagonale, 1 sur sur-diagonale
            for i in range(s):
                J[cursor + i][cursor + i] = lam
                if i < s - 1:
                    J[cursor + i][cursor + i + 1] = 1.0 + 0.0j
            cursor += s
            inserted += s

        group_info.append({
            "lambda": lam,
            "mult_alg": mult_alg,
            "dims": dims,
            "blocks_est": blocks,
            "inserted_cols": inserted
        })

    # Si P incomplète, compléter par vecteurs standard pour rendre inversible (heuristique)
    if len(P_cols) < n:
        # compléter avec e_i qui augmentent le rang
        for i in range(n):
            e = [0.0 + 0.0j] * n
            e[i] = 1.0 + 0.0j
            P_cols = _extend_basis(P_cols, [e], tol=10 * tol)
            if len(P_cols) >= n:
                break

    # Assembler P (colonnes)
    P = zeros(n, n)
    for j in range(min(n, len(P_cols))):
        col = P_cols[j]
        for i in range(n):
            P[i][j] = col[i]

    ok = True
    try:
        Pinv = inverse_via_solve(P, tol=10 * tol)
    except Exception:
        ok = False
        Pinv = identity(n)

    return {
        "P": P,
        "J": J,
        "Pinv": Pinv,
        "eigs": eigs,
        "groups": group_info,
        "ok": ok
    }


# =============================================================================
# exp(J) pour une matrice de Jordan bloc-diagonale
# exp(lam I + N) = exp(lam) * sum_{k=0}^{s-1} N^k/k!
# =============================================================================

def _expm_jordan_block(lam: complex, s: int) -> MatrixC:
    B = zeros(s, s)
    # N: super-diagonale de 1
    # exp(J) = exp(lam) * (I + N + N^2/2! + ... + N^{s-1}/(s-1)!)
    # Pour une Jordan classique, la formule donne exp(lam) sur diagonale,
    # exp(lam)/k! sur la k-ième sur-diagonale.
    el = cmath.exp(lam)
    for i in range(s):
        B[i][i] = el
    # sur-diagonales
    fact = 1.0
    for k in range(1, s):
        fact *= k  # k!
        coeff = el / fact
        for i in range(s - k):
            B[i][i + k] = coeff
    return B


def expm_from_jordan(J: MatrixC, tol: float = 1e-12) -> MatrixC:
    """
    Suppose J bloc-diagonale en blocs de Jordan (diagonale = lambdas, superdiag = 0/1).
    Reconstruit exp(J) en détectant les blocs via les 1 sur la super-diagonale.
    """
    n, m = shape(J)
    if n != m:
        raise ValueError("expm_from_jordan: non carrée")

    E = zeros(n, n)
    i = 0
    while i < n:
        lam = J[i][i]
        s = 1
        while i + s < n and abs(J[i + s - 1][i + s] - (1.0 + 0.0j)) < 10 * tol and abs(J[i + s][i + s] - lam) < 10 * tol:
            s += 1
        B = _expm_jordan_block(lam, s)
        for a in range(s):
            for b in range(s):
                E[i + a][i + b] = B[a][b]
        i += s
    return E


# =============================================================================
# Exponentielle via Dunford/Jordan: exp(A) ≈ P exp(J) P^{-1}
# =============================================================================

def expm_dunford(
    A_in: List[List[float]] | List[List[complex]],
    tol: float = 1e-8,
    qr_iters: int = 800
) -> Dict[str, Any]:
    """
    Calcule exp(A) via décomposition de Jordan numérique:
      A ≈ P J P^{-1}
      exp(A) ≈ P exp(J) P^{-1}

    Sortie:
    {
      "expm": MatrixC,
      "P": MatrixC, "J": MatrixC, "Pinv": MatrixC,
      "ok": bool,
      "jordan": (infos)
    }
    """
    jd = jordan_decomposition_numeric(A_in, tol=tol, qr_iters=qr_iters)
    P, J, Pinv = jd["P"], jd["J"], jd["Pinv"]

    EJ = expm_from_jordan(J, tol=10 * tol)
    expA = matmul(matmul(P, EJ), Pinv)

    return {
        "expm": expA,
        "P": P,
        "J": J,
        "Pinv": Pinv,
        "ok": bool(jd.get("ok", False)),
        "jordan": jd
    }


# =============================================================================
# Fallback numpy (optionnel) – pour vérifier
# =============================================================================

def expm_numpy(A_in: List[List[float]] | List[List[complex]]) -> Optional[MatrixC]:
    try:
        import numpy as np
        try:
            from scipy.linalg import expm as _expm
        except Exception:
            return None
        A = np.array(A_in, dtype=complex)
        E = _expm(A)
        return [[complex(E[i, j]) for j in range(E.shape[1])] for i in range(E.shape[0])]
    except Exception:
        return None


# =============================================================================
# MATRICES DE RÉCURRENCE / DYNAMIQUE
# =============================================================================

def fibonacci_matrix() -> Matrix:
    """
    Matrice de Fibonacci:
    [1 1]
    [1 0]
    F^n donne (F_{n+1}, F_n).
    """
    return [[1.0, 1.0],
            [1.0, 0.0]]


def companion_matrix(coeffs: List[float]) -> Matrix:
    """
    Matrice compagnon d'un polynôme
    p(x) = x^n + a_{n-1} x^{n-1} + ... + a_0
    coeffs = [a_0, a_1, ..., a_{n-1}]
    """
    n = len(coeffs)
    C = [[0.0]*n for _ in range(n)]
    for i in range(1, n):
        C[i][i-1] = 1.0
    for i in range(n):
        C[i][n-1] = -coeffs[i]
    return C


def leslie_matrix(fertility: List[float], survival: List[float]) -> Matrix:
    """
    Matrice de Leslie (dynamique de population).
    """
    n = len(fertility)
    L = [[0.0]*n for _ in range(n)]
    for j in range(n):
        L[0][j] = fertility[j]
    for i in range(1, n):
        L[i][i-1] = survival[i-1]
    return L


# =============================================================================
# ROTATIONS / TRANSFORMATIONS GÉOMÉTRIQUES
# =============================================================================

def rotation_2d(theta: float) -> Matrix:
    c = math.cos(theta)
    s = math.sin(theta)
    return [[c, -s],
            [s,  c]]


def rotation_x(theta: float) -> Matrix:
    c = math.cos(theta)
    s = math.sin(theta)
    return [[1.0, 0.0, 0.0],
            [0.0, c,  -s ],
            [0.0, s,   c ]]


def rotation_y(theta: float) -> Matrix:
    c = math.cos(theta)
    s = math.sin(theta)
    return [[ c, 0.0, s ],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c ]]


def rotation_z(theta: float) -> Matrix:
    c = math.cos(theta)
    s = math.sin(theta)
    return [[c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]]


def reflection_2d(theta: float) -> Matrix:
    """
    Réflexion par rapport à une droite d'angle theta.
    """
    c = math.cos(2*theta)
    s = math.sin(2*theta)
    return [[ c, s],
            [ s,-c]]


def projection_onto_vector(v: List[float]) -> Matrix:
    """
    Projecteur orthogonal sur span(v).
    """
    denom = sum(x*x for x in v)
    if denom == 0:
        raise ValueError("Vecteur nul")
    n = len(v)
    P = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            P[i][j] = v[i]*v[j]/denom
    return P


# =============================================================================
# MATRICES STOCHASTIQUES / MARKOV
# =============================================================================

def markov_transition(P: Matrix) -> Matrix:
    """
    Vérifie et retourne une matrice de transition de Markov.
    """
    for row in P:
        if abs(sum(row) - 1.0) > 1e-8:
            raise ValueError("Chaque ligne doit sommer à 1")
    return P


def random_walk_line(n: int) -> Matrix:
    """
    Marche aléatoire sur une ligne 0..n-1.
    """
    P = [[0.0]*n for _ in range(n)]
    for i in range(n):
        if i > 0:
            P[i][i-1] += 0.5
        if i < n-1:
            P[i][i+1] += 0.5
    return P


# =============================================================================
# MATRICES CLASSIQUES (PHYSIQUE / MATHS)
# =============================================================================

def pauli_x() -> Matrix:
    return [[0.0, 1.0],
            [1.0, 0.0]]


def pauli_y() -> MatrixC:
    return [[0.0, -1j],
            [1j,  0.0]]


def pauli_z() -> Matrix:
    return [[1.0, 0.0],
            [0.0,-1.0]]


def hadamard(n: int) -> Matrix:
    """
    Matrice de Hadamard d'ordre 2^n.
    """
    if n == 0:
        return [[1.0]]
    if n == 1:
        return [[1.0, 1.0],
                [1.0,-1.0]]
    H = hadamard(n-1)
    m = len(H)
    R = [[0.0]*(2*m) for _ in range(2*m)]
    for i in range(m):
        for j in range(m):
            R[i][j] = H[i][j]
            R[i][j+m] = H[i][j]
            R[i+m][j] = H[i][j]
            R[i+m][j+m] = -H[i][j]
    return R


# =============================================================================
# TRANSFORMATIONS DISCRÈTES
# =============================================================================

def dft_matrix(n: int) -> MatrixC:
    """
    Matrice de Fourier discrète (DFT).
    """
    W = [[0j]*n for _ in range(n)]
    omega = cmath.exp(-2j*cmath.pi/n)
    for i in range(n):
        for j in range(n):
            W[i][j] = omega**(i*j)
    return W


def dct_matrix(n: int) -> Matrix:
    """
    DCT-II (JPEG).
    """
    C = [[0.0]*n for _ in range(n)]
    for i in range(n):
        alpha = math.sqrt(1/n) if i == 0 else math.sqrt(2/n)
        for j in range(n):
            C[i][j] = alpha * math.cos(math.pi*i*(2*j+1)/(2*n))
    return C


# =============================================================================
# MATRICES DE TEST / NUMÉRIQUES
# =============================================================================

def hilbert(n: int) -> Matrix:
    """
    Matrice de Hilbert (très mal conditionnée).
    """
    return [[1.0/(i+j+1) for j in range(n)] for i in range(n)]


def vandermonde(x: List[float]) -> Matrix:
    n = len(x)
    return [[x[i]**j for j in range(n)] for i in range(n)]


def circulant(first_row: List[float]) -> Matrix:
    n = len(first_row)
    C = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            C[i][j] = first_row[(j-i) % n]
    return C




# =============================================================================
# MATRICES ÉLÉMENTAIRES (GAUSS)
# =============================================================================

def elementary_swap(i: int, j: int, n: int) -> Matrix:
    """
    Matrice élémentaire d'échange de lignes i <-> j.
    """
    E = identity(n)
    E[i][i] = 0.0
    E[j][j] = 0.0
    E[i][j] = 1.0
    E[j][i] = 1.0
    return E


def elementary_scale(i: int, alpha: float, n: int) -> Matrix:
    """
    Multiplie la ligne i par alpha.
    """
    E = identity(n)
    E[i][i] = alpha
    return E


def elementary_add(i: int, j: int, alpha: float, n: int) -> Matrix:
    """
    Ajoute alpha * ligne j à la ligne i.
    """
    E = identity(n)
    E[i][j] = alpha
    return E


# =============================================================================
# MATRICES DE PERMUTATION
# =============================================================================

def permutation_matrix(p: List[int]) -> Matrix:
    """
    Matrice de permutation associée à p :
    e_i -> e_{p[i]}
    """
    n = len(p)
    P = [[0.0]*n for _ in range(n)]
    for i in range(n):
        P[i][p[i]] = 1.0
    return P


def reverse_permutation(n: int) -> Matrix:
    """
    Matrice qui inverse l'ordre des coordonnées.
    """
    return permutation_matrix(list(reversed(range(n))))


# =============================================================================
# SYMÉTRIES ET TRANSFORMATIONS AFFINES
# =============================================================================

def symmetry_origin(n: int) -> Matrix:
    """
    Symétrie centrale: x -> -x
    """
    return homothety(-1.0, n)


def symmetry_axis_2d(theta: float) -> Matrix:
    """
    Symétrie par rapport à un axe passant par l'origine (2D).
    """
    c = math.cos(2*theta)
    s = math.sin(2*theta)
    return [[ c, s],
            [ s,-c]]


def symmetry_plane(normal: List[float]) -> Matrix:
    """
    Symétrie orthogonale par rapport à un plan (3D ou nD).
    """
    n = len(normal)
    norm2 = sum(x*x for x in normal)
    if norm2 == 0:
        raise ValueError("Vecteur normal nul")
    S = identity(n)
    for i in range(n):
        for j in range(n):
            S[i][j] -= 2*normal[i]*normal[j]/norm2
    return S


def homothety(k: float, n: int) -> Matrix:
    """
    Homothétie de rapport k.
    """
    H = [[0.0]*n for _ in range(n)]
    for i in range(n):
        H[i][i] = k
    return H


def shear_2d(kx: float = 0.0, ky: float = 0.0) -> Matrix:
    """
    Cisaillement 2D.
    """
    return [[1.0, kx],
            [ky,  1.0]]


# =============================================================================
# PROJECTEURS CLASSIQUES
# =============================================================================

def projector_coordinate(i: int, n: int) -> Matrix:
    """
    Projecteur sur l'axe e_i.
    """
    P = [[0.0]*n for _ in range(n)]
    P[i][i] = 1.0
    return P


def projector_subspace(indices: List[int], n: int) -> Matrix:
    """
    Projecteur sur le sous-espace engendré par {e_i | i in indices}.
    """
    P = [[0.0]*n for _ in range(n)]
    for i in indices:
        P[i][i] = 1.0
    return P


# =============================================================================
# MATRICES CANONIQUES (FORMES STANDARD)
# =============================================================================

def jordan_block(lambda_: float, size: int) -> Matrix:
    """
    Bloc de Jordan J(lambda, size).
    """
    J = [[0.0]*size for _ in range(size)]
    for i in range(size):
        J[i][i] = lambda_
        if i < size-1:
            J[i][i+1] = 1.0
    return J


def nilpotent_shift(n: int) -> Matrix:
    """
    Matrice nilpotente stricte (1 sur la sur-diagonale).
    """
    N = [[0.0]*n for _ in range(n)]
    for i in range(n-1):
        N[i][i+1] = 1.0
    return N


def diagonal(values: List[float]) -> Matrix:
    """
    Matrice diagonale diag(values).
    """
    n = len(values)
    D = [[0.0]*n for _ in range(n)]
    for i in range(n):
        D[i][i] = values[i]
    return D


# =============================================================================
# MATRICES D’OPÉRATEURS CLASSIQUES
# =============================================================================

def finite_difference_first(n: int, h: float = 1.0) -> Matrix:
    """
    Approximation discrète de la dérivée première.
    """
    D = [[0.0]*n for _ in range(n)]
    for i in range(n-1):
        D[i][i] = -1.0/h
        D[i][i+1] = 1.0/h
    return D


def finite_difference_second(n: int, h: float = 1.0) -> Matrix:
    """
    Approximation discrète de la dérivée seconde (Laplacien 1D).
    """
    L = [[0.0]*n for _ in range(n)]
    for i in range(n):
        L[i][i] = -2.0/(h*h)
        if i > 0:
            L[i][i-1] = 1.0/(h*h)
        if i < n-1:
            L[i][i+1] = 1.0/(h*h)
    return L


# =============================================================================
# OUTILS INTERNES MINIMAUX
# =============================================================================

def identity(n: int) -> Matrix:
    I = [[0.0]*n for _ in range(n)]
    for i in range(n):
        I[i][i] = 1.0
    return I



# =============================================================================
# MATRICES DE BASE ABSOLUES
# =============================================================================

def zero_matrix(n: int, m: int) -> Matrix:
    return [[0.0]*m for _ in range(n)]


def ones_matrix(n: int, m: int) -> Matrix:
    return [[1.0]*m for _ in range(n)]


def constant_matrix(n: int, m: int, c: float) -> Matrix:
    return [[c]*m for _ in range(n)]


def sparse_diagonal(n: int, indices: List[int], values: List[float]) -> Matrix:
    """
    Matrice diagonale creuse avec quelques coefficients non nuls.
    """
    D = [[0.0]*n for _ in range(n)]
    for i, v in zip(indices, values):
        D[i][i] = v
    return D


# =============================================================================
# MATRICES STOCHASTIQUES
# =============================================================================

def row_stochastic(P: Matrix, tol: float = 1e-10) -> bool:
    """
    Test : chaque ligne somme à 1 et coefficients >= 0.
    """
    for row in P:
        if any(x < -tol for x in row):
            return False
        if abs(sum(row) - 1.0) > tol:
            return False
    return True


def column_stochastic(P: Matrix, tol: float = 1e-10) -> bool:
    """
    Test : chaque colonne somme à 1.
    """
    n = len(P)
    m = len(P[0])
    for j in range(m):
        s = 0.0
        for i in range(n):
            if P[i][j] < -tol:
                return False
            s += P[i][j]
        if abs(s - 1.0) > tol:
            return False
    return True

import random

def random_stochastic(n: int) -> Matrix:
    """
    Matrice stochastique aléatoire (lignes).
    """
    P = [[random.random() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        s = sum(P[i])
        for j in range(n):
            P[i][j] /= s
    return P


# =============================================================================
# MATRICES DE STRUCTURE (TOEPLITZ / HANKEL)
# =============================================================================

def toeplitz(first_col: List[float], first_row: List[float]) -> Matrix:
    """
    Matrice Toeplitz : diagonales constantes.
    """
    n = len(first_col)
    m = len(first_row)
    T = [[0.0]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if j >= i:
                T[i][j] = first_row[j-i]
            else:
                T[i][j] = first_col[i-j]
    return T


def hankel(first_col: List[float], last_row: List[float]) -> Matrix:
    """
    Matrice Hankel : anti-diagonales constantes.
    """
    n = len(first_col)
    m = len(last_row)
    H = [[0.0]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            k = i + j
            if k < n:
                H[i][j] = first_col[k]
            else:
                H[i][j] = last_row[k-n+1]
    return H


# =============================================================================
# VANDERMONDE (VERSIONS)
# =============================================================================

def vandermonde_classic(x: List[float], increasing: bool = True) -> Matrix:
    """
    Vandermonde classique.
    """
    n = len(x)
    V = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            p = j if increasing else n-1-j
            V[i][j] = x[i]**p
    return V


def generalized_vandermonde(x: List[float], powers: List[int]) -> Matrix:
    """
    Vandermonde généralisée avec puissances arbitraires.
    """
    n = len(x)
    m = len(powers)
    V = [[0.0]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            V[i][j] = x[i]**powers[j]
    return V


# =============================================================================
# MATRICES DE BANDE
# =============================================================================

def band_matrix(
    n: int,
    lower: int,
    upper: int,
    value: float = 1.0
) -> Matrix:
    """
    Matrice à bande (|i-j| <= bande).
    """
    A = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(max(0, i-lower), min(n, i+upper+1)):
            A[i][j] = value
    return A


def tridiagonal(a: float, b: float, c: float, n: int) -> Matrix:
    """
    Tridiagonale :
    a sous-diagonale, b diagonale, c sur-diagonale.
    """
    T = [[0.0]*n for _ in range(n)]
    for i in range(n):
        T[i][i] = b
        if i > 0:
            T[i][i-1] = a
        if i < n-1:
            T[i][i+1] = c
    return T


# =============================================================================
# MATRICES DE GRAPHES
# =============================================================================

def adjacency_matrix(edges: List[Tuple[int, int]], n: int, directed: bool = False) -> Matrix:
    """
    Matrice d'adjacence d'un graphe.
    """
    A = [[0.0]*n for _ in range(n)]
    for i, j in edges:
        A[i][j] = 1.0
        if not directed:
            A[j][i] = 1.0
    return A


def degree_matrix(A: Matrix) -> Matrix:
    """
    Matrice des degrés.
    """
    n = len(A)
    D = [[0.0]*n for _ in range(n)]
    for i in range(n):
        D[i][i] = sum(A[i])
    return D


def laplacian_matrix(A: Matrix) -> Matrix:
    """
    Laplacien de graphe : L = D - A.
    """
    D = degree_matrix(A)
    n = len(A)
    L = [[D[i][j] - A[i][j] for j in range(n)] for i in range(n)]
    return L


def normalized_laplacian(A: Matrix, tol: float = 1e-12) -> Matrix:
    """
    Laplacien normalisé : I - D^{-1/2} A D^{-1/2}.
    """
    n = len(A)
    D = degree_matrix(A)
    L = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                L[i][j] = 1.0
            if D[i][i] > tol and D[j][j] > tol:
                L[i][j] -= A[i][j]/math.sqrt(D[i][i]*D[j][j])
    return L


# =============================================================================
# MATRICES LOGIQUES / INCIDENCE
# =============================================================================

def incidence_matrix(edges: List[Tuple[int, int]], n: int) -> Matrix:
    """
    Matrice d'incidence orientée.
    """
    m = len(edges)
    B = [[0.0]*m for _ in range(n)]
    for k, (i, j) in enumerate(edges):
        B[i][k] = 1.0
        B[j][k] = -1.0
    return B


def boolean_matrix(A: Matrix) -> List[List[int]]:
    """
    Matrice booléenne associée.
    """
    return [[1 if abs(x) > 0 else 0 for x in row] for row in A]
