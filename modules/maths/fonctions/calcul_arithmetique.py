"""
Module de calcul arithmétique
Contient les opérations de base et avancées pour la calculatrice
"""

import math
from fractions import Fraction
from typing import Tuple, List


def addition(a, b):
    """
    Additionne deux nombres
    
    Args:
        a: Premier nombre
        b: Deuxième nombre
    
    Returns:
        La somme de a et b
    """
    return a + b


def soustraction(a, b):
    """
    Soustrait b de a
    
    Args:
        a: Premier nombre
        b: Deuxième nombre
    
    Returns:
        La différence a - b
    """
    return a - b


def multiplication(a, b):
    """
    Multiplie deux nombres
    
    Args:
        a: Premier nombre
        b: Deuxième nombre
    
    Returns:
        Le produit de a et b
    """
    return a * b


def division(a, b):
    """
    Divise a par b
    
    Args:
        a: Premier nombre (numérateur)
        b: Deuxième nombre (dénominateur)
    
    Returns:
        Le quotient de a / b
    
    Raises:
        ValueError: Si b est égal à zéro
    """
    if b == 0:
        raise ValueError("Division par zéro impossible")
    return a / b


def puissance(a, b):
    """
    Élève a à la puissance b
    
    Args:
        a: Le nombre de base
        b: L'exposant
    
    Returns:
        a élevé à la puissance b
    """
    return a ** b


def modulo(a, b):
    """
    Calcule le reste de la division de a par b
    
    Args:
        a: Premier nombre
        b: Deuxième nombre
    
    Returns:
        Le reste de a modulo b
    
    Raises:
        ValueError: Si b est égal à zéro
    """
    if b == 0:
        raise ValueError("Modulo par zéro impossible")
    return a % b


def pgcd(a, b):
    """
    Calcule le Plus Grand Commun Diviseur de a et b
    
    Args:
        a: Premier nombre entier
        b: Deuxième nombre entier
    
    Returns:
        Le PGCD de a et b
    """
    return math.gcd(int(a), int(b))


def ppcm(a, b):
    """
    Calcule le Plus Petit Commun Multiple de a et b
    
    Args:
        a: Premier nombre entier
        b: Deuxième nombre entier
    
    Returns:
        Le PPCM de a et b
    """
    return abs(int(a) * int(b)) // pgcd(a, b)


def logarithme(x, base=math.e):
    """
    Calcule le logarithme de x dans une base donnée
    
    Args:
        x: Le nombre (doit être > 0)
        base: La base du logarithme (par défaut: e pour ln)
    
    Returns:
        Le logarithme de x en base donnée
    
    Raises:
        ValueError: Si x <= 0 ou base <= 0
    """
    if x <= 0:
        raise ValueError("Le logarithme nécessite un nombre positif")
    if base <= 0 or base == 1:
        raise ValueError("La base doit être positive et différente de 1")
    return math.log(x, base)


def logarithme_naturel(x):
    """
    Calcule le logarithme naturel (ln) de x
    
    Args:
        x: Le nombre (doit être > 0)
    
    Returns:
        ln(x)
    """
    return logarithme(x, math.e)


def logarithme_decimal(x):
    """
    Calcule le logarithme décimal (log10) de x
    
    Args:
        x: Le nombre (doit être > 0)
    
    Returns:
        log10(x)
    """
    return math.log10(x)


def exponentielle(x):
    """
    Calcule e^x (exponentielle de x)
    
    Args:
        x: L'exposant
    
    Returns:
        e^x
    """
    return math.exp(x)


def inverse(x):
    """
    Calcule l'inverse de x (1/x)
    
    Args:
        x: Le nombre
    
    Returns:
        1/x
    
    Raises:
        ValueError: Si x == 0
    """
    if x == 0:
        raise ValueError("Impossible de calculer l'inverse de zéro")
    return 1 / x


def fraction_irreductible(a, b):
    """
    Représente a/b sous forme de fraction irréductible
    
    Args:
        a: Numérateur
        b: Dénominateur
    
    Returns:
        Une chaîne représentant la fraction irréductible "a/b"
    
    Raises:
        ValueError: Si b == 0
    """
    if b == 0:
        raise ValueError("Le dénominateur ne peut pas être zéro")
    frac = Fraction(int(a), int(b))
    if frac.denominator == 1:
        return str(frac.numerator)
    return f"{frac.numerator}/{frac.denominator}"


def racine_carree(x):
    """
    Calcule la racine carrée de x
    
    Args:
        x: Le nombre (doit être >= 0)
    
    Returns:
        √x
    
    Raises:
        ValueError: Si x < 0
    """
    if x < 0:
        raise ValueError("Impossible de calculer la racine carrée d'un nombre négatif")
    return math.sqrt(x)


def racine_nieme(x, n):
    """
    Calcule la racine n-ième de x
    
    Args:
        x: Le nombre
        n: L'indice de la racine
    
    Returns:
        La racine n-ième de x
    """
    if n == 0:
        raise ValueError("L'indice de la racine ne peut pas être zéro")
    if x < 0 and n % 2 == 0:
        raise ValueError("Impossible de calculer une racine paire d'un nombre négatif")
    if x < 0:
        return -abs(x) ** (1 / n)
    return x ** (1 / n)


def valeur_absolue(x):
    """
    Calcule la valeur absolue de x
    
    Args:
        x: Le nombre
    
    Returns:
        |x|
    """
    return abs(x)


def factorielle(n):
    """
    Calcule la factorielle de n (n!)
    
    Args:
        n: Un entier positif ou nul
    
    Returns:
        n!
    
    Raises:
        ValueError: Si n < 0 ou n n'est pas entier
    """
    if n < 0:
        raise ValueError("La factorielle n'est définie que pour les entiers positifs")
    if not isinstance(n, int) and not n.is_integer():
        raise ValueError("La factorielle nécessite un entier")
    return math.factorial(int(n))


def arrondi_inferieur(x):
    """
    Arrondit x à l'entier inférieur (floor)
    
    Args:
        x: Le nombre
    
    Returns:
        L'entier inférieur ou égal à x
    """
    return math.floor(x)


def arrondi_superieur(x):
    """
    Arrondit x à l'entier supérieur (ceil)
    
    Args:
        x: Le nombre
    
    Returns:
        L'entier supérieur ou égal à x
    """
    return math.ceil(x)


def arrondi(x, decimales=0):
    """
    Arrondit x au nombre de décimales spécifié
    
    Args:
        x: Le nombre
        decimales: Nombre de décimales (par défaut: 0)
    
    Returns:
        x arrondi
    """
    return round(x, decimales)


def minimum(*nombres):
    """
    Retourne le minimum parmi les nombres donnés
    
    Args:
        *nombres: Liste de nombres
    
    Returns:
        Le plus petit nombre
    """
    return min(nombres)


def maximum(*nombres):
    """
    Retourne le maximum parmi les nombres donnés
    
    Args:
        *nombres: Liste de nombres
    
    Returns:
        Le plus grand nombre
    """
    return max(nombres)


def moyenne(nombres):
    """
    Calcule la moyenne arithmétique d'une liste de nombres
    
    Args:
        nombres: Liste de nombres
    
    Returns:
        La moyenne
    
    Raises:
        ValueError: Si la liste est vide
    """
    if not nombres:
        raise ValueError("Impossible de calculer la moyenne d'une liste vide")
    return sum(nombres) / len(nombres)


def somme(nombres):
    """
    Calcule la somme d'une liste de nombres
    
    Args:
        nombres: Liste de nombres
    
    Returns:
        La somme de tous les nombres
    """
    return sum(nombres)


def produit(nombres):
    """
    Calcule le produit d'une liste de nombres
    
    Args:
        nombres: Liste de nombres
    
    Returns:
        Le produit de tous les nombres
    """
    resultat = 1
    for n in nombres:
        resultat *= n
    return resultat


def pourcentage(valeur, total):
    """
    Calcule quel pourcentage représente valeur par rapport au total
    
    Args:
        valeur: La valeur partielle
        total: Le total
    
    Returns:
        Le pourcentage
    
    Raises:
        ValueError: Si total == 0
    """
    if total == 0:
        raise ValueError("Le total ne peut pas être zéro")
    return (valeur / total) * 100


def calculer_pourcentage(nombre, pourcent):
    """
    Calcule pourcent% de nombre
    
    Args:
        nombre: Le nombre de base
        pourcent: Le pourcentage à calculer
    
    Returns:
        pourcent% de nombre
    """
    return (nombre * pourcent) / 100


def variation_pourcentage(valeur_initiale, valeur_finale):
    """
    Calcule la variation en pourcentage entre deux valeurs
    
    Args:
        valeur_initiale: La valeur de départ
        valeur_finale: La valeur d'arrivée
    
    Returns:
        Le pourcentage de variation
    
    Raises:
        ValueError: Si valeur_initiale == 0
    """
    if valeur_initiale == 0:
        raise ValueError("La valeur initiale ne peut pas être zéro")
    return ((valeur_finale - valeur_initiale) / valeur_initiale) * 100


def division_entiere(a, b):
    """
    Effectue la division entière de a par b
    
    Args:
        a: Le dividende
        b: Le diviseur
    
    Returns:
        Le quotient entier
    
    Raises:
        ValueError: Si b == 0
    """
    if b == 0:
        raise ValueError("Division par zéro impossible")
    return a // b


def combinaison(n, k):
    """
    Calcule C(n,k) = n! / (k! * (n-k)!)
    
    Args:
        n: Nombre total d'éléments
        k: Nombre d'éléments à choisir
    
    Returns:
        Le nombre de combinaisons
    """
    return math.comb(int(n), int(k))


def arrangement(n, k):
    """
    Calcule A(n,k) = n! / (n-k)!
    
    Args:
        n: Nombre total d'éléments
        k: Nombre d'éléments à arranger
    
    Returns:
        Le nombre d'arrangements
    """
    return math.perm(int(n), int(k))


def est_premier(n):
    """
    Vérifie si n est un nombre premier
    
    Args:
        n: Le nombre à tester
    
    Returns:
        True si n est premier, False sinon
    """
    n = int(n)
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def fibonacci(n):
    """
    Calcule le n-ième nombre de Fibonacci
    
    Args:
        n: L'indice (commence à 0)
    
    Returns:
        Le n-ième nombre de Fibonacci
    """
    n = int(n)
    if n < 0:
        raise ValueError("L'indice doit être positif")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def diviseurs(n):
    """
    Trouve tous les diviseurs d'un nombre
    
    Args:
        n: Le nombre
    
    Returns:
        Liste des diviseurs
    """
    n = abs(int(n))
    if n == 0:
        return []
    divs = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)


def somme_diviseurs(n):
    """
    Calcule la somme de tous les diviseurs de n
    
    Args:
        n: Le nombre
    
    Returns:
        La somme des diviseurs
    """
    return sum(diviseurs(n))


def signe(x):
    """
    Retourne le signe de x
    
    Args:
        x: Le nombre
    
    Returns:
        1 si x > 0, -1 si x < 0, 0 si x == 0
    """
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def est_pair(n):
    """
    Vérifie si n est pair
    
    Args:
        n: Le nombre
    
    Returns:
        True si n est pair, False sinon
    """
    return int(n) % 2 == 0


def est_impair(n):
    """
    Vérifie si n est impair
    
    Args:
        n: Le nombre
    
    Returns:
        True si n est impair, False sinon
    """
    return int(n) % 2 != 0


def mediane(nombres):
    """
    Calcule la médiane d'une liste de nombres
    
    Args:
        nombres: Liste de nombres
    
    Returns:
        La médiane
    
    Raises:
        ValueError: Si la liste est vide
    """
    if not nombres:
        raise ValueError("Impossible de calculer la médiane d'une liste vide")
    tri = sorted(nombres)
    n = len(tri)
    if n % 2 == 0:
        return (tri[n // 2 - 1] + tri[n // 2]) / 2
    else:
        return tri[n // 2]


def ecart_type(nombres):
    """
    Calcule l'écart-type d'une liste de nombres
    
    Args:
        nombres: Liste de nombres
    
    Returns:
        L'écart-type
    """
    if not nombres:
        raise ValueError("Impossible de calculer l'écart-type d'une liste vide")
    moy = moyenne(nombres)
    variance = sum((x - moy) ** 2 for x in nombres) / len(nombres)
    return math.sqrt(variance)


def variance(nombres):
    """
    Calcule la variance d'une liste de nombres
    
    Args:
        nombres: Liste de nombres
    
    Returns:
        La variance
    """
    if not nombres:
        raise ValueError("Impossible de calculer la variance d'une liste vide")
    moy = moyenne(nombres)
    return sum((x - moy) ** 2 for x in nombres) / len(nombres)


def moyenne_harmonique(nombres):
    """
    Calcule la moyenne harmonique d'une liste de nombres
    
    Args:
        nombres: Liste de nombres (tous non nuls)
    
    Returns:
        La moyenne harmonique
    
    Raises:
        ValueError: Si la liste est vide ou contient zéro
    """
    if not nombres:
        raise ValueError("Impossible de calculer la moyenne harmonique d'une liste vide")
    if any(x == 0 for x in nombres):
        raise ValueError("La moyenne harmonique ne peut pas être calculée avec des zéros")
    return len(nombres) / sum(1 / x for x in nombres)


def moyenne_geometrique(nombres):
    """
    Calcule la moyenne géométrique d'une liste de nombres
    
    Args:
        nombres: Liste de nombres positifs
    
    Returns:
        La moyenne géométrique
    
    Raises:
        ValueError: Si la liste est vide ou contient des nombres négatifs
    """
    if not nombres:
        raise ValueError("Impossible de calculer la moyenne géométrique d'une liste vide")
    if any(x < 0 for x in nombres):
        raise ValueError("La moyenne géométrique nécessite des nombres positifs")
    produit_total = produit(nombres)
    return produit_total ** (1 / len(nombres))


def moyenne_quadratique(nombres):
    """
    Calcule la moyenne quadratique (RMS - Root Mean Square) d'une liste de nombres
    
    Args:
        nombres: Liste de nombres
    
    Returns:
        La moyenne quadratique
    """
    if not nombres:
        raise ValueError("Impossible de calculer la moyenne quadratique d'une liste vide")
    return math.sqrt(sum(x ** 2 for x in nombres) / len(nombres))


def somme_carres(nombres):
    """
    Calcule la somme des carrés d'une liste de nombres
    
    Args:
        nombres: Liste de nombres
    
    Returns:
        La somme des carrés
    """
    return sum(x ** 2 for x in nombres)


def distance_euclidienne(point1, point2):
    """
    Calcule la distance euclidienne entre deux points
    
    Args:
        point1: Tuple/liste de coordonnées du premier point
        point2: Tuple/liste de coordonnées du deuxième point
    
    Returns:
        La distance euclidienne
    
    Raises:
        ValueError: Si les points n'ont pas la même dimension
    """
    if len(point1) != len(point2):
        raise ValueError("Les points doivent avoir la même dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))


def norme(vecteur):
    """
    Calcule la norme euclidienne d'un vecteur
    
    Args:
        vecteur: Liste de coordonnées
    
    Returns:
        La norme du vecteur
    """
    return math.sqrt(sum(x ** 2 for x in vecteur))


def produit_scalaire(vecteur1, vecteur2):
    """
    Calcule le produit scalaire de deux vecteurs
    
    Args:
        vecteur1: Premier vecteur
        vecteur2: Deuxième vecteur
    
    Returns:
        Le produit scalaire
    
    Raises:
        ValueError: Si les vecteurs n'ont pas la même dimension
    """
    if len(vecteur1) != len(vecteur2):
        raise ValueError("Les vecteurs doivent avoir la même dimension")
    return sum(x * y for x, y in zip(vecteur1, vecteur2))


def angle_entre_vecteurs(vecteur1, vecteur2):
    """
    Calcule l'angle en radians entre deux vecteurs
    
    Args:
        vecteur1: Premier vecteur
        vecteur2: Deuxième vecteur
    
    Returns:
        L'angle en radians
    """
    ps = produit_scalaire(vecteur1, vecteur2)
    norme1 = norme(vecteur1)
    norme2 = norme(vecteur2)
    if norme1 == 0 or norme2 == 0:
        raise ValueError("Les vecteurs ne peuvent pas être nuls")
    cos_angle = ps / (norme1 * norme2)
    # Limiter à [-1, 1] pour éviter les erreurs d'arrondi
    cos_angle = max(-1, min(1, cos_angle))
    return math.acos(cos_angle)


def covariance(x_valeurs, y_valeurs):
    """
    Calcule la covariance entre deux ensembles de données
    
    Args:
        x_valeurs: Premier ensemble de données
        y_valeurs: Deuxième ensemble de données
    
    Returns:
        La covariance
    
    Raises:
        ValueError: Si les listes n'ont pas la même taille
    """
    if len(x_valeurs) != len(y_valeurs):
        raise ValueError("Les deux ensembles doivent avoir la même taille")
    if not x_valeurs:
        raise ValueError("Les ensembles ne peuvent pas être vides")
    
    x_moy = moyenne(x_valeurs)
    y_moy = moyenne(y_valeurs)
    return sum((x - x_moy) * (y - y_moy) for x, y in zip(x_valeurs, y_valeurs)) / len(x_valeurs)


def coefficient_correlation(x_valeurs, y_valeurs):
    """
    Calcule le coefficient de corrélation de Pearson entre deux ensembles
    
    Args:
        x_valeurs: Premier ensemble de données
        y_valeurs: Deuxième ensemble de données
    
    Returns:
        Le coefficient de corrélation (entre -1 et 1)
    """
    cov = covariance(x_valeurs, y_valeurs)
    ecart_x = ecart_type(x_valeurs)
    ecart_y = ecart_type(y_valeurs)
    
    if ecart_x == 0 or ecart_y == 0:
        raise ValueError("L'écart-type ne peut pas être nul")
    
    return cov / (ecart_x * ecart_y)


def coefficient_binomial(n, k):
    """
    Calcule le coefficient binomial C(n,k) - alias de combinaison
    
    Args:
        n: Nombre total
        k: Nombre à choisir
    
    Returns:
        C(n,k)
    """
    return combinaison(n, k)


def trinome_discriminant(a, b, c):
    """
    Calcule le discriminant d'un trinôme ax² + bx + c
    
    Args:
        a: Coefficient de x²
        b: Coefficient de x
        c: Terme constant
    
    Returns:
        Le discriminant Δ = b² - 4ac
    """
    return b ** 2 - 4 * a * c


def triangle_aire_heron(a, b, c):
    """
    Calcule l'aire d'un triangle avec la formule de Héron
    
    Args:
        a: Premier côté
        b: Deuxième côté
        c: Troisième côté
    
    Returns:
        L'aire du triangle
    
    Raises:
        ValueError: Si les côtés ne forment pas un triangle valide
    """
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError("Les côtés doivent être positifs")
    if a + b <= c or a + c <= b or b + c <= a:
        raise ValueError("Les côtés ne forment pas un triangle valide")
    
    s = (a + b + c) / 2  # demi-périmètre
    return math.sqrt(s * (s - a) * (s - b) * (s - c))


def distance_manhattan(point1, point2):
    """
    Calcule la distance de Manhattan entre deux points
    
    Args:
        point1: Tuple/liste de coordonnées du premier point
        point2: Tuple/liste de coordonnées du deuxième point
    
    Returns:
        La distance de Manhattan
    """
    if len(point1) != len(point2):
        raise ValueError("Les points doivent avoir la même dimension")
    return sum(abs(x - y) for x, y in zip(point1, point2))


def troncature(x, decimales=0):
    """
    Tronque un nombre à n décimales (sans arrondir)
    
    Args:
        x: Le nombre
        decimales: Nombre de décimales à garder
    
    Returns:
        Le nombre tronqué
    """
    facteur = 10 ** decimales
    return int(x * facteur) / facteur


def reste_chinois(remainders, moduli):
    """
    Résout un système de congruences (Théorème des restes chinois simplifié)
    
    Args:
        remainders: Liste des restes
        moduli: Liste des modules (doivent être premiers entre eux deux à deux)
    
    Returns:
        La solution x
    """
    if len(remainders) != len(moduli):
        raise ValueError("Les listes doivent avoir la même taille")
    
    total = 0
    prod = produit(moduli)
    
    for remainder, mod in zip(remainders, moduli):
        p = prod // mod
        total += remainder * inverse_modulaire(p, mod) * p
    
    return total % prod


def inverse_modulaire(a, m):
    """
    Calcule l'inverse modulaire de a modulo m (a * x ≡ 1 (mod m))
    
    Args:
        a: Le nombre
        m: Le module
    
    Returns:
        L'inverse modulaire
    """
    if pgcd(a, m) != 1:
        raise ValueError("L'inverse modulaire n'existe pas")
    
    # Algorithme d'Euclide étendu
    def egcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = egcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    _, x, _ = egcd(a % m, m)
    return (x % m + m) % m


def nombre_or():
    """
    Retourne le nombre d'or (φ = (1 + √5) / 2)
    
    Returns:
        Le nombre d'or
    """
    return (1 + math.sqrt(5)) / 2


def constante_euler():
    """
    Retourne la constante d'Euler-Mascheroni (γ ≈ 0.5772)
    
    Returns:
        Une approximation de la constante d'Euler
    """
    return 0.5772156649015329


def interpolation_lineaire(x, x0, y0, x1, y1):
    """
    Effectue une interpolation linéaire
    
    Args:
        x: Point où évaluer
        x0, y0: Premier point connu
        x1, y1: Deuxième point connu
    
    Returns:
        La valeur interpolée en x
    """
    if x0 == x1:
        raise ValueError("x0 et x1 doivent être différents")
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def suite_arithmetique_terme(u0, r, n):
    """
    Calcule le n-ième terme d'une suite arithmétique
    
    Args:
        u0: Premier terme
        r: Raison
        n: Indice du terme
    
    Returns:
        Le terme u_n = u0 + n * r
    """
    return u0 + n * r


def suite_geometrique_terme(u0, q, n):
    """
    Calcule le n-ième terme d'une suite géométrique
    
    Args:
        u0: Premier terme
        q: Raison
        n: Indice du terme
    
    Returns:
        Le terme u_n = u0 * q^n
    """
    return u0 * (q ** n)


def somme_suite_arithmetique(u0, r, n):
    """
    Calcule la somme des n premiers termes d'une suite arithmétique
    
    Args:
        u0: Premier terme
        r: Raison
        n: Nombre de termes
    
    Returns:
        La somme S_n = n * (2*u0 + (n-1)*r) / 2
    """
    return n * (2 * u0 + (n - 1) * r) / 2


def somme_suite_geometrique(u0, q, n):
    """
    Calcule la somme des n premiers termes d'une suite géométrique
    
    Args:
        u0: Premier terme
        q: Raison
        n: Nombre de termes
    
    Returns:
        La somme S_n
    """
    if q == 1:
        return n * u0
    return u0 * (1 - q ** n) / (1 - q)


def somme_cubes(nombres):
    """
    Calcule la somme des cubes d'une liste de nombres
    
    Args:
        nombres: Liste de nombres
    
    Returns:
        La somme des cubes
    """
    return sum(x ** 3 for x in nombres)


def somme_n_cubes(n):
    """
    Calcule la somme des cubes des n premiers entiers: 1³ + 2³ + ... + n³
    Formule: [n(n+1)/2]²
    
    Args:
        n: Nombre d'entiers
    
    Returns:
        La somme des cubes
    """
    return ((n * (n + 1)) // 2) ** 2


def somme_n_carres(n):
    """
    Calcule la somme des carrés des n premiers entiers: 1² + 2² + ... + n²
    Formule: n(n+1)(2n+1)/6
    
    Args:
        n: Nombre d'entiers
    
    Returns:
        La somme des carrés
    """
    return (n * (n + 1) * (2 * n + 1)) // 6


def somme_n_entiers(n):
    """
    Calcule la somme des n premiers entiers: 1 + 2 + ... + n
    Formule: n(n+1)/2
    
    Args:
        n: Nombre d'entiers
    
    Returns:
        La somme
    """
    return (n * (n + 1)) // 2


def aire_carre(cote):
    """
    Calcule l'aire d'un carré
    
    Args:
        cote: Longueur du côté
    
    Returns:
        L'aire du carré
    """
    return cote ** 2


def aire_rectangle(longueur, largeur):
    """
    Calcule l'aire d'un rectangle
    
    Args:
        longueur: Longueur du rectangle
        largeur: Largeur du rectangle
    
    Returns:
        L'aire du rectangle
    """
    return longueur * largeur


def aire_triangle(base, hauteur):
    """
    Calcule l'aire d'un triangle
    
    Args:
        base: Base du triangle
        hauteur: Hauteur du triangle
    
    Returns:
        L'aire du triangle
    """
    return (base * hauteur) / 2


def aire_cercle(rayon):
    """
    Calcule l'aire d'un cercle
    
    Args:
        rayon: Rayon du cercle
    
    Returns:
        L'aire du cercle
    """
    return math.pi * rayon ** 2


def aire_disque(rayon):
    """
    Calcule l'aire d'un disque (alias de aire_cercle)
    
    Args:
        rayon: Rayon du disque
    
    Returns:
        L'aire du disque
    """
    return aire_cercle(rayon)


def aire_ellipse(demi_grand_axe, demi_petit_axe):
    """
    Calcule l'aire d'une ellipse
    
    Args:
        demi_grand_axe: Demi grand axe (a)
        demi_petit_axe: Demi petit axe (b)
    
    Returns:
        L'aire de l'ellipse (π * a * b)
    """
    return math.pi * demi_grand_axe * demi_petit_axe


def aire_trapeze(base1, base2, hauteur):
    """
    Calcule l'aire d'un trapèze
    
    Args:
        base1: Première base
        base2: Deuxième base
        hauteur: Hauteur du trapèze
    
    Returns:
        L'aire du trapèze
    """
    return ((base1 + base2) * hauteur) / 2


def aire_parallelogramme(base, hauteur):
    """
    Calcule l'aire d'un parallélogramme
    
    Args:
        base: Base du parallélogramme
        hauteur: Hauteur du parallélogramme
    
    Returns:
        L'aire du parallélogramme
    """
    return base * hauteur


def aire_losange(diagonale1, diagonale2):
    """
    Calcule l'aire d'un losange
    
    Args:
        diagonale1: Première diagonale
        diagonale2: Deuxième diagonale
    
    Returns:
        L'aire du losange
    """
    return (diagonale1 * diagonale2) / 2


def aire_polygone_regulier(nombre_cotes, longueur_cote):
    """
    Calcule l'aire d'un polygone régulier
    
    Args:
        nombre_cotes: Nombre de côtés du polygone
        longueur_cote: Longueur d'un côté
    
    Returns:
        L'aire du polygone régulier
    """
    if nombre_cotes < 3:
        raise ValueError("Un polygone doit avoir au moins 3 côtés")
    return (nombre_cotes * longueur_cote ** 2) / (4 * math.tan(math.pi / nombre_cotes))


def aire_secteur_circulaire(rayon, angle_radians):
    """
    Calcule l'aire d'un secteur circulaire
    
    Args:
        rayon: Rayon du cercle
        angle_radians: Angle du secteur en radians
    
    Returns:
        L'aire du secteur
    """
    return (rayon ** 2 * angle_radians) / 2


def aire_couronne(rayon_exterieur, rayon_interieur):
    """
    Calcule l'aire d'une couronne (anneau)
    
    Args:
        rayon_exterieur: Rayon extérieur
        rayon_interieur: Rayon intérieur
    
    Returns:
        L'aire de la couronne
    """
    if rayon_interieur >= rayon_exterieur:
        raise ValueError("Le rayon intérieur doit être inférieur au rayon extérieur")
    return math.pi * (rayon_exterieur ** 2 - rayon_interieur ** 2)


def perimetre_carre(cote):
    """
    Calcule le périmètre d'un carré
    
    Args:
        cote: Longueur du côté
    
    Returns:
        Le périmètre
    """
    return 4 * cote


def perimetre_rectangle(longueur, largeur):
    """
    Calcule le périmètre d'un rectangle
    
    Args:
        longueur: Longueur
        largeur: Largeur
    
    Returns:
        Le périmètre
    """
    return 2 * (longueur + largeur)


def perimetre_cercle(rayon):
    """
    Calcule le périmètre (circonférence) d'un cercle
    
    Args:
        rayon: Rayon du cercle
    
    Returns:
        Le périmètre
    """
    return 2 * math.pi * rayon


def circonference(rayon):
    """
    Calcule la circonférence d'un cercle (alias de perimetre_cercle)
    
    Args:
        rayon: Rayon du cercle
    
    Returns:
        La circonférence
    """
    return perimetre_cercle(rayon)


def perimetre_triangle(a, b, c):
    """
    Calcule le périmètre d'un triangle
    
    Args:
        a, b, c: Longueurs des trois côtés
    
    Returns:
        Le périmètre
    """
    return a + b + c


def perimetre_polygone(cotes):
    """
    Calcule le périmètre d'un polygone
    
    Args:
        cotes: Liste des longueurs des côtés
    
    Returns:
        Le périmètre
    """
    return sum(cotes)


# ============================================================================
# CALCUL FRACTIONNAIRE SYMBOLIQUE
# ============================================================================

def simplifier_fraction(numerateur, denominateur):
    """
    Simplifie une fraction en divisant par le PGCD
    
    Args:
        numerateur: Le numérateur
        denominateur: Le dénominateur
    
    Returns:
        Tuple (numérateur_simplifié, dénominateur_simplifié)
    
    Raises:
        ValueError: Si le dénominateur est zéro
    """
    if denominateur == 0:
        raise ValueError("Le dénominateur ne peut pas être zéro")
    
    # Gérer le signe
    if denominateur < 0:
        numerateur = -numerateur
        denominateur = -denominateur
    
    pgcd_val = pgcd(abs(numerateur), abs(denominateur))
    return (numerateur // pgcd_val, denominateur // pgcd_val)


def addition_fractions(num1, den1, num2, den2):
    """
    Additionne deux fractions: a/b + c/d
    
    Args:
        num1, den1: Numérateur et dénominateur de la première fraction
        num2, den2: Numérateur et dénominateur de la deuxième fraction
    
    Returns:
        Tuple (numérateur, dénominateur) de la fraction simplifiée
    """
    numerateur = num1 * den2 + num2 * den1
    denominateur = den1 * den2
    return simplifier_fraction(numerateur, denominateur)


def soustraction_fractions(num1, den1, num2, den2):
    """
    Soustrait deux fractions: a/b - c/d
    
    Args:
        num1, den1: Numérateur et dénominateur de la première fraction
        num2, den2: Numérateur et dénominateur de la deuxième fraction
    
    Returns:
        Tuple (numérateur, dénominateur) de la fraction simplifiée
    """
    numerateur = num1 * den2 - num2 * den1
    denominateur = den1 * den2
    return simplifier_fraction(numerateur, denominateur)


def multiplication_fractions(num1, den1, num2, den2):
    """
    Multiplie deux fractions: (a/b) * (c/d)
    
    Args:
        num1, den1: Numérateur et dénominateur de la première fraction
        num2, den2: Numérateur et dénominateur de la deuxième fraction
    
    Returns:
        Tuple (numérateur, dénominateur) de la fraction simplifiée
    """
    numerateur = num1 * num2
    denominateur = den1 * den2
    return simplifier_fraction(numerateur, denominateur)


def division_fractions(num1, den1, num2, den2):
    """
    Divise deux fractions: (a/b) / (c/d) = (a/b) * (d/c)
    
    Args:
        num1, den1: Numérateur et dénominateur de la première fraction
        num2, den2: Numérateur et dénominateur de la deuxième fraction
    
    Returns:
        Tuple (numérateur, dénominateur) de la fraction simplifiée
    
    Raises:
        ValueError: Si le numérateur de la deuxième fraction est zéro
    """
    if num2 == 0:
        raise ValueError("Division par zéro impossible")
    return multiplication_fractions(num1, den1, den2, num2)


def meme_denominateur(*fractions):
    """
    Met plusieurs fractions au même dénominateur (PPCM des dénominateurs)
    
    Args:
        *fractions: Tuples (numérateur, dénominateur)
    
    Returns:
        Liste de tuples (numérateur_modifié, dénominateur_commun)
    """
    if not fractions:
        return []
    
    # Calculer le PPCM de tous les dénominateurs
    denominateurs = [den for _, den in fractions]
    ppcm_val = denominateurs[0]
    for den in denominateurs[1:]:
        ppcm_val = ppcm(ppcm_val, den)
    
    # Adapter chaque fraction
    resultat = []
    for num, den in fractions:
        facteur = ppcm_val // den
        resultat.append((num * facteur, ppcm_val))
    
    return resultat


def decimal_vers_fraction(decimal_val, precision=1e-10):
    """
    Convertit un nombre décimal en fraction
    
    Args:
        decimal_val: Le nombre décimal
        precision: Précision de la conversion
    
    Returns:
        Tuple (numérateur, dénominateur)
    """
    frac = Fraction(decimal_val).limit_denominator()
    return (frac.numerator, frac.denominator)


def fraction_vers_decimal(numerateur, denominateur):
    """
    Convertit une fraction en nombre décimal
    
    Args:
        numerateur: Le numérateur
        denominateur: Le dénominateur
    
    Returns:
        Le nombre décimal
    
    Raises:
        ValueError: Si le dénominateur est zéro
    """
    if denominateur == 0:
        raise ValueError("Le dénominateur ne peut pas être zéro")
    return numerateur / denominateur


def fraction_mixte(numerateur, denominateur):
    """
    Convertit une fraction impropre en nombre mixte (partie entière + fraction)
    
    Args:
        numerateur: Le numérateur
        denominateur: Le dénominateur
    
    Returns:
        Tuple (partie_entière, numérateur_reste, dénominateur)
    """
    if denominateur == 0:
        raise ValueError("Le dénominateur ne peut pas être zéro")
    
    partie_entiere = numerateur // denominateur
    reste = numerateur % denominateur
    return (partie_entiere, reste, denominateur)


def fraction_impropre(partie_entiere, numerateur, denominateur):
    """
    Convertit un nombre mixte en fraction impropre
    
    Args:
        partie_entiere: La partie entière
        numerateur: Le numérateur de la partie fractionnaire
        denominateur: Le dénominateur
    
    Returns:
        Tuple (numérateur_total, dénominateur)
    """
    numerateur_total = partie_entiere * denominateur + numerateur
    return (numerateur_total, denominateur)


def comparer_fractions(num1, den1, num2, den2):
    """
    Compare deux fractions
    
    Args:
        num1, den1: Première fraction
        num2, den2: Deuxième fraction
    
    Returns:
        -1 si fraction1 < fraction2, 0 si égales, 1 si fraction1 > fraction2
    """
    # Mettre au même dénominateur
    val1 = num1 * den2
    val2 = num2 * den1
    
    if val1 < val2:
        return -1
    elif val1 > val2:
        return 1
    else:
        return 0


def puissance_fraction(numerateur, denominateur, exposant):
    """
    Élève une fraction à une puissance
    
    Args:
        numerateur: Le numérateur
        denominateur: Le dénominateur
        exposant: L'exposant (entier)
    
    Returns:
        Tuple (numérateur, dénominateur) du résultat simplifié
    """
    if exposant == 0:
        return (1, 1)
    elif exposant > 0:
        num_res = numerateur ** exposant
        den_res = denominateur ** exposant
    else:  # exposant négatif
        num_res = denominateur ** abs(exposant)
        den_res = numerateur ** abs(exposant)
    
    return simplifier_fraction(num_res, den_res)


# ============================================================================
# DÉVELOPPEMENT ET FACTORISATION
# ============================================================================

def developper_carre_binome(a, b):
    """
    Développe (a + b)²
    Identité: (a + b)² = a² + 2ab + b²
    
    Args:
        a, b: Les coefficients
    
    Returns:
        Tuple (a², 2ab, b²) représentant les coefficients
    """
    return (a**2, 2*a*b, b**2)


def developper_carre_difference(a, b):
    """
    Développe (a - b)²
    Identité: (a - b)² = a² - 2ab + b²
    
    Args:
        a, b: Les coefficients
    
    Returns:
        Tuple (a², -2ab, b²) représentant les coefficients
    """
    return (a**2, -2*a*b, b**2)


def developper_difference_carres(a, b):
    """
    Développe (a + b)(a - b)
    Identité: (a + b)(a - b) = a² - b²
    
    Args:
        a, b: Les coefficients
    
    Returns:
        Tuple (a², -b²) représentant a² - b²
    """
    return (a**2, -b**2)


def developper_cube_somme(a, b):
    """
    Développe (a + b)³
    Identité: (a + b)³ = a³ + 3a²b + 3ab² + b³
    
    Args:
        a, b: Les coefficients
    
    Returns:
        Tuple (a³, 3a²b, 3ab², b³)
    """
    return (a**3, 3*a**2*b, 3*a*b**2, b**3)


def developper_cube_difference(a, b):
    """
    Développe (a - b)³
    Identité: (a - b)³ = a³ - 3a²b + 3ab² - b³
    
    Args:
        a, b: Les coefficients
    
    Returns:
        Tuple (a³, -3a²b, 3ab², -b³)
    """
    return (a**3, -3*a**2*b, 3*a*b**2, -b**3)


def developper_somme_cubes(a, b):
    """
    Factorise a³ + b³ = (a + b)(a² - ab + b²)
    Retourne les facteurs
    
    Args:
        a, b: Les valeurs
    
    Returns:
        Tuple ((a+b), (a²-ab+b²))
    """
    facteur1 = a + b
    facteur2 = a**2 - a*b + b**2
    return (facteur1, facteur2)


def developper_difference_cubes(a, b):
    """
    Factorise a³ - b³ = (a - b)(a² + ab + b²)
    Retourne les facteurs
    
    Args:
        a, b: Les valeurs
    
    Returns:
        Tuple ((a-b), (a²+ab+b²))
    """
    facteur1 = a - b
    facteur2 = a**2 + a*b + b**2
    return (facteur1, facteur2)


def developper_binome_newton(a, b, n):
    """
    Développe (a + b)ⁿ en utilisant le binôme de Newton
    Retourne les coefficients
    
    Args:
        a, b: Les termes du binôme
        n: L'exposant
    
    Returns:
        Liste des termes du développement
    """
    termes = []
    for k in range(n + 1):
        coeff = combinaison(n, k) * (a ** (n - k)) * (b ** k)
        termes.append(coeff)
    return termes


def developper_produit_trinomes(a1, b1, c1, a2, b2, c2):
    """
    Développe (a1x² + b1x + c1)(a2x² + b2x + c2)
    
    Args:
        a1, b1, c1: Coefficients du premier trinôme
        a2, b2, c2: Coefficients du deuxième trinôme
    
    Returns:
        Tuple (coeff_x⁴, coeff_x³, coeff_x², coeff_x, terme_constant)
    """
    x4 = a1 * a2
    x3 = a1 * b2 + b1 * a2
    x2 = a1 * c2 + b1 * b2 + c1 * a2
    x1 = b1 * c2 + c1 * b2
    x0 = c1 * c2
    return (x4, x3, x2, x1, x0)


def factoriser_trinome_simple(a, b, c):
    """
    Factorise un trinôme ax² + bx + c si possible
    
    Args:
        a: Coefficient de x²
        b: Coefficient de x
        c: Terme constant
    
    Returns:
        Tuple de tuples représentant les facteurs ou None si non factorisable sur ℝ
        Format: ((a1, b1), (a2, b2)) pour (a1x + b1)(a2x + b2)
    """
    delta = trinome_discriminant(a, b, c)
    
    if delta < 0:
        return None  # Pas de factorisation réelle
    elif delta == 0:
        # Une racine double
        x0 = -b / (2 * a)
        return ((a, -a * x0), (1, 0))
    else:
        # Deux racines distinctes
        racine_delta = math.sqrt(delta)
        x1 = (-b + racine_delta) / (2 * a)
        x2 = (-b - racine_delta) / (2 * a)
        return ((a, -a * x1), (1, -x2))


def racines_trinome(a, b, c):
    """
    Calcule les racines d'un trinôme ax² + bx + c = 0
    
    Args:
        a: Coefficient de x²
        b: Coefficient de x
        c: Terme constant
    
    Returns:
        Liste des racines réelles ou tuple de complexes si Δ < 0
    """
    if a == 0:
        if b == 0:
            return [] if c != 0 else "infinité de solutions"
        return [-c / b]
    
    delta = trinome_discriminant(a, b, c)
    
    if delta < 0:
        # Racines complexes
        partie_reelle = -b / (2 * a)
        partie_imag = math.sqrt(-delta) / (2 * a)
        return [(partie_reelle, partie_imag), (partie_reelle, -partie_imag)]
    elif delta == 0:
        # Une racine double
        return [-b / (2 * a)]
    else:
        # Deux racines distinctes
        racine_delta = math.sqrt(delta)
        x1 = (-b + racine_delta) / (2 * a)
        x2 = (-b - racine_delta) / (2 * a)
        return [x1, x2]


def forme_canonique_trinome(a, b, c):
    """
    Convertit un trinôme ax² + bx + c en forme canonique a(x - α)² + β
    
    Args:
        a: Coefficient de x²
        b: Coefficient de x
        c: Terme constant
    
    Returns:
        Tuple (a, α, β) pour la forme a(x - α)² + β
    """
    if a == 0:
        raise ValueError("Le coefficient a doit être non nul")
    
    alpha = -b / (2 * a)
    beta = c - (b**2) / (4 * a)
    
    return (a, alpha, beta)


def forme_canonique_vers_developpee(a, alpha, beta):
    """
    Convertit de la forme canonique a(x - α)² + β vers ax² + bx + c
    
    Args:
        a: Coefficient
        alpha: Centre de la parabole
        beta: Ordonnée du sommet
    
    Returns:
        Tuple (a, b, c) de la forme développée
    """
    b = -2 * a * alpha
    c = a * alpha**2 + beta
    return (a, b, c)


# ============================================================================
# IDENTITÉS REMARQUABLES
# ============================================================================

def identite_somme_carres(a, b):
    """
    (a + b)² = a² + 2ab + b²
    
    Args:
        a, b: Les valeurs
    
    Returns:
        Le résultat développé
    """
    return a**2 + 2*a*b + b**2


def identite_difference_carres(a, b):
    """
    (a - b)² = a² - 2ab + b²
    
    Args:
        a, b: Les valeurs
    
    Returns:
        Le résultat développé
    """
    return a**2 - 2*a*b + b**2


def identite_produit_somme_difference(a, b):
    """
    (a + b)(a - b) = a² - b²
    
    Args:
        a, b: Les valeurs
    
    Returns:
        Le résultat
    """
    return a**2 - b**2


def identite_carre_trinome(a, b, c):
    """
    (a + b + c)² = a² + b² + c² + 2ab + 2ac + 2bc
    
    Args:
        a, b, c: Les valeurs
    
    Returns:
        Le résultat développé
    """
    return a**2 + b**2 + c**2 + 2*a*b + 2*a*c + 2*b*c


def identite_lagrange(a1, b1, a2, b2):
    """
    Identité de Lagrange: (a₁² + b₁²)(a₂² + b₂²) = (a₁a₂ + b₁b₂)² + (a₁b₂ - a₂b₁)²
    
    Args:
        a1, b1, a2, b2: Les valeurs
    
    Returns:
        Tuple (côté_gauche, côté_droit) pour vérification
    """
    gauche = (a1**2 + b1**2) * (a2**2 + b2**2)
    droite = (a1*a2 + b1*b2)**2 + (a1*b2 - a2*b1)**2
    return (gauche, droite)


def identite_sophie_germain(a, b):
    """
    Identité de Sophie Germain: a⁴ + 4b⁴ = (a² + 2b² + 2ab)(a² + 2b² - 2ab)
    
    Args:
        a, b: Les valeurs
    
    Returns:
        Tuple (résultat_gauche, résultat_droit) pour vérification
    """
    gauche = a**4 + 4*b**4
    facteur1 = a**2 + 2*b**2 + 2*a*b
    facteur2 = a**2 + 2*b**2 - 2*a*b
    droite = facteur1 * facteur2
    return (gauche, droite)


def identite_brahmagupta_fibonacci(a, b, c, d):
    """
    Identité de Brahmagupta-Fibonacci:
    (a² + b²)(c² + d²) = (ac - bd)² + (ad + bc)²
    
    Args:
        a, b, c, d: Les valeurs
    
    Returns:
        Tuple (résultat_gauche, résultat_droit) pour vérification
    """
    gauche = (a**2 + b**2) * (c**2 + d**2)
    droite = (a*c - b*d)**2 + (a*d + b*c)**2
    return (gauche, droite)


def formule_heron_identite(a, b, c):
    """
    Utilise l'identité pour calculer 16 * Aire² d'un triangle
    16A² = 2a²b² + 2b²c² + 2c²a² - a⁴ - b⁴ - c⁴
    
    Args:
        a, b, c: Les côtés du triangle
    
    Returns:
        16 * Aire²
    """
    return (2*a**2*b**2 + 2*b**2*c**2 + 2*c**2*a**2 
            - a**4 - b**4 - c**4)


def identite_parallelogramme(a, b, c, d):
    """
    Loi du parallélogramme: |a+b|² + |a-b|² = 2(|a|² + |b|²)
    Pour vecteurs avec coordonnées
    
    Args:
        a, b: Coordonnées du premier vecteur
        c, d: Coordonnées du deuxième vecteur
    
    Returns:
        Tuple (gauche, droite) pour vérification
    """
    gauche = (a+c)**2 + (b+d)**2 + (a-c)**2 + (b-d)**2
    droite = 2 * (a**2 + b**2 + c**2 + d**2)
    return (gauche, droite)


# ============================================================================
# VOLUMES DE SOLIDES
# ============================================================================

def volume_cube(arete):
    """
    Calcule le volume d'un cube
    
    Args:
        arete: Longueur de l'arête
    
    Returns:
        Le volume (a³)
    """
    return arete ** 3


def volume_parallelepipede(longueur, largeur, hauteur):
    """
    Calcule le volume d'un parallélépipède rectangle
    
    Args:
        longueur: Longueur
        largeur: Largeur
        hauteur: Hauteur
    
    Returns:
        Le volume
    """
    return longueur * largeur * hauteur


def volume_sphere(rayon):
    """
    Calcule le volume d'une sphère
    
    Args:
        rayon: Rayon de la sphère
    
    Returns:
        Le volume (4/3 * π * r³)
    """
    return (4/3) * math.pi * rayon ** 3


def volume_cylindre(rayon, hauteur):
    """
    Calcule le volume d'un cylindre
    
    Args:
        rayon: Rayon de la base
        hauteur: Hauteur du cylindre
    
    Returns:
        Le volume (π * r² * h)
    """
    return math.pi * rayon ** 2 * hauteur


def volume_cone(rayon, hauteur):
    """
    Calcule le volume d'un cône
    
    Args:
        rayon: Rayon de la base
        hauteur: Hauteur du cône
    
    Returns:
        Le volume (1/3 * π * r² * h)
    """
    return (1/3) * math.pi * rayon ** 2 * hauteur


def volume_pyramide(aire_base, hauteur):
    """
    Calcule le volume d'une pyramide
    
    Args:
        aire_base: Aire de la base
        hauteur: Hauteur de la pyramide
    
    Returns:
        Le volume (1/3 * aire_base * h)
    """
    return (1/3) * aire_base * hauteur


def volume_prisme(aire_base, hauteur):
    """
    Calcule le volume d'un prisme
    
    Args:
        aire_base: Aire de la base
        hauteur: Hauteur du prisme
    
    Returns:
        Le volume (aire_base * h)
    """
    return aire_base * hauteur


def volume_tetraedre(arete):
    """
    Calcule le volume d'un tétraèdre régulier
    
    Args:
        arete: Longueur de l'arête
    
    Returns:
        Le volume (a³/(6√2))
    """
    return (arete ** 3) / (6 * math.sqrt(2))


def volume_tore(rayon_majeur, rayon_mineur):
    """
    Calcule le volume d'un tore
    
    Args:
        rayon_majeur: Rayon du cercle central (R)
        rayon_mineur: Rayon du tube (r)
    
    Returns:
        Le volume (2π²Rr²)
    """
    return 2 * math.pi ** 2 * rayon_majeur * rayon_mineur ** 2


def volume_ellipsoide(a, b, c):
    """
    Calcule le volume d'un ellipsoïde
    
    Args:
        a, b, c: Demi-axes de l'ellipsoïde
    
    Returns:
        Le volume (4/3 * π * abc)
    """
    return (4/3) * math.pi * a * b * c


def volume_calotte_spherique(rayon, hauteur):
    """
    Calcule le volume d'une calotte sphérique
    
    Args:
        rayon: Rayon de la sphère
        hauteur: Hauteur de la calotte
    
    Returns:
        Le volume
    """
    return (math.pi * hauteur ** 2 * (3 * rayon - hauteur)) / 3


def surface_sphere(rayon):
    """
    Calcule la surface d'une sphère
    
    Args:
        rayon: Rayon de la sphère
    
    Returns:
        La surface (4πr²)
    """
    return 4 * math.pi * rayon ** 2


def surface_cylindre(rayon, hauteur):
    """
    Calcule la surface totale d'un cylindre
    
    Args:
        rayon: Rayon de la base
        hauteur: Hauteur
    
    Returns:
        La surface totale (2πr² + 2πrh)
    """
    return 2 * math.pi * rayon ** 2 + 2 * math.pi * rayon * hauteur


def surface_cone(rayon, hauteur):
    """
    Calcule la surface totale d'un cône
    
    Args:
        rayon: Rayon de la base
        hauteur: Hauteur du cône
    
    Returns:
        La surface totale (πr² + πr√(r²+h²))
    """
    generatrice = math.sqrt(rayon ** 2 + hauteur ** 2)
    return math.pi * rayon ** 2 + math.pi * rayon * generatrice


def surface_cube(arete):
    """
    Calcule la surface totale d'un cube
    
    Args:
        arete: Longueur de l'arête
    
    Returns:
        La surface (6a²)
    """
    return 6 * arete ** 2


def surface_parallelepipede(longueur, largeur, hauteur):
    """
    Calcule la surface totale d'un parallélépipède
    
    Args:
        longueur, largeur, hauteur: Dimensions
    
    Returns:
        La surface totale
    """
    return 2 * (longueur * largeur + longueur * hauteur + largeur * hauteur)


# ============================================================================
# RÉSOLUTION D'ÉQUATIONS
# ============================================================================

def resoudre_equation_degre1(a, b):
    """
    Résout l'équation ax + b = 0
    
    Args:
        a: Coefficient de x
        b: Terme constant
    
    Returns:
        La solution x ou un message si pas de solution/infinité de solutions
    """
    if a == 0:
        if b == 0:
            return "Infinité de solutions (∀x ∈ ℝ)"
        else:
            return "Pas de solution (∅)"
    return -b / a


def resoudre_equation_degre2(a, b, c):
    """
    Résout l'équation ax² + bx + c = 0
    
    Args:
        a, b, c: Coefficients du trinôme
    
    Returns:
        Dictionnaire contenant les solutions et informations
    """
    if a == 0:
        return {"type": "degré 1", "solutions": [resoudre_equation_degre1(b, c)]}
    
    delta = trinome_discriminant(a, b, c)
    
    if delta < 0:
        partie_reelle = -b / (2 * a)
        partie_imag = math.sqrt(-delta) / (2 * a)
        return {
            "type": "complexes",
            "delta": delta,
            "solutions": [
                f"{partie_reelle} + {partie_imag}i",
                f"{partie_reelle} - {partie_imag}i"
            ]
        }
    elif delta == 0:
        x0 = -b / (2 * a)
        return {
            "type": "racine double",
            "delta": delta,
            "solutions": [x0]
        }
    else:
        racine_delta = math.sqrt(delta)
        x1 = (-b + racine_delta) / (2 * a)
        x2 = (-b - racine_delta) / (2 * a)
        return {
            "type": "deux racines réelles",
            "delta": delta,
            "solutions": sorted([x1, x2])
        }


def resoudre_equation_degre3(a, b, c, d):
    """
    Résout l'équation ax³ + bx² + cx + d = 0 (méthode de Cardan simplifiée)
    
    Args:
        a, b, c, d: Coefficients
    
    Returns:
        Liste des solutions réelles
    """
    if a == 0:
        return resoudre_equation_degre2(b, c, d)
    
    # Normalisation
    b, c, d = b/a, c/a, d/a
    
    # Changement de variable pour éliminer le terme en x²
    p = c - b**2/3
    q = 2*b**3/27 - b*c/3 + d
    
    discriminant = -(4*p**3 + 27*q**2)
    
    solutions = []
    
    if abs(discriminant) < 1e-10:  # Une ou deux racines réelles
        if abs(p) < 1e-10:
            solutions.append(-b/3)
        else:
            u = (-q/2)**(1/3) if q != 0 else 0
            solutions.append(2*u - b/3)
            solutions.append(-u - b/3)
    elif discriminant > 0:  # Trois racines réelles distinctes
        m = 2 * math.sqrt(-p/3)
        theta = math.acos(3*q/(p*m)) / 3
        for k in range(3):
            x = m * math.cos(theta - 2*math.pi*k/3) - b/3
            solutions.append(x)
    else:  # Une racine réelle
        sqrt_val = math.sqrt(q**2/4 + p**3/27)
        u = (-q/2 + sqrt_val)**(1/3) if -q/2 + sqrt_val >= 0 else -(abs(-q/2 + sqrt_val)**(1/3))
        v = (-q/2 - sqrt_val)**(1/3) if -q/2 - sqrt_val >= 0 else -(abs(-q/2 - sqrt_val)**(1/3))
        solutions.append(u + v - b/3)
    
    return {"solutions": sorted(set(round(s, 10) for s in solutions))}


def resoudre_equation_bicarree(a, b, c):
    """
    Résout l'équation ax⁴ + bx² + c = 0 (équation bicarrée)
    
    Args:
        a, b, c: Coefficients
    
    Returns:
        Liste des solutions réelles
    """
    # On pose X = x², donc aX² + bX + c = 0
    resultats_X = resoudre_equation_degre2(a, b, c)
    
    solutions = []
    if resultats_X["type"] == "deux racines réelles":
        for X in resultats_X["solutions"]:
            if X > 0:
                solutions.append(math.sqrt(X))
                solutions.append(-math.sqrt(X))
            elif X == 0:
                solutions.append(0)
    elif resultats_X["type"] == "racine double":
        X = resultats_X["solutions"][0]
        if X > 0:
            solutions.append(math.sqrt(X))
            solutions.append(-math.sqrt(X))
        elif X == 0:
            solutions.append(0)
    
    return {"solutions": sorted(set(solutions))}


def resoudre_equation_produit_nul(facteurs):
    """
    Résout une équation de la forme (x-a)(x-b)...(x-n) = 0
    
    Args:
        facteurs: Liste des valeurs [a, b, ..., n]
    
    Returns:
        Liste des solutions
    """
    return {"solutions": sorted(set(facteurs))}


# ============================================================================
# RÉSOLUTION D'INÉQUATIONS
# ============================================================================

def resoudre_inequation_degre1(a, b, signe):
    """
    Résout les inéquations ax + b > 0, ax + b ≥ 0, ax + b < 0, ax + b ≤ 0
    
    Args:
        a: Coefficient de x
        b: Terme constant
        signe: '>', '>=', '<', '<='
    
    Returns:
        Dictionnaire avec l'intervalle solution
    """
    if a == 0:
        if b == 0:
            if signe in ['>=', '<=']:
                return {"solution": "ℝ tout entier"}
            else:
                return {"solution": "ensemble vide ∅"}
        else:
            verif = (b > 0 and signe in ['>', '>=']) or (b < 0 and signe in ['<', '<='])
            return {"solution": "ℝ tout entier" if verif else "ensemble vide ∅"}
    
    x0 = -b / a
    
    if a > 0:
        if signe == '>':
            return {"solution": f"]{x0}, +∞[", "borne": x0, "type": "ouvert à gauche"}
        elif signe == '>=':
            return {"solution": f"[{x0}, +∞[", "borne": x0, "type": "fermé à gauche"}
        elif signe == '<':
            return {"solution": f"]-∞, {x0}[", "borne": x0, "type": "ouvert à droite"}
        else:  # <=
            return {"solution": f"]-∞, {x0}]", "borne": x0, "type": "fermé à droite"}
    else:  # a < 0
        if signe == '>':
            return {"solution": f"]-∞, {x0}[", "borne": x0, "type": "ouvert à droite"}
        elif signe == '>=':
            return {"solution": f"]-∞, {x0}]", "borne": x0, "type": "fermé à droite"}
        elif signe == '<':
            return {"solution": f"]{x0}, +∞[", "borne": x0, "type": "ouvert à gauche"}
        else:  # <=
            return {"solution": f"[{x0}, +∞[", "borne": x0, "type": "fermé à gauche"}


def resoudre_inequation_degre2(a, b, c, signe):
    """
    Résout les inéquations ax² + bx + c > 0, ≥ 0, < 0, ≤ 0
    
    Args:
        a, b, c: Coefficients du trinôme
        signe: '>', '>=', '<', '<='
    
    Returns:
        Dictionnaire avec l'intervalle solution et le tableau de signes
    """
    if a == 0:
        return resoudre_inequation_degre1(b, c, signe)
    
    delta = trinome_discriminant(a, b, c)
    racines = racines_trinome(a, b, c)
    
    # Déterminer le signe du trinôme
    signe_a = "positif" if a > 0 else "négatif"
    
    if delta < 0:
        # Pas de racine réelle, signe constant
        toujours_positif = (a > 0)
        if signe in ['>', '>=']:
            solution = "ℝ tout entier" if toujours_positif else "ensemble vide ∅"
        else:
            solution = "ensemble vide ∅" if toujours_positif else "ℝ tout entier"
        
        return {
            "delta": delta,
            "racines": "aucune racine réelle",
            "signe_constant": "toujours positif" if toujours_positif else "toujours négatif",
            "solution": solution
        }
    
    elif delta == 0:
        # Une racine double
        x0 = racines[0]
        
        if signe == '>':
            solution = f"ℝ \\ {{{x0}}}"
        elif signe == '>=':
            solution = "ℝ tout entier"
        elif signe == '<':
            solution = "ensemble vide ∅"
        else:  # <=
            solution = f"{{{x0}}}"
        
        return {
            "delta": delta,
            "racines": [x0],
            "type": "racine double",
            "zones_critiques": [x0],
            "solution": solution
        }
    
    else:
        # Deux racines distinctes
        x1, x2 = sorted(racines)
        
        if a > 0:
            # Parabole tournée vers le haut: négatif entre les racines
            if signe == '>':
                solution = f"]-∞, {x1}[ ∪ ]{x2}, +∞["
            elif signe == '>=':
                solution = f"]-∞, {x1}] ∪ [{x2}, +∞["
            elif signe == '<':
                solution = f"]{x1}, {x2}["
            else:  # <=
                solution = f"[{x1}, {x2}]"
        else:
            # Parabole tournée vers le bas: positif entre les racines
            if signe == '>':
                solution = f"]{x1}, {x2}["
            elif signe == '>=':
                solution = f"[{x1}, {x2}]"
            elif signe == '<':
                solution = f"]-∞, {x1}[ ∪ ]{x2}, +∞["
            else:  # <=
                solution = f"]-∞, {x1}] ∪ [{x2}, +∞["
        
        return {
            "delta": delta,
            "racines": [x1, x2],
            "type": "deux racines distinctes",
            "zones_critiques": [x1, x2],
            "signe_parabole": "vers le haut" if a > 0 else "vers le bas",
            "tableau_signes": {
                f"]-∞, {x1}[": "+" if a > 0 else "-",
                f"{x1}": "0",
                f"]{x1}, {x2}[": "-" if a > 0 else "+",
                f"{x2}": "0",
                f"]{x2}, +∞[": "+" if a > 0 else "-"
            },
            "solution": solution
        }


def resoudre_inequation_produit(facteurs, signes_facteurs, signe):
    """
    Résout une inéquation de type (x-a)(x-b)...(x-n) > 0 (ou ≥, <, ≤)
    
    Args:
        facteurs: Liste des valeurs critiques [a, b, ..., n]
        signes_facteurs: Liste des signes de chaque facteur (1 ou -1)
        signe: '>', '>=', '<', '<='
    
    Returns:
        Dictionnaire avec l'intervalle solution et le tableau de signes
    """
    if not facteurs:
        return {"solution": "ℝ tout entier" if signe in ['>', '>='] else "ensemble vide ∅"}
    
    zones_critiques = sorted(set(facteurs))
    n = len(zones_critiques)
    
    # Construire les intervalles
    intervalles = [f"]-∞, {zones_critiques[0]}["]
    for i in range(n):
        intervalles.append(f"{zones_critiques[i]}")
        if i < n - 1:
            intervalles.append(f"]{zones_critiques[i]}, {zones_critiques[i+1]}[")
    intervalles.append(f"]{zones_critiques[-1]}, +∞[")
    
    # Déterminer le signe dans chaque intervalle
    signe_produit = 1
    for s in signes_facteurs:
        signe_produit *= s
    
    tableau_signes = {}
    solutions = []
    
    for i, intervalle in enumerate(intervalles):
        if zones_critiques[0] in intervalle and '[' not in intervalle and ']' not in intervalle:
            # C'est une valeur critique
            tableau_signes[intervalle] = "0"
            if signe in ['>=', '<=']:
                solutions.append(intervalle)
        else:
            # Calculer le signe dans cet intervalle
            # (logique simplifiée - dans un cas réel, il faudrait tester)
            signe_actuel = "+" if i % 2 == 0 else "-"
            tableau_signes[intervalle] = signe_actuel
            
            if (signe in ['>', '>='] and signe_actuel == '+') or \
               (signe in ['<', '<='] and signe_actuel == '-'):
                solutions.append(intervalle)
    
    solution_str = " ∪ ".join(solutions) if solutions else "ensemble vide ∅"
    
    return {
        "zones_critiques": zones_critiques,
        "tableau_signes": tableau_signes,
        "solution": solution_str
    }

def signe_global(signe):
    """
    Retourne le signe global attendu pour l'inéquation
    
    Args:
        signe: '>', '>=', '<', '<='
    Returns:
        '+' pour '>' et '>=', '-' pour '<' et '<='
    """
    return '+' if signe in ['>', '>='] else '-'

def resoudre_inequation_quotient(num_a, num_b, den_a, den_b, signe):
    """
    Résout une inéquation de type (ax + b) / (cx + d) > 0 (ou ≥, <, ≤)
    
    Args:
        num_a, num_b: Coefficients du numérateur
        den_a, den_b: Coefficients du dénominateur
        signe: '>', '>=', '<', '<='
    
    Returns:
        Dictionnaire avec l'intervalle solution et les valeurs interdites
    """
    # Racine du numérateur
    racine_num = None if num_a == 0 else -num_b / num_a
    # Valeur interdite (racine du dénominateur)
    valeur_interdite = None if den_a == 0 else -den_b / den_a
    
    if valeur_interdite is None:
        return {"erreur": "Dénominateur constant"}
    
    zones_critiques = []
    if racine_num is not None:
        zones_critiques.append(racine_num)
    zones_critiques.append(valeur_interdite)
    zones_critiques = sorted(set(zones_critiques))
    
    # Déterminer les signes
    signe_num_gauche = "+" if num_a > 0 else "-"
    signe_den_gauche = "+" if den_a > 0 else "-"
    
    # Construction du tableau de signes
    tableau = {
        "numerateur": {},
        "denominateur": {},
        "quotient": {}
    }
    
    solutions = []
    
    # Intervalle avant première zone critique
    if zones_critiques:
        interval = f"]-∞, {zones_critiques[0]}["
        signe_n = "+" if (racine_num is None or zones_critiques[0] < racine_num) and num_a > 0 else "-"
        signe_d = "+" if zones_critiques[0] < valeur_interdite and den_a > 0 else "-"
        signe_q = "+" if signe_n == signe_d else "-"
        
        tableau["quotient"][interval] = signe_q
        
        if (signe_global in ['>', '>='] and signe_q == '+') or \
           (signe_global in ['<', '<='] and signe_q == '-'):
            solutions.append(interval)
    
    # Entre les zones critiques et après
    for i in range(len(zones_critiques)):
        val = zones_critiques[i]
        
        # La valeur elle-même
        if val == valeur_interdite:
            tableau["quotient"][f"{val}"] = "interdit"
        elif val == racine_num:
            tableau["quotient"][f"{val}"] = "0"
            if signe in ['>=', '<=']:
                solutions.append(f"{{{val}}}")
        
        # Après cette valeur
        if i < len(zones_critiques) - 1:
            interval = f"]{val}, {zones_critiques[i+1]}["
        else:
            interval = f"]{val}, +∞["
        
        # Test de signe (simplifié)
        test_point = val + 0.1 if i == len(zones_critiques) - 1 else (val + zones_critiques[i+1]) / 2
        signe_n_test = "+" if (num_a * test_point + num_b) > 0 else "-"
        signe_d_test = "+" if (den_a * test_point + den_b) > 0 else "-"
        signe_q = "+" if signe_n_test == signe_d_test else "-"
        
        tableau["quotient"][interval] = signe_q
        
        if (signe in ['>', '>='] and signe_q == '+') or \
           (signe in ['<', '<='] and signe_q == '-'):
            solutions.append(interval)
    
    solution_str = " ∪ ".join(solutions) if solutions else "ensemble vide ∅"
    
    return {
        "valeur_interdite": valeur_interdite,
        "racine_numerateur": racine_num,
        "zones_critiques": zones_critiques,
        "tableau_signes": tableau,
        "solution": solution_str
    }
