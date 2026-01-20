"""
Module de calcul avec les nombres complexes
Contient toutes les opérations, conversions et formules pour les nombres complexes
"""

import math
import cmath


class ComplexePersonnalise:
    """
    Classe pour représenter et manipuler des nombres complexes
    avec différentes formes d'écriture
    """
    
    def __init__(self, reel=0, imaginaire=0, forme='cartesienne'):
        """
        Initialise un nombre complexe
        
        Args:
            reel: Partie réelle (ou module en forme polaire)
            imaginaire: Partie imaginaire (ou argument en forme polaire)
            forme: 'cartesienne' ou 'polaire'
        """
        if forme == 'cartesienne':
            self.reel = reel
            self.imaginaire = imaginaire
        elif forme == 'polaire':
            # Conversion polaire -> cartésienne
            self.reel = reel * math.cos(imaginaire)
            self.imaginaire = reel * math.sin(imaginaire)
        else:
            raise ValueError("Forme doit être 'cartesienne' ou 'polaire'")
    
    def __str__(self):
        """Représentation en chaîne (forme algébrique)"""
        return forme_algebrique(self.reel, self.imaginaire)
    
    def __repr__(self):
        return f"ComplexePersonnalise({self.reel}, {self.imaginaire})"
    
    def module(self):
        """Calcule le module du nombre complexe"""
        return module_complexe(self.reel, self.imaginaire)
    
    def argument(self):
        """Calcule l'argument du nombre complexe"""
        return argument_complexe(self.reel, self.imaginaire)
    
    def conjugue(self):
        """Retourne le conjugué du nombre complexe"""
        return ComplexePersonnalise(self.reel, -self.imaginaire)
    
    def forme_polaire(self):
        """Retourne (module, argument)"""
        return (self.module(), self.argument())
    
    def forme_exponentielle(self):
        """Retourne la forme exponentielle en chaîne"""
        return forme_exponentielle(self.reel, self.imaginaire)


# ============================================================================
# CONVERSIONS ET REPRÉSENTATIONS
# ============================================================================

def forme_algebrique(a, b):
    """
    Représente un nombre complexe sous forme algébrique: a + bi
    
    Args:
        a: Partie réelle
        b: Partie imaginaire
    
    Returns:
        Chaîne de caractères représentant le nombre complexe
    """
    if b == 0:
        return str(a)
    if a == 0:
        if b == 1:
            return "i"
        elif b == -1:
            return "-i"
        return f"{b}i"
    
    if b == 1:
        return f"{a} + i"
    elif b == -1:
        return f"{a} - i"
    elif b > 0:
        return f"{a} + {b}i"
    else:
        return f"{a} - {abs(b)}i"


def forme_polaire_str(module, argument):
    """
    Représente un nombre complexe sous forme polaire: r(cos θ + i sin θ)
    
    Args:
        module: Le module r
        argument: L'argument θ en radians
    
    Returns:
        Chaîne de caractères représentant la forme polaire
    """
    return f"{module}(cos({argument}) + i·sin({argument}))"


def forme_exponentielle(a, b):
    """
    Représente un nombre complexe sous forme exponentielle: r·e^(iθ)
    
    Args:
        a: Partie réelle
        b: Partie imaginaire
    
    Returns:
        Chaîne de caractères représentant la forme exponentielle
    """
    r = module_complexe(a, b)
    theta = argument_complexe(a, b)
    return f"{r}·e^(i·{theta})"


def cartesien_vers_polaire(a, b):
    """
    Convertit de la forme cartésienne (a, b) vers la forme polaire (r, θ)
    
    Args:
        a: Partie réelle
        b: Partie imaginaire
    
    Returns:
        Tuple (module, argument)
    """
    r = module_complexe(a, b)
    theta = argument_complexe(a, b)
    return (r, theta)


def polaire_vers_cartesien(r, theta):
    """
    Convertit de la forme polaire (r, θ) vers la forme cartésienne (a, b)
    
    Args:
        r: Module
        theta: Argument en radians
    
    Returns:
        Tuple (partie_reelle, partie_imaginaire)
    """
    a = r * math.cos(theta)
    b = r * math.sin(theta)
    return (a, b)


# ============================================================================
# OPÉRATIONS DE BASE
# ============================================================================

def addition_complexe(a1, b1, a2, b2):
    """
    Addition de deux nombres complexes: (a1 + b1i) + (a2 + b2i)
    
    Args:
        a1, b1: Parties réelle et imaginaire du premier complexe
        a2, b2: Parties réelle et imaginaire du second complexe
    
    Returns:
        Tuple (partie_reelle, partie_imaginaire) du résultat
    """
    return (a1 + a2, b1 + b2)


def soustraction_complexe(a1, b1, a2, b2):
    """
    Soustraction de deux nombres complexes: (a1 + b1i) - (a2 + b2i)
    
    Args:
        a1, b1: Parties réelle et imaginaire du premier complexe
        a2, b2: Parties réelle et imaginaire du second complexe
    
    Returns:
        Tuple (partie_reelle, partie_imaginaire) du résultat
    """
    return (a1 - a2, b1 - b2)


def multiplication_complexe(a1, b1, a2, b2):
    """
    Multiplication de deux nombres complexes: (a1 + b1i) * (a2 + b2i)
    Formule: (a1*a2 - b1*b2) + (a1*b2 + a2*b1)i
    
    Args:
        a1, b1: Parties réelle et imaginaire du premier complexe
        a2, b2: Parties réelle et imaginaire du second complexe
    
    Returns:
        Tuple (partie_reelle, partie_imaginaire) du résultat
    """
    partie_reelle = a1 * a2 - b1 * b2
    partie_imaginaire = a1 * b2 + a2 * b1
    return (partie_reelle, partie_imaginaire)


def division_complexe(a1, b1, a2, b2):
    """
    Division de deux nombres complexes: (a1 + b1i) / (a2 + b2i)
    
    Args:
        a1, b1: Parties réelle et imaginaire du numérateur
        a2, b2: Parties réelle et imaginaire du dénominateur
    
    Returns:
        Tuple (partie_reelle, partie_imaginaire) du résultat
    
    Raises:
        ValueError: Si le dénominateur est zéro
    """
    denominateur = a2 ** 2 + b2 ** 2
    if denominateur == 0:
        raise ValueError("Division par zéro impossible")
    
    partie_reelle = (a1 * a2 + b1 * b2) / denominateur
    partie_imaginaire = (b1 * a2 - a1 * b2) / denominateur
    return (partie_reelle, partie_imaginaire)


def multiplication_scalaire(a, b, scalaire):
    """
    Multiplie un nombre complexe par un scalaire réel
    
    Args:
        a, b: Parties réelle et imaginaire du complexe
        scalaire: Le nombre réel multiplicateur
    
    Returns:
        Tuple (partie_reelle, partie_imaginaire) du résultat
    """
    return (a * scalaire, b * scalaire)


# ============================================================================
# PROPRIÉTÉS ET CARACTÉRISTIQUES
# ============================================================================

def module_complexe(a, b):
    """
    Calcule le module (norme) d'un nombre complexe: |z| = √(a² + b²)
    
    Args:
        a: Partie réelle
        b: Partie imaginaire
    
    Returns:
        Le module du nombre complexe
    """
    return math.sqrt(a ** 2 + b ** 2)


def argument_complexe(a, b):
    """
    Calcule l'argument (angle) d'un nombre complexe en radians
    
    Args:
        a: Partie réelle
        b: Partie imaginaire
    
    Returns:
        L'argument en radians (entre -π et π)
    """
    return math.atan2(b, a)


def argument_principal(a, b):
    """
    Calcule l'argument principal d'un nombre complexe (entre -π et π)
    
    Args:
        a: Partie réelle
        b: Partie imaginaire
    
    Returns:
        L'argument principal en radians
    """
    return argument_complexe(a, b)


def conjugue(a, b):
    """
    Calcule le conjugué d'un nombre complexe: z̄ = a - bi
    
    Args:
        a: Partie réelle
        b: Partie imaginaire
    
    Returns:
        Tuple (partie_reelle, partie_imaginaire) du conjugué
    """
    return (a, -b)


def partie_reelle(z):
    """
    Extrait la partie réelle d'un nombre complexe Python
    
    Args:
        z: Nombre complexe
    
    Returns:
        La partie réelle
    """
    return z.real


def partie_imaginaire(z):
    """
    Extrait la partie imaginaire d'un nombre complexe Python
    
    Args:
        z: Nombre complexe
    
    Returns:
        La partie imaginaire
    """
    return z.imag


# ============================================================================
# PUISSANCES ET RACINES
# ============================================================================

def puissance_complexe(a, b, n):
    """
    Calcule la puissance n-ième d'un nombre complexe
    Utilise la forme polaire: z^n = r^n * e^(i*n*θ)
    
    Args:
        a, b: Parties réelle et imaginaire
        n: L'exposant
    
    Returns:
        Tuple (partie_reelle, partie_imaginaire) du résultat
    """
    r = module_complexe(a, b)
    theta = argument_complexe(a, b)
    
    r_n = r ** n
    theta_n = n * theta
    
    return polaire_vers_cartesien(r_n, theta_n)


def racine_nieme_complexe(a, b, n, k=0):
    """
    Calcule la k-ième racine n-ième d'un nombre complexe
    Formule: z^(1/n) = r^(1/n) * e^(i*(θ + 2πk)/n) pour k = 0, 1, ..., n-1
    
    Args:
        a, b: Parties réelle et imaginaire
        n: L'indice de la racine
        k: L'indice de la solution (de 0 à n-1)
    
    Returns:
        Tuple (partie_reelle, partie_imaginaire) de la k-ième racine
    """
    if n == 0:
        raise ValueError("L'indice de la racine ne peut pas être zéro")
    
    r = module_complexe(a, b)
    theta = argument_complexe(a, b)
    
    r_racine = r ** (1 / n)
    theta_racine = (theta + 2 * math.pi * k) / n
    
    return polaire_vers_cartesien(r_racine, theta_racine)


def toutes_racines_niemes(a, b, n):
    """
    Calcule toutes les racines n-ièmes d'un nombre complexe
    
    Args:
        a, b: Parties réelle et imaginaire
        n: L'indice de la racine
    
    Returns:
        Liste de tuples (partie_reelle, partie_imaginaire) pour toutes les racines
    """
    racines = []
    for k in range(n):
        racines.append(racine_nieme_complexe(a, b, n, k))
    return racines


def carre_complexe(a, b):
    """
    Calcule le carré d'un nombre complexe: z²
    
    Args:
        a, b: Parties réelle et imaginaire
    
    Returns:
        Tuple (partie_reelle, partie_imaginaire) du carré
    """
    return multiplication_complexe(a, b, a, b)


def racine_carree_complexe(a, b):
    """
    Calcule la racine carrée principale d'un nombre complexe
    
    Args:
        a, b: Parties réelle et imaginaire
    
    Returns:
        Tuple (partie_reelle, partie_imaginaire) de la racine carrée
    """
    return racine_nieme_complexe(a, b, 2, 0)


# ============================================================================
# FONCTIONS TRANSCENDANTES
# ============================================================================

def exponentielle_complexe(a, b):
    """
    Calcule e^(a+bi) = e^a * (cos(b) + i*sin(b))
    
    Args:
        a, b: Parties réelle et imaginaire
    
    Returns:
        Tuple (partie_reelle, partie_imaginaire)
    """
    exp_a = math.exp(a)
    return (exp_a * math.cos(b), exp_a * math.sin(b))


def logarithme_complexe(a, b):
    """
    Calcule le logarithme naturel d'un nombre complexe
    ln(z) = ln(|z|) + i*arg(z)
    
    Args:
        a, b: Parties réelle et imaginaire
    
    Returns:
        Tuple (partie_reelle, partie_imaginaire)
    """
    r = module_complexe(a, b)
    if r == 0:
        raise ValueError("Le logarithme de zéro n'est pas défini")
    theta = argument_complexe(a, b)
    return (math.log(r), theta)


def cosinus_complexe(a, b):
    """
    Calcule cos(a+bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
    
    Args:
        a, b: Parties réelle et imaginaire
    
    Returns:
        Tuple (partie_reelle, partie_imaginaire)
    """
    z = complex(a, b)
    resultat = cmath.cos(z)
    return (resultat.real, resultat.imag)


def sinus_complexe(a, b):
    """
    Calcule sin(a+bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
    
    Args:
        a, b: Parties réelle et imaginaire
    
    Returns:
        Tuple (partie_reelle, partie_imaginaire)
    """
    z = complex(a, b)
    resultat = cmath.sin(z)
    return (resultat.real, resultat.imag)


def tangente_complexe(a, b):
    """
    Calcule tan(a+bi)
    
    Args:
        a, b: Parties réelle et imaginaire
    
    Returns:
        Tuple (partie_reelle, partie_imaginaire)
    """
    z = complex(a, b)
    resultat = cmath.tan(z)
    return (resultat.real, resultat.imag)


def cosinus_hyperbolique_complexe(a, b):
    """
    Calcule cosh(a+bi)
    
    Args:
        a, b: Parties réelle et imaginaire
    
    Returns:
        Tuple (partie_reelle, partie_imaginaire)
    """
    z = complex(a, b)
    resultat = cmath.cosh(z)
    return (resultat.real, resultat.imag)


def sinus_hyperbolique_complexe(a, b):
    """
    Calcule sinh(a+bi)
    
    Args:
        a, b: Parties réelle et imaginaire
    
    Returns:
        Tuple (partie_reelle, partie_imaginaire)
    """
    z = complex(a, b)
    resultat = cmath.sinh(z)
    return (resultat.real, resultat.imag)


def tangente_hyperbolique_complexe(a, b):
    """
    Calcule tanh(a+bi)
    
    Args:
        a, b: Parties réelle et imaginaire
    
    Returns:
        Tuple (partie_reelle, partie_imaginaire)
    """
    z = complex(a, b)
    resultat = cmath.tanh(z)
    return (resultat.real, resultat.imag)


# ============================================================================
# FORMULES SPÉCIALES
# ============================================================================

def formule_euler(theta):
    """
    Formule d'Euler: e^(iθ) = cos(θ) + i*sin(θ)
    
    Args:
        theta: L'angle en radians
    
    Returns:
        Tuple (cos(theta), sin(theta))
    """
    return (math.cos(theta), math.sin(theta))


def formule_moivre(r, theta, n):
    """
    Formule de Moivre: [r(cos θ + i sin θ)]^n = r^n(cos(nθ) + i sin(nθ))
    
    Args:
        r: Module
        theta: Argument en radians
        n: Puissance
    
    Returns:
        Tuple (partie_reelle, partie_imaginaire)
    """
    r_n = r ** n
    theta_n = n * theta
    return polaire_vers_cartesien(r_n, theta_n)


def inverse_complexe(a, b):
    """
    Calcule l'inverse d'un nombre complexe: 1/z
    
    Args:
        a, b: Parties réelle et imaginaire
    
    Returns:
        Tuple (partie_reelle, partie_imaginaire) de 1/z
    
    Raises:
        ValueError: Si z = 0
    """
    module_carre = a ** 2 + b ** 2
    if module_carre == 0:
        raise ValueError("Impossible de calculer l'inverse de zéro")
    return (a / module_carre, -b / module_carre)


def distance_complexe(a1, b1, a2, b2):
    """
    Calcule la distance entre deux nombres complexes: |z1 - z2|
    
    Args:
        a1, b1: Parties réelle et imaginaire de z1
        a2, b2: Parties réelle et imaginaire de z2
    
    Returns:
        La distance entre z1 et z2
    """
    diff_a = a1 - a2
    diff_b = b1 - b2
    return module_complexe(diff_a, diff_b)


def produit_conjugues(a, b):
    """
    Calcule z * z̄ = |z|²
    
    Args:
        a, b: Parties réelle et imaginaire
    
    Returns:
        Le produit z * z̄ (toujours réel et positif)
    """
    return a ** 2 + b ** 2


def rotation_complexe(a, b, angle):
    """
    Effectue une rotation d'un nombre complexe d'un angle donné
    z' = z * e^(iθ)
    
    Args:
        a, b: Parties réelle et imaginaire
        angle: Angle de rotation en radians
    
    Returns:
        Tuple (partie_reelle, partie_imaginaire) après rotation
    """
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return multiplication_complexe(a, b, cos_angle, sin_angle)


# ============================================================================
# VÉRIFICATIONS ET TESTS
# ============================================================================

def est_reel(a, b, tolerance=1e-10):
    """
    Vérifie si un nombre complexe est réel (partie imaginaire nulle)
    
    Args:
        a, b: Parties réelle et imaginaire
        tolerance: Tolérance pour la comparaison
    
    Returns:
        True si le nombre est réel, False sinon
    """
    return abs(b) < tolerance


def est_imaginaire_pur(a, b, tolerance=1e-10):
    """
    Vérifie si un nombre complexe est imaginaire pur (partie réelle nulle)
    
    Args:
        a, b: Parties réelle et imaginaire
        tolerance: Tolérance pour la comparaison
    
    Returns:
        True si le nombre est imaginaire pur, False sinon
    """
    return abs(a) < tolerance


def est_nul(a, b, tolerance=1e-10):
    """
    Vérifie si un nombre complexe est nul
    
    Args:
        a, b: Parties réelle et imaginaire
        tolerance: Tolérance pour la comparaison
    
    Returns:
        True si le nombre est nul, False sinon
    """
    return abs(a) < tolerance and abs(b) < tolerance


def egaux_complexes(a1, b1, a2, b2, tolerance=1e-10):
    """
    Vérifie si deux nombres complexes sont égaux
    
    Args:
        a1, b1: Premier nombre complexe
        a2, b2: Deuxième nombre complexe
        tolerance: Tolérance pour la comparaison
    
    Returns:
        True si les nombres sont égaux, False sinon
    """
    return abs(a1 - a2) < tolerance and abs(b1 - b2) < tolerance


# ============================================================================
# CONSTANTES COMPLEXES
# ============================================================================

def unite_imaginaire():
    """
    Retourne l'unité imaginaire i = (0, 1)
    
    Returns:
        Tuple (0, 1)
    """
    return (0, 1)


def racines_unite(n):
    """
    Calcule les n racines n-ièmes de l'unité
    
    Args:
        n: Le degré
    
    Returns:
        Liste des n racines de l'unité
    """
    return toutes_racines_niemes(1, 0, n)


def nombre_i():
    """
    Retourne le nombre i sous forme Python complex
    
    Returns:
        1j (nombre complexe Python)
    """
    return 1j


# ============================================================================
# FORMULES ET THÉORÈMES AVANCÉS
# ============================================================================

def appliquer_formule_moivre(r, theta, n):
    """
    Applique la formule de De Moivre: [r(cos θ + i sin θ)]^n = r^n(cos(nθ) + i sin(nθ))
    Cette fonction est un alias plus explicite de formule_moivre
    
    Args:
        r: Module
        theta: Argument en radians
        n: Puissance
    
    Returns:
        Tuple (partie_reelle, partie_imaginaire) du résultat
    """
    r_n = r ** n
    theta_n = n * theta
    return polaire_vers_cartesien(r_n, theta_n)


def formule_moivre_verification(r, theta, n):
    """
    Vérifie la formule de De Moivre en comparant les deux méthodes de calcul
    
    Args:
        r, theta: Forme polaire
        n: Puissance
    
    Returns:
        Dictionnaire avec les deux résultats et leur comparaison
    """
    # Méthode 1 : Formule de De Moivre
    resultat_moivre = appliquer_formule_moivre(r, theta, n)
    
    # Méthode 2 : Conversion puis puissance
    a, b = polaire_vers_cartesien(r, theta)
    resultat_puissance = puissance_complexe(a, b, n)
    
    return {
        "formule_moivre": resultat_moivre,
        "methode_classique": resultat_puissance,
        "sont_egaux": abs(resultat_moivre[0] - resultat_puissance[0]) < 1e-10 and 
                      abs(resultat_moivre[1] - resultat_puissance[1]) < 1e-10
    }


def forme_trigonometrique(a, b):
    """
    Représente un nombre complexe sous forme trigonométrique
    z = r[cos(θ) + i·sin(θ)]
    
    Args:
        a, b: Parties réelle et imaginaire
    
    Returns:
        Chaîne représentant la forme trigonométrique
    """
    r = module_complexe(a, b)
    theta = argument_complexe(a, b)
    return f"{r}[cos({theta}) + i·sin({theta})]"


def forme_phaseur(a, b, unite_angle='radians'):
    """
    Représente un nombre complexe en notation phaseur (utilisée en électricité)
    z = r∠θ
    
    Args:
        a, b: Parties réelle et imaginaire
        unite_angle: 'radians' ou 'degres'
    
    Returns:
        Chaîne représentant le phaseur
    """
    r = module_complexe(a, b)
    theta = argument_complexe(a, b)
    
    if unite_angle == 'degres':
        theta_deg = math.degrees(theta)
        return f"{r}∠{theta_deg}°"
    else:
        return f"{r}∠{theta}"


def forme_euler_string(a, b):
    """
    Représente un nombre complexe avec la formule d'Euler
    z = r·e^(iθ) en utilisant la notation mathématique
    
    Args:
        a, b: Parties réelle et imaginaire
    
    Returns:
        Chaîne représentant la forme d'Euler
    """
    r = module_complexe(a, b)
    theta = argument_complexe(a, b)
    
    if r == 0:
        return "0"
    if r == 1:
        return f"e^(i·{theta})"
    return f"{r}·e^(i·{theta})"


def decomposition_partie_reelle_imaginaire(z_str):
    """
    Décompose une représentation complexe en ses parties
    
    Args:
        z_str: Nombre complexe Python
    
    Returns:
        Dictionnaire avec toutes les représentations
    """
    if isinstance(z_str, complex):
        a, b = z_str.real, z_str.imag
    else:
        return {"erreur": "Format non reconnu"}
    
    return {
        "forme_algebrique": forme_algebrique(a, b),
        "forme_cartesienne": f"({a}, {b})",
        "forme_polaire": forme_polaire_str(*cartesien_vers_polaire(a, b)),
        "forme_exponentielle": forme_exponentielle(a, b),
        "forme_trigonometrique": forme_trigonometrique(a, b),
        "forme_phaseur_rad": forme_phaseur(a, b, 'radians'),
        "forme_phaseur_deg": forme_phaseur(a, b, 'degres'),
        "module": module_complexe(a, b),
        "argument_radians": argument_complexe(a, b),
        "argument_degres": math.degrees(argument_complexe(a, b))
    }


def notation_ingenieur(a, b):
    """
    Notation utilisée en ingénierie (j au lieu de i)
    
    Args:
        a, b: Parties réelle et imaginaire
    
    Returns:
        Chaîne avec notation j
    """
    algebrique = forme_algebrique(a, b)
    return algebrique.replace('i', 'j')


def forme_matricielle_2x2(a, b):
    """
    Représente un nombre complexe comme matrice 2x2
    z = a + bi correspond à [[a, -b], [b, a]]
    
    Args:
        a, b: Parties réelle et imaginaire
    
    Returns:
        Liste représentant la matrice 2x2
    """
    return [[a, -b], [b, a]]


def complexe_depuis_matricielle(matrice):
    """
    Extrait un nombre complexe depuis sa représentation matricielle
    
    Args:
        matrice: Matrice 2x2 [[a, -b], [b, a]]
    
    Returns:
        Tuple (a, b) du nombre complexe
    """
    if len(matrice) != 2 or len(matrice[0]) != 2:
        raise ValueError("La matrice doit être 2x2")
    
    a = matrice[0][0]
    b = matrice[1][0]
    
    # Vérification de la cohérence
    if abs(matrice[0][1] + b) > 1e-10 or abs(matrice[1][1] - a) > 1e-10:
        raise ValueError("La matrice ne représente pas un nombre complexe valide")
    
    return (a, b)


def forme_vecteur_2d(a, b):
    """
    Représente un nombre complexe comme vecteur 2D
    
    Args:
        a, b: Parties réelle et imaginaire
    
    Returns:
        Tuple (a, b) représentant le vecteur
    """
    return (a, b)


def affichage_complet_complexe(a, b):
    """
    Affiche toutes les représentations possibles d'un nombre complexe
    
    Args:
        a, b: Parties réelle et imaginaire
    
    Returns:
        Dictionnaire avec toutes les formes possibles
    """
    r, theta = cartesien_vers_polaire(a, b)
    
    return {
        "forme_cartesienne": f"({a}, {b})",
        "forme_algebrique": forme_algebrique(a, b),
        "forme_algebrique_ingenieur": notation_ingenieur(a, b),
        "forme_polaire": forme_polaire_str(r, theta),
        "forme_polaire_tuple": (r, theta),
        "forme_exponentielle": forme_exponentielle(a, b),
        "forme_euler": forme_euler_string(a, b),
        "forme_trigonometrique": forme_trigonometrique(a, b),
        "forme_phaseur_radians": forme_phaseur(a, b, 'radians'),
        "forme_phaseur_degres": forme_phaseur(a, b, 'degres'),
        "forme_matricielle": forme_matricielle_2x2(a, b),
        "forme_vecteur": forme_vecteur_2d(a, b),
        "module": r,
        "argument_radians": theta,
        "argument_degres": math.degrees(theta),
        "conjugue": conjugue(a, b),
        "partie_reelle": a,
        "partie_imaginaire": b
    }


# ============================================================================
# FORMULES TRIGONOMÉTRIQUES AVEC COMPLEXES
# ============================================================================

def formule_euler_cos(theta):
    """
    Exprime cos(θ) avec la formule d'Euler: cos(θ) = (e^(iθ) + e^(-iθ))/2
    
    Args:
        theta: Angle en radians
    
    Returns:
        Valeur de cos(θ)
    """
    return (math.exp(1j * theta) + math.exp(-1j * theta)).real / 2


def formule_euler_sin(theta):
    """
    Exprime sin(θ) avec la formule d'Euler: sin(θ) = (e^(iθ) - e^(-iθ))/(2i)
    
    Args:
        theta: Angle en radians
    
    Returns:
        Valeur de sin(θ)
    """
    return ((math.exp(1j * theta) - math.exp(-1j * theta)) / (2j)).real


def formule_euler_tan(theta):
    """
    Exprime tan(θ) avec les formules d'Euler
    
    Args:
        theta: Angle en radians
    
    Returns:
        Valeur de tan(θ)
    """
    return formule_euler_sin(theta) / formule_euler_cos(theta)


def linearisation_cos_puissance(n):
    """
    Linéarise cos^n(x) en somme de cosinus
    Utilise la formule de De Moivre et le binôme de Newton
    
    Args:
        n: Puissance
    
    Returns:
        Description textuelle de la linéarisation
    """
    if n == 1:
        return "cos(x)"
    elif n == 2:
        return "(1 + cos(2x))/2"
    elif n == 3:
        return "(3cos(x) + cos(3x))/4"
    else:
        return f"Linéarisation de cos^{n}(x) nécessite un calcul symbolique"


def linearisation_sin_puissance(n):
    """
    Linéarise sin^n(x) en somme de sinus et cosinus
    
    Args:
        n: Puissance
    
    Returns:
        Description textuelle de la linéarisation
    """
    if n == 1:
        return "sin(x)"
    elif n == 2:
        return "(1 - cos(2x))/2"
    elif n == 3:
        return "(3sin(x) - sin(3x))/4"
    else:
        return f"Linéarisation de sin^{n}(x) nécessite un calcul symbolique"


def formule_produit_cos_sin(p, q):
    """
    Transforme cos(px)sin(qx) en somme
    cos(p)sin(q) = [sin(p+q) - sin(p-q)]/2
    
    Args:
        p, q: Coefficients
    
    Returns:
        Description de la formule
    """
    return f"cos({p}x)sin({q}x) = [sin({p+q}x) - sin({p-q}x)]/2"


def formule_produit_cos_cos(p, q):
    """
    Transforme cos(px)cos(qx) en somme
    cos(p)cos(q) = [cos(p+q) + cos(p-q)]/2
    
    Args:
        p, q: Coefficients
    
    Returns:
        Description de la formule
    """
    return f"cos({p}x)cos({q}x) = [cos({p+q}x) + cos({p-q}x)]/2"


def formule_produit_sin_sin(p, q):
    """
    Transforme sin(px)sin(qx) en somme
    sin(p)sin(q) = [cos(p-q) - cos(p+q)]/2
    
    Args:
        p, q: Coefficients
    
    Returns:
        Description de la formule
    """
    return f"sin({p}x)sin({q}x) = [cos({p-q}x) - cos({p+q}x)]/2"


# ============================================================================
# APPLICATIONS GÉOMÉTRIQUES
# ============================================================================

def rotation_plan_complexe(z_a, z_b, angle):
    """
    Effectue une rotation dans le plan complexe autour d'un centre
    
    Args:
        z_a, z_b: Nombre complexe à faire tourner (a, b)
        angle: Angle de rotation en radians
    
    Returns:
        Tuple (a', b') après rotation autour de l'origine
    """
    return rotation_complexe(z_a, z_b, angle)


def homothetie_complexe(z_a, z_b, centre_a, centre_b, rapport):
    """
    Applique une homothétie dans le plan complexe
    
    Args:
        z_a, z_b: Point à transformer
        centre_a, centre_b: Centre de l'homothétie
        rapport: Rapport d'homothétie
    
    Returns:
        Tuple (a', b') après homothétie
    """
    # Translation vers origine
    rel_a = z_a - centre_a
    rel_b = z_b - centre_b
    
    # Homothétie
    new_a = rel_a * rapport
    new_b = rel_b * rapport
    
    # Translation retour
    return (new_a + centre_a, new_b + centre_b)


def similitude_complexe(z_a, z_b, centre_a, centre_b, rapport, angle):
    """
    Applique une similitude (homothétie + rotation) dans le plan complexe
    
    Args:
        z_a, z_b: Point à transformer
        centre_a, centre_b: Centre de la similitude
        rapport: Rapport de similitude
        angle: Angle de rotation
    
    Returns:
        Tuple (a', b') après similitude
    """
    # Translation vers origine
    rel_a = z_a - centre_a
    rel_b = z_b - centre_b
    
    # Similitude (multiplication par rapport·e^(i·angle))
    r = rapport
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    
    new_a = r * (rel_a * cos_angle - rel_b * sin_angle)
    new_b = r * (rel_a * sin_angle + rel_b * cos_angle)
    
    # Translation retour
    return (new_a + centre_a, new_b + centre_b)


def reflexion_axe_reel(a, b):
    """
    Réflexion par rapport à l'axe réel (conjugaison)
    
    Args:
        a, b: Nombre complexe
    
    Returns:
        Tuple (a, -b)
    """
    return conjugue(a, b)


def reflexion_axe_imaginaire(a, b):
    """
    Réflexion par rapport à l'axe imaginaire
    
    Args:
        a, b: Nombre complexe
    
    Returns:
        Tuple (-a, b)
    """
    return (-a, b)


def reflexion_origine(a, b):
    """
    Réflexion par rapport à l'origine (symétrie centrale)
    
    Args:
        a, b: Nombre complexe
    
    Returns:
        Tuple (-a, -b)
    """
    return (-a, -b)
