import re
from typing import List, Dict, Any, Optional, Tuple
from fractions import Fraction


def detect_sequence_logic_break(
    seq: List[str],
    *,
    index_base: int = 1,
    min_learn: int = 3
) -> Dict[str, Any]:
    """
    Détecte une logique de progression dominante et la première rupture.
    Supporte: progressions de caractères, numériques (arithmétiques/géométriques),
    préfixes/suffixes, patterns composites, et plus encore.
    Aucune connaissance a priori (pas de table d'ambiguïtés).
    """

    def out(i: int) -> int:
        return i + index_base

    n = len(seq)
    if n < min_learn + 1:
        return {"status": "insufficient_data"}

    # ==========================================================
    # 0) Cas : séquence purement numérique (entiers ou décimaux)
    # ==========================================================
    try:
        numbers = [float(x) for x in seq[:min_learn]]
        
        # Vérifier progression arithmétique
        deltas = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
        if len(set(deltas)) == 1:
            step = deltas[0]
            for i in range(min_learn, n):
                try:
                    val = float(seq[i])
                except ValueError:
                    return {
                        "rule": "arithmetic_progression",
                        "step": step,
                        "break_at": out(i),
                        "expected": numbers[-1] + step,
                        "observed": seq[i],
                        "note": "valeur non numérique"
                    }
                expected = numbers[-1] + step
                if abs(val - expected) > 1e-9:
                    return {
                        "rule": "arithmetic_progression",
                        "step": step,
                        "break_at": out(i),
                        "expected": expected,
                        "observed": val,
                        "note": "rupture de progression arithmétique"
                    }
                numbers.append(val)
            return {
                "rule": "arithmetic_progression",
                "step": step,
                "status": "ok"
            }
        
        # Vérifier progression géométrique
        if all(numbers[i] != 0 for i in range(len(numbers)-1)):
            ratios = [numbers[i+1] / numbers[i] for i in range(len(numbers)-1)]
            if len(set(ratios)) == 1 and abs(ratios[0] - 1.0) > 1e-9:
                ratio = ratios[0]
                for i in range(min_learn, n):
                    try:
                        val = float(seq[i])
                    except ValueError:
                        return {
                            "rule": "geometric_progression",
                            "ratio": ratio,
                            "break_at": out(i),
                            "expected": numbers[-1] * ratio,
                            "observed": seq[i],
                            "note": "valeur non numérique"
                        }
                    expected = numbers[-1] * ratio
                    if abs(val - expected) > 1e-9 * abs(expected):
                        return {
                            "rule": "geometric_progression",
                            "ratio": ratio,
                            "break_at": out(i),
                            "expected": expected,
                            "observed": val,
                            "note": "rupture de progression géométrique"
                        }
                    numbers.append(val)
                return {
                    "rule": "geometric_progression",
                    "ratio": ratio,
                    "status": "ok"
                }
    except ValueError:
        pass


    # ==========================================================
    # 1) Cas : progression caractère unique (A B C ... / a b c ...)
    # ==========================================================
    if all(isinstance(x, str) and len(x) == 1 for x in seq[:min_learn]):
        codes = [ord(x) for x in seq[:min_learn]]
        deltas = [codes[i+1] - codes[i] for i in range(len(codes)-1)]

        if len(set(deltas)) == 1:
            step = deltas[0]
            for i in range(min_learn, n):
                if len(seq[i]) != 1:
                    return {
                        "rule": "char_code_progression",
                        "step": step,
                        "break_at": out(i),
                        "expected": chr(ord(seq[i-1]) + step),
                        "observed": seq[i],
                        "note": "longueur invalide"
                    }
                if ord(seq[i]) != ord(seq[i-1]) + step:
                    return {
                        "rule": "char_code_progression",
                        "step": step,
                        "break_at": out(i),
                        "expected": chr(ord(seq[i-1]) + step),
                        "observed": seq[i],
                        "note": "rupture de progression symbolique"
                    }
            return {
                "rule": "char_code_progression",
                "step": step,
                "status": "ok"
            }

    # ==========================================================
    # 2) Cas : pattern composite (lettre + nombre, ex: A1, B2, C3)
    # ==========================================================
    pattern_match = re.match(r'^([A-Za-z]+)(\d+)$', seq[0])
    if pattern_match and all(re.match(r'^([A-Za-z]+)(\d+)$', x) for x in seq[:min_learn]):
        matches = [re.match(r'^([A-Za-z]+)(\d+)$', x) for x in seq[:min_learn]]
        letters = [m.group(1) for m in matches]
        numbers = [int(m.group(2)) for m in matches]
        
        # Vérifier si les lettres progressent
        letters_progress = all(len(letters[i]) == 1 for i in range(len(letters)))
        if letters_progress:
            letter_deltas = [ord(letters[i+1]) - ord(letters[i]) for i in range(len(letters)-1)]
            letter_step = letter_deltas[0] if len(set(letter_deltas)) == 1 else None
        else:
            letter_step = None
            
        # Vérifier si les nombres progressent
        num_deltas = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
        num_step = num_deltas[0] if len(set(num_deltas)) == 1 else None
        
        if letter_step is not None or num_step is not None:
            for i in range(min_learn, n):
                match = re.match(r'^([A-Za-z]+)(\d+)$', seq[i])
                if not match:
                    return {
                        "rule": "composite_pattern",
                        "letter_step": letter_step,
                        "num_step": num_step,
                        "break_at": out(i),
                        "expected": f"{chr(ord(letters[-1]) + letter_step) if letter_step else letters[-1]}{numbers[-1] + num_step if num_step else numbers[-1]}",
                        "observed": seq[i],
                        "note": "format composite invalide"
                    }
                letter, num = match.group(1), int(match.group(2))
                
                expected_letter = chr(ord(letters[-1]) + letter_step) if letter_step else letters[-1]
                expected_num = numbers[-1] + num_step if num_step else numbers[-1]
                
                if (letter_step and letter != expected_letter) or (num_step and num != expected_num):
                    return {
                        "rule": "composite_pattern",
                        "letter_step": letter_step,
                        "num_step": num_step,
                        "break_at": out(i),
                        "expected": f"{expected_letter}{expected_num}",
                        "observed": seq[i],
                        "note": "rupture de pattern composite"
                    }
                letters.append(letter)
                numbers.append(num)
            return {
                "rule": "composite_pattern",
                "letter_step": letter_step,
                "num_step": num_step,
                "status": "ok"
            }



    # ==========================================================
    # 3) Cas : structure commune + partie variable
    #    ex: S1 S2 S3 / item_001 item_002 ...
    # ==========================================================
    def split_common_prefix(a: str, b: str) -> Optional[Tuple[str, str, str]]:
        """retourne (prefix, a_suffix, b_suffix)"""
        i = 0
        while i < min(len(a), len(b)) and a[i] == b[i]:
            i += 1
        if i == 0:
            return None
        return a[:i], a[i:], b[i:]

    def split_common_suffix(a: str, b: str) -> Optional[Tuple[str, str, str]]:
        """retourne (a_prefix, b_prefix, suffix)"""
        i = 0
        while i < min(len(a), len(b)) and a[-(i+1)] == b[-(i+1)]:
            i += 1
        if i == 0:
            return None
        return a[:-i] if i > 0 else a, b[:-i] if i > 0 else b, a[-i:] if i > 0 else ""

    # Essayer avec préfixe commun
    parts_prefix = []
    for i in range(min_learn - 1):
        s = split_common_prefix(seq[i], seq[i+1])
        if s is None:
            break
        parts_prefix.append(s)

    if len(parts_prefix) == min_learn - 1:
        prefix = parts_prefix[0][0]

        # vérifier que le préfixe est invariant
        if all(p[0] == prefix for p in parts_prefix):
            # extraire parties variables
            vars_ = []
            for x in seq[:min_learn]:
                if not x.startswith(prefix):
                    break
                vars_.append(x[len(prefix):])
            else:
                # tenter progression numérique
                try:
                    nums = [int(v) for v in vars_]
                    step = nums[1] - nums[0]
                    if all(nums[i+1] - nums[i] == step for i in range(len(nums)-1)):
                        for i in range(min_learn, n):
                            x = seq[i]
                            if not x.startswith(prefix):
                                return {
                                    "rule": "prefix_plus_integer",
                                    "prefix": prefix,
                                    "break_at": out(i),
                                    "expected": f"{prefix}{nums[-1] + step}",
                                    "observed": x,
                                    "note": "préfixe modifié"
                                }
                            try:
                                v = int(x[len(prefix):])
                            except ValueError:
                                return {
                                    "rule": "prefix_plus_integer",
                                    "prefix": prefix,
                                    "break_at": out(i),
                                    "expected": f"{prefix}{nums[-1] + step}",
                                    "observed": x,
                                    "note": "partie variable non numérique"
                                }
                            if v != nums[-1] + step:
                                return {
                                    "rule": "prefix_plus_integer",
                                    "prefix": prefix,
                                    "step": step,
                                    "break_at": out(i),
                                    "expected": f"{prefix}{nums[-1] + step}",
                                    "observed": x,
                                    "note": "rupture de progression numérique"
                                }
                            nums.append(v)
                        return {
                            "rule": "prefix_plus_integer",
                            "prefix": prefix,
                            "step": step,
                            "status": "ok"
                        }
                except ValueError:
                    pass

    # Essayer avec suffixe commun
    parts_suffix = []
    for i in range(min_learn - 1):
        s = split_common_suffix(seq[i], seq[i+1])
        if s is None:
            break
        parts_suffix.append(s)

    if len(parts_suffix) == min_learn - 1:
        suffix = parts_suffix[0][2]

        # vérifier que le suffixe est invariant
        if all(p[2] == suffix for p in parts_suffix):
            # extraire parties variables
            vars_ = []
            for x in seq[:min_learn]:
                if not x.endswith(suffix):
                    break
                vars_.append(x[:-len(suffix)] if suffix else x)
            else:
                # tenter progression numérique
                try:
                    nums = [int(v) for v in vars_]
                    step = nums[1] - nums[0]
                    if all(nums[i+1] - nums[i] == step for i in range(len(nums)-1)):
                        for i in range(min_learn, n):
                            x = seq[i]
                            if not x.endswith(suffix):
                                return {
                                    "rule": "integer_plus_suffix",
                                    "suffix": suffix,
                                    "break_at": out(i),
                                    "expected": f"{nums[-1] + step}{suffix}",
                                    "observed": x,
                                    "note": "suffixe modifié"
                                }
                            try:
                                v = int(x[:-len(suffix)] if suffix else x)
                            except ValueError:
                                return {
                                    "rule": "integer_plus_suffix",
                                    "suffix": suffix,
                                    "break_at": out(i),
                                    "expected": f"{nums[-1] + step}{suffix}",
                                    "observed": x,
                                    "note": "partie variable non numérique"
                                }
                            if v != nums[-1] + step:
                                return {
                                    "rule": "integer_plus_suffix",
                                    "suffix": suffix,
                                    "step": step,
                                    "break_at": out(i),
                                    "expected": f"{nums[-1] + step}{suffix}",
                                    "observed": x,
                                    "note": "rupture de progression numérique"
                                }
                            nums.append(v)
                        return {
                            "rule": "integer_plus_suffix",
                            "suffix": suffix,
                            "step": step,
                            "status": "ok"
                        }
                except ValueError:
                    pass

    # ==========================================================
    # 4) Cas : pattern de répétition cyclique
    # ==========================================================
    for cycle_len in range(2, min_learn):
        if min_learn % cycle_len == 0:
            pattern = seq[:cycle_len]
            matches = True
            for i in range(cycle_len, min_learn):
                if seq[i] != pattern[i % cycle_len]:
                    matches = False
                    break
            
            if matches:
                for i in range(min_learn, n):
                    expected = pattern[i % cycle_len]
                    if seq[i] != expected:
                        return {
                            "rule": "cyclic_repetition",
                            "pattern": pattern,
                            "cycle_length": cycle_len,
                            "break_at": out(i),
                            "expected": expected,
                            "observed": seq[i],
                            "note": "rupture de cycle"
                        }
                return {
                    "rule": "cyclic_repetition",
                    "pattern": pattern,
                    "cycle_length": cycle_len,
                    "status": "ok"
                }



    # ==========================================================
    # 5) Rien de stable détecté
    # ==========================================================
    return {
        "rule": None,
        "status": "no_dominant_logic_detected"
    }
