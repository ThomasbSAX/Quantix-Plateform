import re
import json
import math
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

try:
    import requests
except Exception:
    requests = None

from .logic import detect_sequence_logic_break




'''
Ce code implémente un auditeur automatique de cohérence de données tabulaires (CSV ou DataFrame pandas), combinant une analyse algorithmique déterministe rapide et une interprétation optionnelle par LLM.

Architecture générale.
Le cœur est la classe QuantixCoherenceDescriber. Elle lit un échantillon de données, infère les types et formats de colonnes, détecte des ruptures de cohérence localisées (avec indices exacts), calcule un score global de cohérence, puis peut déléguer à un LLM une synthèse interprétative sans recalcul statistique. Une fonction describe_llm de façade permet un usage direct.

Configuration.
LLMConfig regroupe les paramètres d’appel au LLM (URL API locale type LM Studio, modèles, température, timeout). DescribeConfig fixe les paramètres algorithmiques : taille d’échantillon, seuils statistiques (z-score, MAD), seuils de tolérance (valeurs manquantes, formats cassés), budget temps et convention d’indexation des lignes (0 pandas / 1 humain).

Lecture et échantillonnage.
Si l’entrée est un chemin CSV, seule une tranche initiale (sample_rows) est lue pour limiter le coût. Si l’entrée est un DataFrame, il est utilisé directement. L’analyse est donc O(n_sample × nb_col), stable et bornée en temps.

Inférence de type.
Pour chaque colonne, le code estime des taux de parsing numérique et date. Si ≥ 0.98, le type est considéré pur (float ou date). Entre 0.8 et 0.98, le type est mixte. Sinon, string. Cette décision conditionne les tests appliqués ensuite.

Inférence de format.
Les valeurs string sont transformées en signatures de forme (D pour chiffre, L/l pour lettres, autres caractères conservés). La signature majoritaire définit un format dominant, à partir duquel une regex approximative est construite. Les lignes ne respectant pas cette regex sont signalées comme violations de format.

Détection de séquences d’identifiants.
Le code reconnaît des identifiants de type préfixe + entier (ex. S168). Si la majorité de la colonne suit ce schéma, il teste la monotonie et le pas de la séquence (strictement croissante, pas = 1). Les doublons, inversions ou sauts sont localisés ligne par ligne.

Détection d’ambiguïtés typographiques.
Dans les colonnes textuelles, il repère les confusions classiques OCR ou saisie (O/0, I/1, l/1, S/5). Une correction suggérée est produite, sans modification automatique.

Statistiques numériques et outliers.
Pour les colonnes numériques plausibles, il calcule moyenne, écart-type et quantiles. Les valeurs aberrantes sont détectées par z-score si l’écart-type est non nul, sinon par MAD (robuste). Les indices de lignes concernées sont conservés.

Qualité de colonne.
Chaque colonne reçoit des indicateurs simples : taux de valeurs manquantes, taux d’unicité, caractère constant. Ces métriques alimentent le score global.

Score de cohérence global.
Un score ∈ [0,1] est calculé comme 1 − pénalité moyenne. Les pénalités agrègent : valeurs manquantes excessives, ruptures de format, mélange de types, ruptures de séquence. Le score est déterministe, stable et indépendant du LLM.

Invariants globaux.
À partir des colonnes très stables (formats dominants très confiants, séquences claires), le code extrait des invariants lisibles humainement, interprétables comme règles implicites du jeu de données.

Couche LLM (optionnelle).
Les résultats algorithmiques sont compressés en “evidence packs” (statistiques, exemples, violations). Le LLM est appelé uniquement pour interpréter, hiérarchiser et formuler des règles ou alertes, avec contrainte stricte de sortie JSON. Aucun recalcul de données n’est effectué côté LLM.

Sortie.
La fonction retourne un dictionnaire structuré contenant : métadonnées, score de cohérence, invariants globaux, diagnostics détaillés par colonne (types, formats, stats, violations), duplications de lignes, et éventuellement un résumé LLM.

Intention.
Ce module sert de brique d’audit de cohérence pour pipelines data (type Quantix), orientée détection explicable des anomalies, robuste, rapide, et compatible avec une interprétation sémantique par LLM sans perte de traçabilité.
'''



@dataclass(frozen=True)
class LLMConfig:
    api_url: str = "http://localhost:1234/v1/chat/completions"
    model_map: str = "mistral-small"
    model_reduce: str = "mistral-small"
    temperature: float = 0.0
    timeout: float = 60.0


@dataclass(frozen=True)
class DescribeConfig:
    sample_rows: int = 2000
    max_violations_per_col: int = 80
    max_examples: int = 6
    max_packs_per_col: int = 18
    z_thresh: float = 3.0
    mad_thresh: float = 3.5
    missing_high: float = 0.2
    mixed_type_high: float = 0.01
    format_break_high: float = 0.005
    time_budget_s: float = 8.0
    force_full_scan: bool = False
    index_base: int = 1  # 1 = humain, 0 = pandas
    random_state: int = 0


class QuantixCoherenceDescriber:
    """
    describe_llm(path_or_df, ...) -> dict
    Couche algo: infère formats/grammaires + détecte ruptures (indices exacts).
    Couche LLM: interprète + hiérarchise + propose règles, sans recalcul exhaustif.
    """

    AMBIGUOUS_DIGIT_MAP = {
        "O": "0", "o": "0",
        "I": "1", "l": "1", "|": "1",
        "S": "5", "s": "5",
        "B": "8",
    }

    def __init__(self, llm: Optional[LLMConfig] = None, cfg: Optional[DescribeConfig] = None):
        self.llm = llm or LLMConfig()
        self.cfg = cfg or DescribeConfig()

    @staticmethod
    def _stable_hash(obj: Any) -> str:
        b = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8", errors="ignore")
        return hashlib.sha256(b).hexdigest()[:16]

    @staticmethod
    def _safe_float(x: Any) -> Optional[float]:
        try:
            if pd.isna(x):
                return None
            return float(x)
        except Exception:
            return None

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        # Cherche le plus grand bloc JSON plausible dans la réponse.
        if not text:
            return None
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return None
        blob = m.group(0).strip()
        try:
            return json.loads(blob)
        except Exception:
            return None

    def _row_index_out(self, i0: int) -> int:
        return i0 + self.cfg.index_base

    def _read_sample(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path, nrows=self.cfg.sample_rows, dtype=str, keep_default_na=False)

    def _read_full(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path, dtype=str, keep_default_na=False)

    @staticmethod
    def _infer_numeric(series: pd.Series) -> Tuple[pd.Series, float]:
        # retourne (num_series, parse_rate)
        s = series.replace("", np.nan)
        num = pd.to_numeric(s, errors="coerce")
        rate = float(num.notna().mean()) if len(series) else 0.0
        return num, rate

    @staticmethod
    def _infer_date(series: pd.Series) -> Tuple[pd.Series, float]:
        s = series.replace("", np.nan)
        dt = pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)
        rate = float(dt.notna().mean()) if len(series) else 0.0
        return dt, rate

    @staticmethod
    def _mask_signature(x: str) -> str:
        # signature de forme (lettre/chiffre/autre), utile pour clustering de formats
        out = []
        for ch in x:
            if ch.isdigit():
                out.append("D")
            elif ch.isalpha():
                out.append("L" if ch.isupper() else "l")
            elif ch.isspace():
                out.append(" ")
            else:
                out.append(ch)
        return "".join(out)

    @staticmethod
    def _topk(items: List[Tuple[Any, int]], k: int) -> List[Any]:
        return [a for a, _ in sorted(items, key=lambda t: (-t[1], str(t[0])) )[:k]]

    def _dominant_regex_candidate(self, series: pd.Series) -> Dict[str, Any]:
        s = series.dropna().astype(str)
        s = s[s != ""]
        if s.empty:
            return {"dominant_regex": None, "confidence": 0.0, "signature": None, "examples_ok": [], "examples_bad": []}

        sigs = s.map(self._mask_signature)
        vc = sigs.value_counts()
        top_sig = str(vc.index[0])
        conf = float(vc.iloc[0] / max(1, len(sigs)))

        ok_examples = s[sigs == top_sig].head(6).tolist()
        bad_examples = s[sigs != top_sig].head(6).tolist()

        # Heuristique: dérive une regex approximative depuis la signature
        # L -> [A-Z], l -> [a-z], D -> \d, ' ' -> \s, autres -> échappés
        rx = []
        for ch in top_sig:
            if ch == "L":
                rx.append(r"[A-Z]")
            elif ch == "l":
                rx.append(r"[a-z]")
            elif ch == "D":
                rx.append(r"\d")
            elif ch == " ":
                rx.append(r"\s")
            else:
                rx.append(re.escape(ch))
        regex = r"^" + "".join(rx) + r"$"

        return {
            "dominant_regex": regex,
            "confidence": conf,
            "signature": top_sig,
            "examples_ok": ok_examples[:6],
            "examples_bad": bad_examples[:6],
        }

    def _find_format_violations(self, series: pd.Series, regex: str, maxn: int) -> List[Dict[str, Any]]:
        s = series.fillna("").astype(str)
        ok = s.str.match(regex, na=False)
        bad_idx = np.flatnonzero(~ok.to_numpy())
        out = []
        for i in bad_idx[:maxn]:
            out.append({
                "row": self._row_index_out(int(i)),
                "kind": "format",
                "value": s.iloc[int(i)],
                "expected": regex,
                "severity": "high",
                "note": "ne matche pas le format dominant"
            })
        return out

    def _detect_ambiguous_chars(self, series: pd.Series, maxn: int) -> List[Dict[str, Any]]:
        # détecte occurrences de lettres ambiguës dans des tokens "numériques" (ex: O au lieu de 0)
        s = series.fillna("").astype(str)
        out = []
        # Heuristique: si la valeur contient au moins un digit, et aussi un char ambigu, signaler
        amb = re.compile(r"[OolI|SBs]")
        for i, v in enumerate(s.tolist()):
            if len(out) >= maxn:
                break
            if not v:
                continue
            if any(ch.isdigit() for ch in v) and amb.search(v):
                # propose correction naïve (ne pas appliquer, juste suggérer)
                corr = "".join(self.AMBIGUOUS_DIGIT_MAP.get(ch, ch) for ch in v)
                if corr != v:
                    out.append({
                        "row": self._row_index_out(i),
                        "kind": "ambiguous_char",
                        "value": v,
                        "expected": corr,
                        "severity": "med",
                        "note": "caractères ambigus (O/0, I/1, etc.)"
                    })
        return out

    def _detect_sequence_id(self, series: pd.Series, maxn: int) -> Dict[str, Any]:
        s = series.fillna("").astype(str)
        # Cherche pattern prefix + integer (ex: S168)
        m = s.str.extract(r"^([A-Za-z]+)(\d+)$", expand=True)
        ok = m[0].notna() & m[1].notna() & (s != "")
        rate = float(ok.mean()) if len(s) else 0.0
        
        # Utiliser logic.py pour une détection plus intelligente
        logic_result = None
        if len(s) >= 4:
            seq_list = s.replace("", np.nan).dropna().tolist()
            if len(seq_list) >= 4:
                try:
                    logic_result = detect_sequence_logic_break(
                        seq_list,
                        index_base=self.cfg.index_base,
                        min_learn=min(3, len(seq_list) - 1)
                    )
                except Exception:
                    pass
        
        if rate < 0.8:
            result = {
                "is_monotone": None, 
                "is_step1": None, 
                "breaks": [], 
                "parse_rate": rate, 
                "rule": None,
                "logic_analysis": logic_result
            }
            # Si logic.py a détecté quelque chose même avec rate < 0.8
            if logic_result and logic_result.get("rule"):
                if "break_at" in logic_result:
                    result["breaks"].append({
                        "row": logic_result["break_at"],
                        "kind": "sequence_logic",
                        "value": logic_result.get("observed"),
                        "expected": logic_result.get("expected"),
                        "severity": "high",
                        "note": f"{logic_result['rule']}: {logic_result.get('note', 'rupture détectée')}"
                    })
            return result

        prefix = m.loc[ok, 0].mode().iloc[0]
        nums = pd.to_numeric(m.loc[ok, 1], errors="coerce")
        # Remet dans l'ordre original, en gardant NaN là où non parseable
        k = pd.Series(np.nan, index=s.index, dtype=float)
        k.loc[ok] = nums.astype(float)

        # breaks de forme (non parseable)
        breaks = []
        bad_idx = np.flatnonzero(~ok.to_numpy())
        for i in bad_idx[:maxn]:
            breaks.append({
                "row": self._row_index_out(int(i)),
                "kind": "sequence",
                "value": s.iloc[int(i)],
                "expected": f"{prefix}<int>",
                "severity": "high",
                "note": "ID non parseable (prefix+entier) dans une colonne majoritairement séquentielle"
            })
        if len(breaks) >= maxn:
            return {
                "is_monotone": None, 
                "is_step1": None, 
                "breaks": breaks, 
                "parse_rate": rate, 
                "rule": f"^{prefix}\\d+$",
                "logic_analysis": logic_result
            }

        # tests monotonicité/step1 sur positions valides consécutives
        kv = k.dropna().astype(int)
        if kv.empty or len(kv) < 3:
            return {
                "is_monotone": None, 
                "is_step1": None, 
                "breaks": breaks, 
                "parse_rate": rate, 
                "rule": f"^{prefix}\\d+$",
                "logic_analysis": logic_result
            }

        # monotone sur indices d'apparition
        vals = kv.to_numpy()
        diffs = np.diff(vals)
        is_monotone = bool(np.all(diffs > 0))
        is_step1 = bool(np.all(diffs == 1))

        if not is_monotone or not is_step1:
            # localise les ruptures sur les paires consécutives valides
            idxs = kv.index.to_numpy()
            for j in range(len(diffs)):
                if len(breaks) >= maxn:
                    break
                if diffs[j] <= 0:
                    breaks.append({
                        "row": self._row_index_out(int(idxs[j+1])),
                        "kind": "sequence",
                        "value": s.loc[idxs[j+1]],
                        "expected": f"{prefix}{int(vals[j]+1)}",
                        "severity": "high",
                        "note": "non monotone (doublon / inversion)"
                    })
                elif diffs[j] != 1:
                    breaks.append({
                        "row": self._row_index_out(int(idxs[j+1])),
                        "kind": "sequence",
                        "value": s.loc[idxs[j+1]],
                        "expected": f"{prefix}{int(vals[j]+1)}",
                        "severity": "high",
                        "note": f"saut de séquence (delta={int(diffs[j])})"
                    })

        return {
            "is_monotone": is_monotone,
            "is_step1": is_step1,
            "breaks": breaks,
            "parse_rate": rate,
            "rule": f"^{re.escape(prefix)}\\d+$",
            "logic_analysis": logic_result
        }

    @staticmethod
    def _robust_outliers_mad(x: pd.Series, thresh: float) -> np.ndarray:
        a = x.dropna().to_numpy(dtype=float)
        if a.size < 10:
            return np.array([], dtype=int)
        med = np.median(a)
        mad = np.median(np.abs(a - med))
        if mad == 0:
            return np.array([], dtype=int)
        score = 0.6745 * (a - med) / mad
        return np.flatnonzero(np.abs(score) > thresh)

    def _numeric_stats_and_outliers(self, series: pd.Series) -> Dict[str, Any]:
        num, rate = self._infer_numeric(series)
        out = {"parse_rate": rate, "mean": None, "std": None, "q05": None, "q50": None, "q95": None, "z_outliers": 0, "outlier_rows": []}
        if rate < 0.8 or num.dropna().empty:
            return out
        a = num.dropna().astype(float)
        out["mean"] = float(a.mean())
        out["std"] = float(a.std(ddof=1)) if len(a) > 1 else 0.0
        out["q05"] = float(a.quantile(0.05))
        out["q50"] = float(a.quantile(0.50))
        out["q95"] = float(a.quantile(0.95))

        # z-score si std > 0, sinon MAD
        if out["std"] and out["std"] > 0:
            z = (a - out["mean"]) / out["std"]
            bad = z.abs() > self.cfg.z_thresh
            out["z_outliers"] = int(bad.sum())
            if out["z_outliers"]:
                # indices des outliers (dans l'index original)
                idxs = a.index[bad].to_numpy()
                out["outlier_rows"] = [int(i) + self.cfg.index_base for i in idxs[:min(40, len(idxs))]]
        else:
            # fallback MAD
            idxs = self._robust_outliers_mad(num, self.cfg.mad_thresh)
            out["z_outliers"] = int(len(idxs))
            if len(idxs):
                out["outlier_rows"] = [int(num.dropna().index[i]) + self.cfg.index_base for i in idxs[:min(40, len(idxs))]]
        return out

    def _build_evidence_packs(self, report: Dict[str, Any]) -> Dict[str, Any]:
        packs = {"global": {}, "columns": {}}
        packs["global"] = {
            "meta": report.get("meta", {}),
            "coherence_score": report.get("coherence_score", None),
            "global_invariants": report.get("global_invariants", [])[:10],
        }
        for col, c in report.get("columns", {}).items():
            v = c.get("violations", [])
            packs["columns"][col] = {
                "type_inferred": c.get("type_inferred"),
                "format": c.get("format"),
                "quality": c.get("quality"),
                "stats": c.get("stats"),
                "sequence": c.get("sequence"),
                "violations_top": v[: min(len(v), 18)],
            }
        return packs

    def _generate_automatic_report(self, report: Dict[str, Any]) -> str:
        """
        Génère un rapport automatique lisible des problèmes détectés.
        """
        lines = []
        lines.append("=" * 80)
        lines.append("RAPPORT AUTOMATIQUE D'ANALYSE DE COHÉRENCE")
        lines.append("=" * 80)
        lines.append("")
        
        # Informations générales
        meta = report.get("meta", {})
        lines.append(f"Fichier: {meta.get('file', 'N/A')}")
        lines.append(f"Lignes analysées: {meta.get('sampled_rows', 'N/A')} / {meta.get('rows', 'Total inconnu')}")
        lines.append(f"Colonnes: {meta.get('cols', 'N/A')}")
        lines.append(f"Score de cohérence: {report.get('coherence_score', 0):.2%}")
        lines.append("")
        
        # Problèmes critiques détectés
        total_issues = 0
        critical_issues = []
        high_issues = []
        medium_issues = []
        
        for col, c in report.get("columns", {}).items():
            violations = c.get("violations", [])
            stats = c.get("stats", {})
            quality = c.get("quality", {})
            sequence = c.get("sequence", {})
            
            # Outliers numériques
            if stats.get("z_outliers", 0) > 0:
                outlier_rows = stats.get("outlier_rows", [])
                issue = {
                    "type": "outliers_numeriques",
                    "severity": "high" if stats["z_outliers"] > 10 else "medium",
                    "column": col,
                    "count": stats["z_outliers"],
                    "rows": outlier_rows[:10],
                    "details": f"{stats['z_outliers']} valeur(s) aberrante(s) détectée(s)"
                }
                if issue["severity"] == "high":
                    high_issues.append(issue)
                else:
                    medium_issues.append(issue)
                total_issues += stats["z_outliers"]
            
            # Violations de séquence et logique
            for v in violations:
                issue = {
                    "type": v.get("kind", "unknown"),
                    "severity": v.get("severity", "low"),
                    "column": col,
                    "row": v.get("row"),
                    "value": v.get("value"),
                    "expected": v.get("expected"),
                    "note": v.get("note", "")
                }
                
                if v.get("severity") == "high":
                    high_issues.append(issue)
                elif v.get("severity") == "med":
                    medium_issues.append(issue)
                total_issues += 1
            
            # Analyse de logique avancée
            logic_analysis = sequence.get("logic_analysis")
            if logic_analysis and logic_analysis.get("rule"):
                if logic_analysis.get("status") == "ok":
                    lines.append(f"✓ Colonne '{col}': Logique {logic_analysis['rule']} respectée")
                elif "break_at" in logic_analysis:
                    critical_issues.append({
                        "type": "rupture_logique",
                        "severity": "critical",
                        "column": col,
                        "row": logic_analysis["break_at"],
                        "rule": logic_analysis["rule"],
                        "expected": logic_analysis.get("expected"),
                        "observed": logic_analysis.get("observed"),
                        "note": logic_analysis.get("note", "")
                    })
                    total_issues += 1
            
            # Taux de valeurs manquantes élevé
            if quality.get("missing_rate", 0) > 0.3:
                high_issues.append({
                    "type": "valeurs_manquantes",
                    "severity": "high",
                    "column": col,
                    "rate": quality["missing_rate"],
                    "details": f"{quality['missing_rate']:.1%} de valeurs manquantes"
                })
                total_issues += 1
        
        lines.append("")
        lines.append(f"RÉSUMÉ: {total_issues} problème(s) détecté(s)")
        lines.append("")
        
        # Problèmes critiques
        if critical_issues:
            lines.append("🔴 PROBLÈMES CRITIQUES:")
            lines.append("-" * 80)
            for issue in critical_issues[:20]:
                lines.append(f"  • Colonne '{issue['column']}' - Ligne {issue.get('row', 'N/A')}")
                lines.append(f"    Type: {issue['type']}")
                if issue.get("rule"):
                    lines.append(f"    Règle: {issue['rule']}")
                lines.append(f"    Attendu: {issue.get('expected', 'N/A')}")
                lines.append(f"    Observé: {issue.get('observed', 'N/A')}")
                lines.append(f"    Note: {issue.get('note', '')}")
                lines.append("")
        
        # Problèmes importants
        if high_issues:
            lines.append("🟠 PROBLÈMES IMPORTANTS:")
            lines.append("-" * 80)
            for issue in high_issues[:30]:
                lines.append(f"  • Colonne '{issue['column']}'")
                if issue['type'] == "outliers_numeriques":
                    lines.append(f"    {issue['details']}")
                    if issue.get("rows"):
                        lines.append(f"    Lignes concernées: {', '.join(map(str, issue['rows'][:5]))}" + 
                                   (" ..." if len(issue['rows']) > 5 else ""))
                elif issue['type'] == "valeurs_manquantes":
                    lines.append(f"    {issue['details']}")
                else:
                    lines.append(f"    Ligne {issue.get('row', 'N/A')}: {issue.get('note', '')}")
                    if issue.get('value'):
                        lines.append(f"    Valeur: '{issue['value']}' → Attendu: '{issue.get('expected', 'N/A')}'")
                lines.append("")
        
        # Problèmes moyens
        if medium_issues and len(medium_issues) <= 20:
            lines.append("🟡 PROBLÈMES MOYENS:")
            lines.append("-" * 80)
            for issue in medium_issues[:20]:
                lines.append(f"  • Colonne '{issue['column']}' - Ligne {issue.get('row', 'N/A')}")
                lines.append(f"    {issue.get('note', issue['type'])}")
                lines.append("")
        elif medium_issues:
            lines.append(f"🟡 {len(medium_issues)} problème(s) de sévérité moyenne détecté(s)")
            lines.append("")
        
        # Invariants détectés
        invariants = report.get("global_invariants", [])
        if invariants:
            lines.append("📋 RÈGLES/INVARIANTS DÉTECTÉS:")
            lines.append("-" * 80)
            for inv in invariants:
                lines.append(f"  ✓ {inv}")
            lines.append("")
        
        # Duplications de lignes
        dup_rows = report.get("row_level", {}).get("duplicate_rows", [])
        if dup_rows:
            lines.append(f"⚠️  LIGNES DUPLIQUÉES: {len(dup_rows)} ligne(s)")
            lines.append(f"   Lignes: {', '.join(map(str, dup_rows[:20]))}" + 
                       (" ..." if len(dup_rows) > 20 else ""))
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)

    def _llm_call(self, model: str, messages: List[Dict[str, str]]) -> Optional[dict]:
        payload = {"model": model, "messages": messages, "temperature": self.llm.temperature}
        try:
            r = requests.post(self.llm.api_url, json=payload, timeout=self.llm.timeout)
            r.raise_for_status()
            data = r.json()
            txt = data["choices"][0]["message"]["content"]
            return self._extract_json(txt) or {"raw": txt}
        except Exception as e:
            return {"error": str(e)}

    def _llm_interpret(self, packs: Dict[str, Any]) -> Dict[str, Any]:
        schema = {
            "llm_summary": {
                "one_paragraph": "string",
                "top_issues": [{"severity": "low|med|high", "where": "string", "what": "string", "rows": [123]}],
                "inferred_invariants": ["string"],
                "recommended_rules": [{"target": "col|global", "rule": "string", "risk": "string"}]
            }
        }
        prompt = {
            "task": "Tu es un auditeur de cohérence de données. Tu ne recalcules rien: tu interprètes uniquement les indicateurs et exemples fournis.",
            "output_rule": "Réponds STRICTEMENT en JSON. Aucune phrase hors JSON.",
            "schema": schema,
            "input": packs
        }
        messages = [
            {"role": "system", "content": "Répondre en JSON strict, sans texte hors JSON."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
        ]
        return self._llm_call(self.llm.model_reduce, messages) or {}

    def describe_llm(
        self,
        path_or_df: Union[str, pd.DataFrame],
        *,
        use_llm: bool = True,
        file_name: Optional[str] = None,
        print_report: bool = True
    ) -> Dict[str, Any]:
        t0 = time.time()
        rng = np.random.default_rng(self.cfg.random_state)

        if isinstance(path_or_df, str):
            path = path_or_df
            df_sample = self._read_sample(path)
            df = None
            rows_total = None
            try:
                # estimation rapide de rows: on évite si budget serré
                pass
            except Exception:
                pass
            file_label = file_name or path
        else:
            df_sample = path_or_df.copy()
            df = path_or_df
            file_label = file_name or "<dataframe>"

        meta = {
            "file": file_label,
            "rows": int(df.shape[0]) if isinstance(df, pd.DataFrame) else None,
            "cols": int(df_sample.shape[1]),
            "sampled_rows": int(df_sample.shape[0]),
        }

        report: Dict[str, Any] = {
            "meta": meta,
            "coherence_score": None,
            "global_invariants": [],
            "columns": {},
            "row_level": {"duplicate_rows": [], "schema_breaks": [], "delimiter_breaks": []},
            "recommendations": [],
            "llm_summary": {"one_paragraph": "", "top_issues": []},
            "automatic_report": ""
        }

        # duplicates lignes (échantillon + full si petit)
        df_for_dups = df_sample
        try:
            dup_idx = df_for_dups.duplicated(keep=False)
            if dup_idx.any():
                rows = np.flatnonzero(dup_idx.to_numpy())[:40]
                report["row_level"]["duplicate_rows"] = [self._row_index_out(int(i)) for i in rows]
        except Exception:
            pass

        for col in df_sample.columns:
            if time.time() - t0 > self.cfg.time_budget_s:
                break

            s = df_sample[col].replace("", np.nan)

            quality = {
                "missing_rate": float(s.isna().mean()) if len(s) else 0.0,
                "unique_rate": float(s.nunique(dropna=True) / max(1, s.notna().sum())),
                "constant_rate": float((s.nunique(dropna=True) <= 1)),
            }

            # type inference (sur sample)
            num, num_rate = self._infer_numeric(df_sample[col])
            dt, dt_rate = self._infer_date(df_sample[col])

            if num_rate >= 0.98:
                t_inf = "float"
            elif dt_rate >= 0.98:
                t_inf = "date"
            elif max(num_rate, dt_rate) >= 0.8:
                t_inf = "mixed"
            else:
                t_inf = "string"

            fmt = self._dominant_regex_candidate(df_sample[col].replace("", np.nan))
            violations: List[Dict[str, Any]] = []

            # séquence potentielle
            seq = {"is_monotone": None, "is_step1": None, "breaks": [], "parse_rate": 0.0, "rule": None}
            if t_inf in ("string", "mixed"):
                seq = self._detect_sequence_id(df_sample[col], maxn=min(30, self.cfg.max_violations_per_col))
                if seq.get("parse_rate", 0.0) and seq["parse_rate"] >= 0.8:
                    violations.extend(seq.get("breaks", [])[: self.cfg.max_violations_per_col])

            # violations format (sample)
            if fmt.get("dominant_regex") and fmt.get("confidence", 0.0) >= 0.7 and t_inf in ("string", "mixed"):
                violations.extend(self._find_format_violations(
                    df_sample[col].replace("", np.nan),
                    fmt["dominant_regex"],
                    maxn=min(40, self.cfg.max_violations_per_col),
                ))

            # ambiguïtés (sample)
            if t_inf in ("string", "mixed"):
                violations.extend(self._detect_ambiguous_chars(
                    df_sample[col],
                    maxn=min(30, self.cfg.max_violations_per_col),
                ))

            stats = self._numeric_stats_and_outliers(df_sample[col]) if t_inf in ("float", "mixed") else {
                "parse_rate": float(num_rate), "mean": None, "std": None, "q05": None, "q50": None, "q95": None, "z_outliers": 0, "outlier_rows": []
            }

            report["columns"][col] = {
                "type_inferred": t_inf,
                "format": fmt,
                "quality": quality,
                "stats": stats,
                "sequence": seq if seq.get("rule") else {"is_monotone": None, "is_step1": None, "breaks": [], "parse_rate": seq.get("parse_rate", 0.0), "rule": None},
                "violations": violations[: self.cfg.max_violations_per_col],
            }

        # score cohérence déterministe (simple, stable)
        penalties = []
        for col, c in report["columns"].items():
            q = c["quality"]
            v = c["violations"]
            fmt_conf = (c.get("format") or {}).get("confidence", 0.0) or 0.0
            format_break_rate = 0.0
            if fmt_conf >= 0.7 and len(df_sample):
                format_break_rate = min(1.0, len([x for x in v if x["kind"] == "format"]) / len(df_sample))

            mixed_pen = 0.0
            if c["type_inferred"] == "mixed":
                # approx: si parse_rate numérique ou date est élevé mais pas total => mélange
                mixed_pen = max(0.0, 1.0 - max(c["stats"]["parse_rate"], c["sequence"].get("parse_rate", 0.0), 0.0))

            seq_break = 1.0 if any(x["kind"] == "sequence" for x in v) else 0.0
            penalties.append(
                0.40 * min(1.0, q["missing_rate"] / max(self.cfg.missing_high, 1e-6)) +
                0.30 * min(1.0, format_break_rate / max(self.cfg.format_break_high, 1e-6)) +
                0.20 * min(1.0, mixed_pen) +
                0.10 * seq_break
            )
        pen = float(np.mean(penalties)) if penalties else 0.0
        score = float(max(0.0, 1.0 - pen))
        report["coherence_score"] = score

        # invariants globaux (heuristiques simples)
        invariants = []
        for col, c in report["columns"].items():
            seq = c.get("sequence", {})
            if seq.get("rule") and seq.get("parse_rate", 0.0) >= 0.8:
                rule = seq["rule"]
                if seq.get("is_step1") is True:
                    invariants.append(f"{col}: identifiants de la forme {rule} et séquence pas=1")
                else:
                    invariants.append(f"{col}: identifiants de la forme {rule} (séquence non garantie)")
            fmt = c.get("format", {})
            if fmt.get("dominant_regex") and fmt.get("confidence", 0.0) >= 0.85:
                invariants.append(f"{col}: format dominant {fmt['dominant_regex']} (conf={fmt['confidence']:.2f})")
        report["global_invariants"] = invariants[:20]

        # LLM interprétation (optionnel)
        if use_llm and (time.time() - t0) < self.cfg.time_budget_s:
            packs = self._build_evidence_packs(report)
            llm_out = self._llm_interpret(packs)
            if isinstance(llm_out, dict) and "llm_summary" in llm_out:
                report["llm_summary"] = llm_out["llm_summary"]
            elif isinstance(llm_out, dict) and "raw" in llm_out:
                report["llm_summary"] = {"one_paragraph": llm_out["raw"][:1500], "top_issues": []}
            elif isinstance(llm_out, dict) and "error" in llm_out:
                report["llm_summary"] = {"one_paragraph": f"LLM error: {llm_out['error']}", "top_issues": []}

        # Génération du rapport automatique
        report["automatic_report"] = self._generate_automatic_report(report)
        
        # Affichage optionnel du rapport
        if print_report:
            print(report["automatic_report"])

        return report


def describe_llm(
    path_or_df: Union[str, pd.DataFrame],
    *,
    llm_api_url: str = "http://localhost:1234/v1/chat/completions",
    model_map: str = "mistral-small",
    model_reduce: str = "mistral-small",
    use_llm: bool = True,
    time_budget_s: float = 8.0,
    sample_rows: int = 2000,
    index_base: int = 1,
    print_report: bool = True
) -> Dict[str, Any]:
    llm = LLMConfig(api_url=llm_api_url, model_map=model_map, model_reduce=model_reduce)
    cfg = DescribeConfig(time_budget_s=time_budget_s, sample_rows=sample_rows, index_base=index_base)
    return QuantixCoherenceDescriber(llm=llm, cfg=cfg).describe_llm(path_or_df, use_llm=use_llm, print_report=print_report)
