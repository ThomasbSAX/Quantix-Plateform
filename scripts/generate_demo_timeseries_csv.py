from __future__ import annotations

import math
import random
from datetime import date, timedelta
from pathlib import Path


def main() -> int:
    random.seed(42)

    root = Path(__file__).resolve().parents[1]
    out_path = root / "data" / "demo_math_lab_timeseries.csv"

    start = date(2025, 1, 1)
    n = 365

    regions = ["Nord", "Sud", "Ouest", "Est"]
    produits = ["Alpha", "Beta", "Gamma", "Delta"]
    canaux = ["Web", "Magasin", "Partenaire"]

    comments = [
        "Livraison rapide",
        "Bon rapport qualité/prix",
        "Service correct",
        "RAS",
        "Très satisfait",
        "Retard de livraison",
        "Support lent",
        "Packaging abîmé",
        "Recommande",
        "Moyen",
    ]

    lines = [
        "date,region,produit,canal,visites,ventes,cout,profit,quantite,remise,satisfaction,temperature,commentaire"
    ]

    for i in range(n):
        d = start + timedelta(days=i)

        region = regions[i % len(regions)]
        produit = produits[(i * 3) % len(produits)]
        canal = canaux[(i * 7) % len(canaux)]

        # saisonnalité (hebdo + annuelle) + tendance légère
        yearly = math.sin(2 * math.pi * i / 365.0)
        weekly = math.sin(2 * math.pi * i / 7.0)
        trend = i / 365.0

        # visites (volume) : >0
        visites = int(800 + 250 * yearly + 120 * weekly + 200 * trend + random.gauss(0, 60))
        visites = max(visites, 50)

        # taux de conversion dépend du canal
        base_conv = {"Web": 0.020, "Magasin": 0.028, "Partenaire": 0.018}[canal]
        conv = base_conv + 0.003 * weekly + random.gauss(0, 0.002)
        conv = max(min(conv, 0.06), 0.003)

        quantite = int(visites * conv)
        quantite = max(quantite, 1)

        # prix moyen selon produit
        base_price = {"Alpha": 95, "Beta": 70, "Gamma": 120, "Delta": 150}[produit]
        prix = base_price * (1.0 + 0.05 * yearly + random.gauss(0, 0.03))
        prix = max(prix, 5)

        remise = max(min(0.05 + 0.08 * max(0.0, weekly) + random.gauss(0, 0.01), 0.25), 0.0)

        ventes = quantite * prix * (1.0 - remise)
        cout = ventes * (0.62 + random.gauss(0, 0.03))
        profit = ventes - cout

        # satisfaction (1..5) corrélée au profit (grossièrement) + bruit
        sat = 3.5 + 0.8 * math.tanh(profit / 2000.0) + random.gauss(0, 0.5)
        satisfaction = int(max(1, min(5, round(sat))))

        temperature = 12 + 10 * yearly + random.gauss(0, 1.8)

        # Injecter un peu de manquants
        commentaire = random.choice(comments)
        if random.random() < 0.08:
            commentaire = ""
        if random.random() < 0.04:
            satisfaction_str = ""
        else:
            satisfaction_str = str(satisfaction)

        line = (
            f"{d.isoformat()},{region},{produit},{canal},{visites},"
            f"{ventes:.2f},{cout:.2f},{profit:.2f},{quantite},{remise:.3f},"
            f"{satisfaction_str},{temperature:.2f},\"{commentaire}\""
        )
        lines.append(line)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"OK: {out_path} ({n} lignes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
