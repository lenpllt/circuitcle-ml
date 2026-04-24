"""
generate_synthetic_logs.py — Génération de logs synthétiques pour CircuitClé
------------------------------------------------------------------------------
Le simulateur CircuitClé (main.py / Tkinter) produit des fichiers .txt dans
historiqueLHC/ et historiqueLHT/. Ce script génère des logs synthétiques qui
reproduisent fidèlement le format et le vocabulaire de ces vrais logs, afin
d'enrichir le dataset ML (build_dataset_ml.py) au-delà des 11 observations
initiales.

Objectif : 160 logs (130 normaux + 30 dangereux), répartis LHC/LHT.
"""

import os
import random
from datetime import datetime, timedelta

random.seed(42)

# ── Répertoires de sortie ─────────────────────────────────────────────────
DOSSIER_LHC = "historiqueLHC"
DOSSIER_LHT = "historiqueLHT"
os.makedirs(DOSSIER_LHC, exist_ok=True)
os.makedirs(DOSSIER_LHT, exist_ok=True)

# ── Vocabulaire LHC ───────────────────────────────────────────────────────
ELEMENTS_LHC = [
    "disjoncteur3 LHC003JA", "disjoncteur5 LHC005JA", "disjoncteur10 LHC010JA",
    "disjoncteur15 LHC015JA", "disjoncteur16 LHC016JA", "disjoncteur2 LHC002JA",
    "contacteur1 LHC001JA", "contacteur4 LHC004JA", "contacteur9 LHC009JA",
    "contacteur12 LHC012JA", "contacteur13 LHC013JA", "contacteur14 LHC014JA",
    "contacteurLG LG", "disjoncteurBC BC",
]
SMALT_LHC   = ["smalt3 LHC003JA", "smalt5 LHC005JA", "smalt10 LHC010JA"]
SERRURES_LHC = ["LHC002CM", "LHC002CM2", "LHC003CM"]
PORTES_LHC   = ["porte3 LHC003JA", "porte5 LHC005JA"]

# ── Vocabulaire LHT ───────────────────────────────────────────────────────
ELEMENTS_LHT = [
    "disjoncteur1 LHT001JA", "disjoncteur2 LHT002JA", "disjoncteur3 LHT003JA",
    "contacteur1 LHT001CA", "contacteur2 LHT002CA", "contacteur3 LHT003CA",
    "transformateur1 LHT001TR", "source1 LHT001SR",
]
SMALT_LHT    = ["smalt1 LHT001JA", "smalt2 LHT002JA"]
SERRURES_LHT = ["LHT001CM", "LHT002CM"]
ECLISSES_LHT = ["eclisse1 LHT001EC", "eclisse2 LHT002EC"]

# ── Générateurs d'actions ─────────────────────────────────────────────────

def actions_lhc_normales(elements, smalt_list, serrures, portes, avec_smalt=False, avec_porte=False):
    """Retourne une liste de lignes d'action LHC pour un log normal."""
    lignes = []
    sel = random.sample(elements, k=random.randint(2, 5))

    for el in sel:
        if random.random() < 0.6:
            lignes.append(f"manivelle inserree pour embroche {el}")
        lignes.append(f"débrochage {el}")
        lignes.append(f"manivelle extraite {el} en position debroche")
        if random.random() < 0.5:
            lignes.append(f"Verrouillage position debroche, {el.split()[0]}, {el.split()[1]}")
        if random.random() < 0.4:
            lignes.append(f"Déverrouillage position debroche, {el.split()[0]}, {el.split()[1]}")
        if random.random() < 0.3:
            lignes.append(f"embrochage {el}")
            lignes.append(f"débrochage {el}")

    for ser in random.sample(serrures, k=random.randint(1, len(serrures))):
        lignes.append(f"deverrouillage clefs serrure mere {ser}")

    if avec_smalt and smalt_list:
        sm = random.choice(smalt_list)
        lignes.append(f"Deverrouillage {sm} de la position ouvert")
        lignes.append(f"fermeture {sm}")
        if random.random() < 0.5:
            lignes.append(f"ouverture {sm}")
        lignes.append(f"Verrouillage position ferme, {sm.split()[0]}, {sm.split()[1]}")

    if avec_porte and portes:
        pt = random.choice(portes)
        lignes.append(f"Ouverture {pt}")
        lignes.append(f"Fermeture {pt}")

    return lignes


def actions_lht_normales(elements, smalt_list, serrures, eclisses,
                         avec_smalt=False, avec_eclisse=False, avec_transfo=False):
    """Retourne une liste de lignes d'action LHT pour un log normal."""
    lignes = []
    sel = random.sample(elements, k=random.randint(2, 4))

    for el in sel:
        lignes.append(f"débrochage {el}")
        if random.random() < 0.5:
            lignes.append(f"Verrouillage position debroche, {el.split()[0]}, {el.split()[1]}")
        if random.random() < 0.4:
            lignes.append(f"insertion {el}")
            lignes.append(f"extraction {el}")

    for ser in random.sample(serrures, k=random.randint(1, len(serrures))):
        lignes.append(f"deverrouillage clefs serrure mere {ser}")

    if avec_smalt and smalt_list:
        sm = random.choice(smalt_list)
        lignes.append(f"Deverrouillage {sm} de la position ouvert")
        lignes.append(f"fermeture {sm}")

    if avec_eclisse and eclisses:
        ec = random.choice(eclisses)
        lignes.append(f"mise en place éclisse {ec}")
        lignes.append(f"retrait éclisse {ec}")

    if avec_transfo:
        lignes.append("vérification transformateur1 LHT001TR")
        lignes.append("Verrouillage transformateur1 LHT001TR")

    return lignes


# ── Fin de log ────────────────────────────────────────────────────────────

def fin_normale(nb_cles_restantes=None):
    if nb_cles_restantes is None:
        nb_cles_restantes = random.randint(0, 5)
    return [f"Clés non utilisées : {nb_cles_restantes}",
            "Fin de l'execution des etapes"]


def fin_danger(type_danger="personnes"):
    phrases = []
    if type_danger == "personnes":
        phrases.append("DANGER détecté pour les personnes")
        phrases.append("Arrêt immédiat de la séquence")
        if random.random() < 0.5:
            phrases.append("Danger ! La condition de sécurité n'est pas respectée")
    else:
        phrases.append("DANGER détecté pour le matériel")
        phrases.append("Arrêt immédiat — risque matériel")
    return phrases


# ── Nom de fichier horodaté ───────────────────────────────────────────────

_base_date = datetime(2025, 4, 1, 8, 0, 0)
_delta_minutes = 0

def nom_fichier_lhc(palier):
    global _delta_minutes
    _delta_minutes += random.randint(30, 240)
    ts = _base_date + timedelta(minutes=_delta_minutes)
    return f"LHC_{palier}_preVD_{ts.strftime('%d-%m-%Y_%H-%M-%S')}.txt"

def nom_fichier_lht(palier):
    global _delta_minutes
    _delta_minutes += random.randint(30, 240)
    ts = _base_date + timedelta(minutes=_delta_minutes)
    return f"LHT_{palier}_{ts.strftime('%d-%m-%Y_%H-%M-%S')}.txt"


# ── Génération d'un log LHC ───────────────────────────────────────────────

def generer_log_lhc(palier, dangereux=False):
    avec_smalt  = random.random() < 0.5
    avec_porte  = random.random() < 0.3

    lignes = actions_lhc_normales(
        ELEMENTS_LHC, SMALT_LHC, SERRURES_LHC, PORTES_LHC,
        avec_smalt=avec_smalt, avec_porte=avec_porte
    )
    random.shuffle(lignes)

    if dangereux:
        # Injection d'actions supplémentaires typiques des séquences dangereuses
        el = random.choice(ELEMENTS_LHC)
        lignes += [
            f"manivelle inserree pour embroche {el}",
            f"débrochage {el}",
            f"embrochage {el}",  # double action = risque
        ]
        lignes += fin_danger(random.choice(["personnes", "materiel"]))
    else:
        lignes += fin_normale()

    nom = nom_fichier_lhc(palier)
    chemin = os.path.join(DOSSIER_LHC, nom)
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes))
    return nom


# ── Génération d'un log LHT ───────────────────────────────────────────────

def generer_log_lht(palier, dangereux=False):
    avec_smalt   = random.random() < 0.4
    avec_eclisse = random.random() < 0.3
    avec_transfo = random.random() < 0.25

    lignes = actions_lht_normales(
        ELEMENTS_LHT, SMALT_LHT, SERRURES_LHT, ECLISSES_LHT,
        avec_smalt=avec_smalt,
        avec_eclisse=avec_eclisse,
        avec_transfo=avec_transfo
    )
    random.shuffle(lignes)

    if dangereux:
        el = random.choice(ELEMENTS_LHT)
        lignes += [
            f"débrochage {el}",
            f"insertion {el}",
            f"extraction {el}",
        ]
        lignes += fin_danger(random.choice(["personnes", "materiel"]))
    else:
        lignes += fin_normale()

    nom = nom_fichier_lht(palier)
    chemin = os.path.join(DOSSIER_LHT, nom)
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes))
    return nom


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    paliers = [900, 1300, 1400]
    compteurs = {"lhc_normal": 0, "lhc_danger": 0, "lht_normal": 0, "lht_danger": 0}

    print("Génération des logs synthétiques LHC...")

    # LHC : 95 normaux + 20 dangereux
    for _ in range(95):
        palier = random.choice(paliers)
        generer_log_lhc(palier, dangereux=False)
        compteurs["lhc_normal"] += 1

    for _ in range(20):
        palier = random.choice(paliers)
        generer_log_lhc(palier, dangereux=True)
        compteurs["lhc_danger"] += 1

    print("Génération des logs synthétiques LHT...")

    # LHT : 35 normaux + 10 dangereux
    for _ in range(35):
        palier = random.choice(paliers)
        generer_log_lht(palier, dangereux=False)
        compteurs["lht_normal"] += 1

    for _ in range(10):
        palier = random.choice(paliers)
        generer_log_lht(palier, dangereux=True)
        compteurs["lht_danger"] += 1

    total = sum(compteurs.values())
    total_danger = compteurs["lhc_danger"] + compteurs["lht_danger"]
    total_normal = compteurs["lhc_normal"] + compteurs["lht_normal"]

    print(f"\n{'='*50}")
    print(f"Logs générés       : {total}")
    print(f"  LHC normaux      : {compteurs['lhc_normal']}")
    print(f"  LHC dangereux    : {compteurs['lhc_danger']}")
    print(f"  LHT normaux      : {compteurs['lht_normal']}")
    print(f"  LHT dangereux    : {compteurs['lht_danger']}")
    print(f"  Total normaux    : {total_normal}  ({total_normal/total*100:.0f} %)")
    print(f"  Total dangereux  : {total_danger}  ({total_danger/total*100:.0f} %)")
    print(f"\nLancer ensuite : python build_dataset_ml.py")
    print(f"puis           : python train_modele_ml.py")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
