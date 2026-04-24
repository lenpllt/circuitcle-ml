"""
generate_synthetic_logs.py — Génération de logs synthétiques pour CircuitClé
------------------------------------------------------------------------------
Le simulateur CircuitClé (main.py / Tkinter) produit des fichiers .txt dans
historiqueLHC/ et historiqueLHT/. Ce script génère des logs synthétiques qui
reproduisent fidèlement le format et le vocabulaire de ces vrais logs, afin
d'enrichir le dataset ML (build_dataset_ml.py) au-delà des 11 observations
initiales.

Scénarios dangereux basés sur les documents métier EDF/DIPDE :
  1. Embrochage sur SMALT fermée (REX Dampierre/St Laurent — conclusion finale du probleme.pdf)
  2. Clef E2 non prisonnière → embrochage LHC005JA → court-circuit triphasé à la terre
  3. Clef J7 non prisonnière → embrochage LHC010JA sur SMALT fermé → court-circuit
  4. Pont de barres débroché + sectionneur mise à la terre (palier 1300)

Objectif : 200 logs (160 normaux + 40 dangereux), répartis LHC/LHT.
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
# Cellules impliquées dans les scénarios dangereux documentés
PONT_BARRES_LHC   = "disjoncteur8 LHC008JA"   # pont de barres LHC
COUPLAGE_LHC      = "disjoncteur5 LHC005JA"   # disjoncteur de couplage
POINT_NEUTRE_LHC  = "disjoncteur7 LHC007JA"   # cellule point neutre
ARRIVEE_ALT_LHC   = "disjoncteur10 LHC010JA"  # arrivée alternateur

SMALT_LHC    = ["smalt3 LHC003JA", "smalt5 LHC005JA", "smalt9 LHC009JA", "smalt10 LHC010JA"]
SERRURES_LHC = ["LHC002CM", "LHC002CM2", "LHC003CM", "LHC009CM"]
PORTES_LHC   = ["porte3 LHC003JA", "porte5 LHC005JA"]

# ── Vocabulaire LHT ───────────────────────────────────────────────────────
ELEMENTS_LHT = [
    "disjoncteur1 LHT001JA", "disjoncteur2 LHT002JA", "disjoncteur3 LHT003JA",
    "contacteur1 LHT001CA", "contacteur2 LHT002CA", "contacteur3 LHT003CA",
    "transformateur1 LHT001TR", "source1 LHT001SR",
]
PONT_BARRES_LHT = "disjoncteur7 LHT007JP"   # pont de barres LHT
SMALT_LHT    = ["smalt1 LHT001JA", "smalt2 LHT002JA", "smalt3 LHT003JA"]
SERRURES_LHT = ["LHT001CM", "LHT002CM", "LHT003CM"]
ECLISSES_LHT = ["eclisse1 LHT001EC", "eclisse2 LHT002EC"]


# ════════════════════════════════════════════════════════════════════════════
# LOGS NORMAUX
# ════════════════════════════════════════════════════════════════════════════

def actions_lhc_normales(elements, smalt_list, serrures, portes,
                          avec_smalt=False, avec_porte=False):
    """Séquence LHC normale : débrochage avec SMALT ouvert avant toute manœuvre."""
    lignes = []
    sel = random.sample(elements, k=random.randint(2, 5))

    for el in sel:
        # Séquence correcte : ouverture SMALT → débrochage → verrouillage
        if avec_smalt and smalt_list and random.random() < 0.5:
            sm = random.choice(smalt_list)
            lignes.append(f"ouverture {sm}")
            lignes.append(f"Deverrouillage {sm} de la position ouvert")
        if random.random() < 0.6:
            lignes.append(f"manivelle inserree pour embroche {el}")
        lignes.append(f"débrochage {el}")
        lignes.append(f"manivelle extraite {el} en position debroche")
        lignes.append(f"Verrouillage position debroche, {el.split()[0]}, {el.split()[1]}")
        if random.random() < 0.4:
            lignes.append(f"Déverrouillage position debroche, {el.split()[0]}, {el.split()[1]}")
            lignes.append(f"embrochage {el}")
            lignes.append(f"débrochage {el}")
            lignes.append(f"Verrouillage position debroche, {el.split()[0]}, {el.split()[1]}")

    for ser in random.sample(serrures, k=random.randint(1, min(2, len(serrures)))):
        lignes.append(f"deverrouillage clefs serrure mere {ser}")

    if avec_smalt and smalt_list:
        sm = random.choice(smalt_list)
        # Ordre correct : débrochage AVANT fermeture SMALT
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
    """Séquence LHT normale."""
    lignes = []
    sel = random.sample(elements, k=random.randint(2, 4))

    for el in sel:
        lignes.append(f"débrochage {el}")
        lignes.append(f"Verrouillage position debroche, {el.split()[0]}, {el.split()[1]}")
        if random.random() < 0.4:
            lignes.append(f"insertion {el}")
            lignes.append(f"extraction {el}")

    for ser in random.sample(serrures, k=random.randint(1, min(2, len(serrures)))):
        lignes.append(f"deverrouillage clefs serrure mere {ser}")

    if avec_smalt and smalt_list:
        sm = random.choice(smalt_list)
        lignes.append(f"ouverture {sm}")
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


# ════════════════════════════════════════════════════════════════════════════
# SCÉNARIOS DANGEREUX — basés sur les documents EDF/DIPDE
# ════════════════════════════════════════════════════════════════════════════

def scenario_embrochage_sur_smalt_ferme():
    """
    REX Dampierre/St Laurent (conclusion finale du probleme.pdf, slide 16) :
    Cellule embrochée alors que le SMALT est encore en position fermée.
    Le dur mécanique peut être dépassé facilement.
    """
    el = random.choice([
        "contacteur9 LHC009JA", "contacteur1 LHC001JA",
        "disjoncteur3 LHC003JA", "contacteur4 LHC004JA"
    ])
    sm_nom = el.split()[1].replace("JA", "JS")
    lignes = [
        f"deverrouillage clefs serrure mere LHC003CM",
        f"Verrouillage position ferme, smalt {sm_nom}",  # SMALT encore fermé
        f"fermeture smalt {sm_nom}",
        f"manivelle inserree pour embroche {el}",
        f"embrochage {el}",   # embrochage avec SMALT fermé = DANGER
        f"DANGER détecté pour les personnes",
        f"Arrêt immédiat de la séquence",
    ]
    return lignes


def scenario_clef_e2_non_prisonniere():
    """
    Clef E2 non prisonnière après débrochage LHC008JA (pont de barres).
    Permet d'embrocher le disjoncteur de couplage LHC005JA → court-circuit triphasé.
    Source : conclusion finale du probleme.pdf, slide 6.
    """
    lignes = [
        f"manivelle inserree pour embroche {PONT_BARRES_LHC}",
        f"débrochage {PONT_BARRES_LHC}",
        f"manivelle extraite {PONT_BARRES_LHC} en position debroche",
        # Clef E2 non prisonnière : elle peut être récupérée
        f"deverrouillage clefs serrure mere LHC003CM",
        f"Déverrouillage position debroche, disjoncteur5, LHC005JA",
        # Embrochage du disjoncteur de couplage alors que le pont est débroché
        f"manivelle inserree pour embroche {COUPLAGE_LHC}",
        f"embrochage {COUPLAGE_LHC}",
        f"fermeture smalt5 LHC005JA",   # fermeture sur pont débroché
        f"DANGER détecté pour le materiel",
        f"Arrêt immédiat — court-circuit triphasé à la terre",
    ]
    return lignes


def scenario_clef_j7_non_prisonniere():
    """
    Clef J7 non prisonnière après débrochage cellule point neutre LHC007JA.
    Permet d'ouvrir le SMALT de LHC010JA, embrocher et fermer le disjoncteur
    avec le pont de barres débroché et SMALT fermé → court-circuit.
    Source : conclusion finale du probleme.pdf, slide 7.
    """
    lignes = [
        f"manivelle inserree pour embroche {POINT_NEUTRE_LHC}",
        f"débrochage {POINT_NEUTRE_LHC}",
        f"manivelle extraite {POINT_NEUTRE_LHC} en position debroche",
        # Clef J7 non prisonnière : récupérable
        f"deverrouillage clefs serrure mere LHC009CM",
        f"Deverrouillage smalt10 LHC010JA de la position ouvert",
        f"ouverture smalt10 LHC010JA",
        # Embrochage avec pont débroché et SMALT fermé
        f"fermeture smalt10 LHC010JA",
        f"manivelle inserree pour embroche {ARRIVEE_ALT_LHC}",
        f"embrochage {ARRIVEE_ALT_LHC}",
        f"DANGER détecté pour les personnes",
        f"Danger ! La condition de sécurité n'est pas respectée",
        f"Arrêt immédiat de la séquence",
    ]
    return lignes


def scenario_pont_barres_mise_a_la_terre():
    """
    Palier 1300 : disjoncteur LHC2005JA embroché et enclenché, débrochage
    du pont de barres LHC2007JP puis fermeture du sectionneur de mise à la
    terre → court-circuit du tableau LGF.
    Source : conclusion finale du probleme.pdf, palier 1300.
    """
    lignes = [
        f"manivelle inserree pour embroche {COUPLAGE_LHC}",
        f"embrochage {COUPLAGE_LHC}",
        f"Verrouillage position embroche, disjoncteur5, LHC005JA",
        # Débrochage du pont de barres avec le couplage embroché = DANGER
        f"manivelle inserree pour embroche {PONT_BARRES_LHC}",
        f"débrochage {PONT_BARRES_LHC}",
        f"manivelle extraite {PONT_BARRES_LHC} en position debroche",
        f"deverrouillage clefs serrure mere LHC009CM",
        # Fermeture sectionneur mise à la terre
        f"Verrouillage position debroche, disjoncteur8, LHC008JA",
        f"fermeture smalt8 LHC008JA",  # sectionneur mise à la terre sur barres sous tension
        f"DANGER détecté pour le materiel",
        f"Arrêt immédiat — mise en court-circuit tableau LGF",
    ]
    return lignes


def scenario_lht_embrochage_smalt_ferme():
    """
    LHT : embrochage d'un disjoncteur avec le SMALT en position fermée.
    Analogue au REX LHC mais sur tableau LHT.
    """
    el = random.choice(["disjoncteur1 LHT001JA", "disjoncteur2 LHT002JA"])
    sm = el.split()[1].replace("JA", "JS")
    lignes = [
        f"deverrouillage clefs serrure mere LHT001CM",
        f"fermeture smalt {sm}",
        f"Verrouillage position ferme, smalt, {sm}",
        f"manivelle inserree pour embroche {el}",
        f"embrochage {el}",
        f"DANGER détecté pour les personnes",
        f"Arrêt immédiat de la séquence",
    ]
    return lignes


def scenario_lht_pont_debroche_transfo_actif():
    """
    LHT : débrochage pont de barres avec transformateur encore alimenté.
    """
    lignes = [
        "vérification transformateur1 LHT001TR",
        "Verrouillage transformateur1 LHT001TR",
        f"manivelle inserree pour embroche {PONT_BARRES_LHT}",
        f"débrochage {PONT_BARRES_LHT}",
        f"deverrouillage clefs serrure mere LHT002CM",
        f"insertion source1 LHT001SR",  # source active avec pont débroché
        f"fermeture smalt1 LHT001JA",
        f"DANGER détecté pour le materiel",
        f"Arrêt immédiat de la séquence",
    ]
    return lignes


# ── Sélection aléatoire du scénario dangereux ─────────────────────────────

SCENARIOS_DANGEREUX_LHC = [
    scenario_embrochage_sur_smalt_ferme,
    scenario_clef_e2_non_prisonniere,
    scenario_clef_j7_non_prisonniere,
    scenario_pont_barres_mise_a_la_terre,
]

SCENARIOS_DANGEREUX_LHT = [
    scenario_lht_embrochage_smalt_ferme,
    scenario_lht_pont_debroche_transfo_actif,
]


# ════════════════════════════════════════════════════════════════════════════
# FIN DE LOG
# ════════════════════════════════════════════════════════════════════════════

def fin_normale(nb_cles_restantes=None):
    if nb_cles_restantes is None:
        nb_cles_restantes = random.randint(0, 5)
    return [
        f"Clés non utilisées : {nb_cles_restantes}",
        "Fin de l'execution des etapes",
    ]


# ── Horodatage ────────────────────────────────────────────────────────────

_base_date = datetime(2025, 4, 1, 8, 0, 0)
_delta_minutes = 0

def _next_ts():
    global _delta_minutes
    _delta_minutes += random.randint(30, 240)
    return _base_date + timedelta(minutes=_delta_minutes)

def nom_fichier_lhc(palier):
    return f"LHC_{palier}_preVD_{_next_ts().strftime('%d-%m-%Y_%H-%M-%S')}.txt"

def nom_fichier_lht(palier):
    return f"LHT_{palier}_{_next_ts().strftime('%d-%m-%Y_%H-%M-%S')}.txt"


# ════════════════════════════════════════════════════════════════════════════
# GÉNÉRATION
# ════════════════════════════════════════════════════════════════════════════

def generer_log_lhc(palier, dangereux=False):
    if dangereux:
        scenario = random.choice(SCENARIOS_DANGEREUX_LHC)
        lignes = scenario()
    else:
        avec_smalt = random.random() < 0.5
        avec_porte = random.random() < 0.3
        lignes = actions_lhc_normales(
            ELEMENTS_LHC, SMALT_LHC, SERRURES_LHC, PORTES_LHC,
            avec_smalt=avec_smalt, avec_porte=avec_porte
        )
        random.shuffle(lignes)
        lignes += fin_normale()

    nom = nom_fichier_lhc(palier)
    with open(os.path.join(DOSSIER_LHC, nom), "w", encoding="utf-8") as f:
        f.write("\n".join(lignes))
    return nom


def generer_log_lht(palier, dangereux=False):
    if dangereux:
        scenario = random.choice(SCENARIOS_DANGEREUX_LHT)
        lignes = scenario()
    else:
        lignes = actions_lht_normales(
            ELEMENTS_LHT, SMALT_LHT, SERRURES_LHT, ECLISSES_LHT,
            avec_smalt=random.random() < 0.4,
            avec_eclisse=random.random() < 0.3,
            avec_transfo=random.random() < 0.25,
        )
        random.shuffle(lignes)
        lignes += fin_normale()

    nom = nom_fichier_lht(palier)
    with open(os.path.join(DOSSIER_LHT, nom), "w", encoding="utf-8") as f:
        f.write("\n".join(lignes))
    return nom


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    # Suppression des anciens logs synthétiques (conserver les vrais)
    for dossier in [DOSSIER_LHC, DOSSIER_LHT]:
        for f in os.listdir(dossier):
            # Les vrais logs ont un format de date différent (ex: 07-05-2025)
            # Les synthétiques commencent tous à partir du 01-04-2025
            chemin = os.path.join(dossier, f)
            if f.endswith(".txt") and ("01-04-2025" in f or "02-04-2025" in f
                    or "03-04-2025" in f or "04-04-2025" in f
                    or "05-04-2025" in f or "06-04-2025" in f
                    or "07-04-2025" in f or "08-04-2025" in f
                    or "09-04-2025" in f or "10-04-2025" in f
                    or "11-04-2025" in f or "12-04-2025" in f
                    or "13-04-2025" in f or "14-04-2025" in f
                    or "15-04-2025" in f or "16-04-2025" in f):
                os.remove(chemin)

    paliers = [900, 1300, 1400]
    compteurs = {"lhc_normal": 0, "lhc_danger": 0, "lht_normal": 0, "lht_danger": 0}

    print("Génération des logs LHC...")
    for _ in range(120):  # 120 normaux LHC
        generer_log_lhc(random.choice(paliers), dangereux=False)
        compteurs["lhc_normal"] += 1
    for _ in range(30):   # 30 dangereux LHC (4 scénarios réels)
        generer_log_lhc(random.choice(paliers), dangereux=True)
        compteurs["lhc_danger"] += 1

    print("Génération des logs LHT...")
    for _ in range(40):   # 40 normaux LHT
        generer_log_lht(random.choice(paliers), dangereux=False)
        compteurs["lht_normal"] += 1
    for _ in range(10):   # 10 dangereux LHT (2 scénarios réels)
        generer_log_lht(random.choice(paliers), dangereux=True)
        compteurs["lht_danger"] += 1

    total = sum(compteurs.values())
    total_danger = compteurs["lhc_danger"] + compteurs["lht_danger"]
    total_normal = compteurs["lhc_normal"] + compteurs["lht_normal"]

    print(f"\n{'='*55}")
    print(f"Logs générés          : {total}")
    print(f"  LHC normaux         : {compteurs['lhc_normal']}")
    print(f"  LHC dangereux       : {compteurs['lhc_danger']}  (4 scénarios métier)")
    print(f"  LHT normaux         : {compteurs['lht_normal']}")
    print(f"  LHT dangereux       : {compteurs['lht_danger']}  (2 scénarios métier)")
    print(f"  Total normaux       : {total_normal}  ({total_normal/total*100:.0f} %)")
    print(f"  Total dangereux     : {total_danger}   ({total_danger/total*100:.0f} %)")
    print(f"\nScénarios dangereux documentés :")
    print(f"  - Embrochage sur SMALT fermée (REX Dampierre/St Laurent)")
    print(f"  - Clef E2 non prisonnière → court-circuit triphasé LHC005JA")
    print(f"  - Clef J7 non prisonnière → court-circuit LHC010JA")
    print(f"  - Pont de barres débroché + mise à la terre (palier 1300)")
    print(f"  - Embrochage sur SMALT fermée LHT")
    print(f"  - Pont débroché + transformateur actif LHT")
    print(f"\nLancer ensuite : python build_dataset_ml.py")
    print(f"puis           : python train_modele_ml.py")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
