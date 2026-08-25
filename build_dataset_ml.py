import os
import re
import pandas as pd


DOSSIERS_HISTORIQUES = ["historiqueLHC", "historiqueLHT"]
FICHIER_SORTIE = "dataset_logs_ml.csv"


# ============================================================
# LECTURE
# ============================================================

def lire_fichier_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ============================================================
# UTILITAIRES
# ============================================================

def extraire_palier_depuis_nom(filename: str) -> int:
    match = re.search(r'_(900|1300|1400)_', filename)
    return int(match.group(1)) if match else 0


def compter_occurrences(texte: str, mot: str) -> int:
    return len(re.findall(rf"\b{re.escape(mot)}\b", texte, flags=re.IGNORECASE))


def extraire_nb_cles_non_utilisees(texte: str) -> int:
    """
    Extrait le nombre de clés non utilisées depuis la ligne de fin de log.
    Exemple : "Clés non utilisées : 4"  ->  4
    Si la ligne est absente (log interrompu par danger), retourne -1.
    """
    match = re.search(r"cl[eé]s?\s+non\s+utilis[eé]es?\s*:\s*(\d+)", texte, flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


def ratio_safe(numerateur: int, denominateur: int) -> float:
    """Retourne le ratio ou 0.0 si le dénominateur est nul."""
    return round(numerateur / denominateur, 4) if denominateur > 0 else 0.0


# ============================================================
# CONSTRUCTION DES FEATURES
# ============================================================

def construire_features(texte: str, nom_fichier: str, dossier: str) -> dict:
    texte_min = texte.lower()
    lignes = [ligne.strip() for ligne in texte.splitlines() if ligne.strip()]

    # ----------------------------------------------------------
    # LABEL — basé uniquement sur les messages de danger métier
    # émis par le simulateur (LHC/LHT_Classe_conditions_fin.py).
    # On exclut "erreur" et "exception" Python qui sont des bugs
    # techniques sans lien avec la sécurité électrique.
    # ----------------------------------------------------------
    label = 1 if (
        "danger détecté" in texte_min
        or "danger detecte" in texte_min
        or "arrêt immédiat" in texte_min
        or "arret immediat" in texte_min
        or "danger ! la condition" in texte_min
    ) else 0

    # ----------------------------------------------------------
    # FEATURES DE BASE — comptages d'actions
    # ----------------------------------------------------------
    nb_ouverture      = compter_occurrences(texte_min, "ouverture")
    nb_fermeture      = compter_occurrences(texte_min, "fermeture")
    nb_embrochage     = compter_occurrences(texte_min, "embrochage")
    nb_debrochage     = (compter_occurrences(texte_min, "débrochage")
                         + compter_occurrences(texte_min, "debrochage"))
    nb_verrouillage   = compter_occurrences(texte_min, "verrouillage")
    nb_deverrouillage = (compter_occurrences(texte_min, "déverrouillage")
                         + compter_occurrences(texte_min, "deverrouillage"))
    nb_insertion  = compter_occurrences(texte_min, "insertion")
    nb_extraction = compter_occurrences(texte_min, "extraction")

    # ----------------------------------------------------------
    # FEATURES DE PRESENCE — éléments impliqués dans la séquence
    # ----------------------------------------------------------
    presence_smalt          = 1 if "smalt"          in texte_min else 0
    presence_porte          = 1 if "porte"          in texte_min else 0
    presence_coffret        = 1 if "coffret"        in texte_min else 0
    presence_transformateur = 1 if "transformateur" in texte_min else 0
    presence_source         = 1 if "source"         in texte_min else 0
    presence_eclisse        = 1 if ("eclisse" in texte_min or "éclisse" in texte_min) else 0

    # ----------------------------------------------------------
    # FEATURES TECHNIQUES — bugs Python (gardées comme features
    # mais retirées du critère de labelling)
    # ----------------------------------------------------------
    presence_erreur    = 1 if "erreur"    in texte_min else 0
    presence_exception = 1 if "exception" in texte_min else 0

    # ----------------------------------------------------------
    # NOUVELLES FEATURES METIER — directement liées au danger
    # ----------------------------------------------------------
    danger_personne = 1 if "danger détecté pour les personnes" in texte_min else 0
    danger_materiel = 1 if "danger détecté pour le materiel"   in texte_min else 0

    nb_arret_immediat = (compter_occurrences(texte_min, "arrêt immédiat")
                         + compter_occurrences(texte_min, "arret immediat"))

    nb_cles_non_utilisees = extraire_nb_cles_non_utilisees(texte)

    ratio_verr_deverr = ratio_safe(nb_verrouillage, nb_deverrouillage)
    ratio_emb_deb     = ratio_safe(nb_embrochage, nb_debrochage)
    ratio_ouv_fer     = ratio_safe(nb_ouverture, nb_fermeture)

    # ----------------------------------------------------------
    # ASSEMBLAGE
    # ----------------------------------------------------------
    return {
        "nom_fichier":              nom_fichier,
        "tableau_type":             "LHC" if "lhc" in dossier.lower() else "LHT",
        "palier":                   extraire_palier_depuis_nom(nom_fichier),
        "nb_lignes":                len(lignes),
        "nb_ouverture":             nb_ouverture,
        "nb_fermeture":             nb_fermeture,
        "nb_embrochage":            nb_embrochage,
        "nb_debrochage":            nb_debrochage,
        "nb_verrouillage":          nb_verrouillage,
        "nb_deverrouillage":        nb_deverrouillage,
        "nb_insertion":             nb_insertion,
        "nb_extraction":            nb_extraction,
        "presence_smalt":           presence_smalt,
        "presence_porte":           presence_porte,
        "presence_coffret":         presence_coffret,
        "presence_transformateur":  presence_transformateur,
        "presence_source":          presence_source,
        "presence_eclisse":         presence_eclisse,
        "presence_erreur":          presence_erreur,
        "presence_exception":       presence_exception,
        "danger_personne":          danger_personne,
        "danger_materiel":          danger_materiel,
        "nb_arret_immediat":        nb_arret_immediat,
        "nb_cles_non_utilisees":    nb_cles_non_utilisees,
        "ratio_verr_deverr":        ratio_verr_deverr,
        "ratio_emb_deb":            ratio_emb_deb,
        "ratio_ouv_fer":            ratio_ouv_fer,
        "label_danger":             label,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    lignes_dataset = []

    for dossier in DOSSIERS_HISTORIQUES:
        if not os.path.isdir(dossier):
            print(f"Dossier introuvable : {dossier}")
            continue

        for nom_fichier in sorted(os.listdir(dossier)):
            if nom_fichier.lower().endswith(".txt"):
                chemin = os.path.join(dossier, nom_fichier)
                contenu = lire_fichier_txt(chemin)
                features = construire_features(contenu, nom_fichier, dossier)
                lignes_dataset.append(features)
                print(f"  [OK] {nom_fichier}  ->  label={features['label_danger']}")

    df = pd.DataFrame(lignes_dataset)

    if df.empty:
        print("Aucune donnée trouvée pour construire le dataset.")
        return

    df["palier"] = df["palier"].fillna(0).astype(int)
    df["nb_cles_non_utilisees"] = df["nb_cles_non_utilisees"].fillna(-1).astype(int)

    df.to_csv(FICHIER_SORTIE, index=False, encoding="utf-8")

    print(f"\n{'='*50}")
    print(f"Dataset créé       : {FICHIER_SORTIE}")
    print(f"Nombre d'exemples  : {len(df)}")
    print(f"Nombre de features : {len(df.columns) - 2}")
    print(f"\nRépartition de la cible :")
    print(df["label_danger"].value_counts(dropna=False).to_string())
    print(f"\nAperçu des nouvelles features métier :")
    cols_metier = ["nom_fichier", "danger_personne", "danger_materiel",
                   "nb_arret_immediat", "nb_cles_non_utilisees",
                   "ratio_verr_deverr", "ratio_emb_deb", "label_danger"]
    print(df[cols_metier].to_string(index=False))
    print(f"{'='*50}")


if __name__ == "__main__":
    main()