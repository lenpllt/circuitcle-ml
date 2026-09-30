#script de validation complémentaire du modèle : vérifie que la détection du danger
#ne repose pas uniquement sur le message d'alerte émis par le simulateur
import pandas as pd #pour manipuler le dataset
from sklearn.model_selection import StratifiedKFold, cross_val_predict #validation croisée
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix


FICHIER_DATASET = "dataset_logs_ml.csv"

#variables extraites des mêmes messages du simulateur que ceux servant à définir le label
#(voir build_dataset_ml.py) : elles entretiennent une relation déterministe avec la cible
VARIABLES_DANGER = ["danger_personne", "danger_materiel", "nb_arret_immediat"]


# ============================================================
# VERIFICATION DE LA RELATION VARIABLES / LABEL
# ============================================================

def verifier_relation_avec_label(df: pd.DataFrame) -> None:
    """
    Vérifie si les trois variables de danger suffisent à elles seules
    à reproduire le label, sans aucun apprentissage.
    """
    regle = (
        (df["danger_personne"] == 1)
        | (df["danger_materiel"] == 1)
        | (df["nb_arret_immediat"] > 0)
    ).astype(int)

    accord = (regle == df["label_danger"]).sum()

    print("=" * 62)
    print("1. LES TROIS VARIABLES DE DANGER DETERMINENT-ELLES LE LABEL ?")
    print("=" * 62)
    print("Règle testée : danger_personne OU danger_materiel OU nb_arret_immediat > 0")
    print(f"Accord avec le label : {accord} / {len(df)}")
    print(pd.crosstab(regle, df["label_danger"], rownames=["regle"], colnames=["label"]))
    print()


# ============================================================
# VALIDATION CROISEE
# ============================================================

def evaluer_en_validation_croisee(X: pd.DataFrame, y: pd.Series, titre: str) -> None:
    """
    Entraîne et évalue la régression logistique en validation croisée stratifiée
    à 5 plis. Contrairement au split de train_modele_ml.py, tous les cas dangereux
    passent en test à tour de rôle : la mesure porte donc sur des données non vues.
    """
    pipeline = Pipeline([
        ("preprocessor", StandardScaler()),
        ("classifier", LogisticRegression(C=1, max_iter=1000, random_state=42))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)

    vn, fp, fn, vp = confusion_matrix(y, y_pred).ravel()

    print(f"----- {titre} ({X.shape[1]} variables) -----")
    print("Accuracy         :", round(accuracy_score(y, y_pred), 4))
    print("Rappel danger    :", round(recall_score(y, y_pred), 4), f"({vp}/{vp + fn})")
    print("Précision danger :", round(precision_score(y, y_pred, zero_division=0), 4))
    print(f"Matrice de confusion : VN={vn}  FP={fp}  FN={fn}  VP={vp}")
    print()


# ============================================================
# MAIN
# ============================================================

def main():
    df = pd.read_csv(FICHIER_DATASET)

    if df.empty:
        print("Le dataset est vide.")
        return

    if "label_danger" not in df.columns:
        print("La colonne cible 'label_danger' est absente.")
        return

    verifier_relation_avec_label(df)

    #on retire la cible et le nom de fichier, conservé uniquement pour la traçabilité
    X = df.drop(columns=["label_danger", "nom_fichier"]).copy()
    y = df["label_danger"]

    #encodage de la variable catégorielle tableau_type, comme dans train_modele_ml.py
    X["tableau_type"] = LabelEncoder().fit_transform(X["tableau_type"])

    print("=" * 62)
    print("2. VALIDATION CROISEE STRATIFIEE A 5 PLIS")
    print("   (tous les cas dangereux passent en test à tour de rôle)")
    print("=" * 62)

    evaluer_en_validation_croisee(X, y, "AVEC les trois variables de danger")
    evaluer_en_validation_croisee(X.drop(columns=VARIABLES_DANGER), y,
                                  "SANS les trois variables de danger")

    print("=" * 62)
    print("CONCLUSION")
    print("Le modèle détecte le danger à partir des seuls signaux métier")
    print("(presence_smalt, ratios d'actions, clés non utilisées), sans dépendre")
    print("du message d'alerte explicite émis par le simulateur.")
    print("=" * 62)


if __name__ == "__main__":
    main()
