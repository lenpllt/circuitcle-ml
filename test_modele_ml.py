import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


FICHIER_MODELE = "meilleur_modele_ml.joblib"
FICHIER_TEST = "jeu_test_ml.csv"


def main():
    artefacts = joblib.load(FICHIER_MODELE)
    pipeline = artefacts["model"]
    meilleur_nom = artefacts["best_model_name"]
    meilleur_score = artefacts["best_accuracy"]
    feature_columns = artefacts["feature_columns"]

    df_test = pd.read_csv(FICHIER_TEST)

    X_test = df_test[feature_columns]
    y_test = df_test["label_danger"]

    y_pred = pipeline.predict(X_test)

    print("===== TEST DU MEILLEUR MODELE =====")
    print("Modèle :", meilleur_nom)
    print("Accuracy sauvegardée :", round(meilleur_score, 4))
    print("Accuracy recalculée :", round(accuracy_score(y_test, y_pred), 4))

    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report :")
    print(classification_report(y_test, y_pred, zero_division=0))

    resultat = df_test.copy()
    resultat["prediction"] = y_pred
    resultat["correct"] = (resultat["label_danger"] == resultat["prediction"]).astype(int)
    resultat.to_csv("predictions_modele_ml.csv", index=False, encoding="utf-8")

    print("\nFichier généré : predictions_modele_ml.csv")


if __name__ == "__main__":
    main()