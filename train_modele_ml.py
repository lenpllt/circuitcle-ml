import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


FICHIER_DATASET = "dataset_logs_ml.csv"
FICHIER_MODELE = "meilleur_modele_ml.joblib"
FICHIER_TEST = "jeu_test_ml.csv"


def main():
    df = pd.read_csv(FICHIER_DATASET)

    if df.empty:
        print("Le dataset est vide.")
        return

    if "label_danger" not in df.columns:
        print("La colonne cible 'label_danger' est absente.")
        return

    # On garde le nom de fichier uniquement pour traçabilité, pas pour l'entraînement
    X = df.drop(columns=["label_danger", "nom_fichier"])
    y = df["label_danger"]

    # Encodage de la variable catégorielle tableau_type
    X = X.copy()
    le = LabelEncoder()
    X["tableau_type"] = le.fit_transform(X["tableau_type"])

    colonnes_numeriques = list(X.columns)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), colonnes_numeriques)
        ]
    )

    # Garantir que tous les exemples dangereux sont dans le jeu d'entraînement.
    # Avec très peu de données et une classe rare, le split aléatoire risque de placer
    # le seul exemple dangereux dans le jeu de test, rendant le modèle aveugle au danger.
    X_danger = X[y == 1]
    y_danger = y[y == 1]
    X_normal = X[y == 0]
    y_normal = y[y == 0]

    X_train_normal, X_test_normal, y_train_normal, y_test_normal = train_test_split(
        X_normal, y_normal,
        test_size=0.30,
        random_state=42
    )

    X_train = pd.concat([X_train_normal, X_danger]).reset_index(drop=True)
    y_train = pd.concat([y_train_normal, y_danger]).reset_index(drop=True)
    X_test = X_test_normal.reset_index(drop=True)
    y_test = y_test_normal.reset_index(drop=True)

    print("Taille totale du dataset :", len(df))
    print(f"  dont dangereux : {len(X_danger)} | normaux : {len(X_normal)}")
    print("Taille jeu d'entraînement :", len(X_train), f"(dont {int(y_train.sum())} dangereux)")
    print("Taille jeu de test :", len(X_test), f"(dont {int(y_test.sum())} dangereux)")
    print("Part d'entraînement :", round(len(X_train) / len(df) * 100, 2), "%")

    modeles = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        # Réseau de neurones (Deep Learning – MLP)
        "MLP (Deep Learning)": MLPClassifier(
            hidden_layer_sizes=(32, 16),  # 2 couches cachées : 32 puis 16 neurones
            activation="relu",
            solver="adam",
            max_iter=2000,
            random_state=42
            # early_stopping désactivé : dataset trop petit pour une validation interne
        )
    }

    meilleur_nom = None
    meilleur_score = -1
    meilleur_pipeline = None

    for nom, modele in modeles.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", modele)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        score = accuracy_score(y_test, y_pred)

        print(f"\n===== {nom} =====")
        print("Accuracy :", round(score, 4))
        print("Matrice de confusion :")
        print(confusion_matrix(y_test, y_pred))
        print("Classification report :")
        print(classification_report(y_test, y_pred, zero_division=0))

        if score > meilleur_score:
            meilleur_score = score
            meilleur_nom = nom
            meilleur_pipeline = pipeline

    # --- Optimisation par GridSearchCV (critère C3.5) ---
    print("\n===== Optimisation GridSearchCV — Logistic Regression =====")
    param_grid = {"classifier__C": [0.01, 0.1, 1, 10, 100]}
    pipeline_lr_gs = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    grid_search = GridSearchCV(pipeline_lr_gs, param_grid, cv=3, scoring="accuracy")
    grid_search.fit(X_train, y_train)
    print("Meilleurs paramètres :", grid_search.best_params_)
    print("Meilleure accuracy (CV) :", round(grid_search.best_score_, 4))
    y_pred_gs = grid_search.best_estimator_.predict(X_test)
    print("Accuracy jeu de test :", round(accuracy_score(y_test, y_pred_gs), 4))
    if accuracy_score(y_test, y_pred_gs) >= meilleur_score:
        meilleur_pipeline = grid_search.best_estimator_
        meilleur_nom = f"LogisticRegression (C={grid_search.best_params_['classifier__C']})"

    joblib.dump(
        {
            "model": meilleur_pipeline,
            "label_encoder_tableau_type": le,
            "feature_columns": list(X.columns),
            "best_model_name": meilleur_nom,
            "best_accuracy": meilleur_score
        },
        FICHIER_MODELE
    )

    df_test = X_test.copy()
    df_test["label_danger"] = y_test.values
    df_test.to_csv(FICHIER_TEST, index=False, encoding="utf-8")

    print("\n==============================")
    print("Meilleur modèle :", meilleur_nom)
    print("Meilleure accuracy :", round(meilleur_score, 4))
    print(f"Modèle sauvegardé dans : {FICHIER_MODELE}")
    print(f"Jeu de test sauvegardé dans : {FICHIER_TEST}")
    print("==============================")


if __name__ == "__main__":
    main()