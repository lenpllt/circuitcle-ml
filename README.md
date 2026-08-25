# CircuitClé V3.5 — Application IA de détection de situations dangereuses

**EDF / DIPDE — M2 Data Science — Léna Pillet — 2026**

Application Streamlit de détection automatique de situations dangereuses dans les consignations électriques de tableaux de distribution (LHC / LHT) par intelligence artificielle (ML + RAG + LLM).

---

## Accès en ligne (sans installation)

| Application | URL |
|---|---|
| App principale (analyse IA) | https://circuitcle-ml.streamlit.app/ |
| Tableau de bord IA (TDB) | https://circuitcle-ml-tdb.streamlit.app/ |

---

## Prérequis d'installation (exécution locale)

- **Python** : 3.10 ou supérieur
- **Système** : macOS, Windows ou Linux
- **Clé API Anthropic** : nécessaire pour les fonctions RAG + LLM (modèle `claude-haiku-4-5-20251001`)
- **pip** : gestionnaire de paquets Python à jour (`pip install --upgrade pip`)

---

## Étapes d'installation

### 1. Cloner ou décompresser le projet

```bash
# Depuis GitHub (optionnel)
git clone https://github.com/lenpllt/circuitcle-ml
cd circuitcle-ml

# Ou simplement décompresser le ZIP dans un dossier local
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

Packages installés : `streamlit`, `scikit-learn`, `pandas`, `plotly`, `anthropic`, `joblib`.

### 3. Configurer la clé API Anthropic

Créer le fichier `.streamlit/secrets.toml` (à partir du modèle fourni) :

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Puis renseigner votre clé API dans `.streamlit/secrets.toml` :

```toml
ANTHROPIC_API_KEY = "sk-ant-VOTRE-CLE-ICI"
```

> ⚠️ Ce fichier contient un secret — ne jamais le versionner ni le partager.

### 4. Lancer l'application

```bash
# Application principale (analyse IA + RAG + LLM)
streamlit run app_ml.py

# Tableau de bord IA (performances ML + benchmark SQL)
streamlit run tdb_ia.py
```

L'application s'ouvre automatiquement dans le navigateur par défaut sur `http://localhost:8501`.

---

## Identifiants de test

| Élément | Valeur |
|---|---|
| Authentification utilisateur | **Aucune** — accès direct sans login ni mot de passe |
| Identifiant administrateur | **Aucun** — voir section *Tableau de bord* ci-dessous |

L'application n'implémente pas de système d'authentification : elle est destinée à un usage interne EDF/DIPDE dans un environnement réseau maîtrisé.

---

## Scénarios de démonstration

Le dossier `DEMO_CircuitCle/` contient deux fichiers de logs prêts à l'emploi pour tester l'application :

| Fichier | Scénario | Résultat attendu |
|---|---|---|
| `LHC_1300_preVD_01-04-2025_12-19-00__NORMAL.txt` | Situation normale | ✅ Aucune anomalie détectée |
| `LHC_900_preVD_18-06-2025_09-23-39__DANGEREUX.txt` | Situation dangereuse | ⚠️ Danger détecté par le modèle ML |

### Comment tester

1. Ouvrir l'application : https://circuitcle-ml.streamlit.app/ (ou `streamlit run app_ml.py` en local)
2. Dans l'interface, **uploader** l'un des fichiers `.txt` du dossier `DEMO_CircuitCle/`
3. L'application analyse le log, prédit la situation (normale ou dangereuse) et permet d'interroger la base RAG via le chatbot IA
4. Tester ensuite avec le second fichier pour comparer les deux scénarios

---

## Base de données SQL — Connexion et identifiants

La base de données est **SQLite en mémoire** (`sqlite3.connect(':memory:')`).

| Paramètre | Valeur |
|---|---|
| Moteur | SQLite 3 (bibliothèque standard Python) |
| Type | In-memory (pas de fichier persistant) |
| Identifiant / mot de passe | **Aucun** — base locale sans authentification |
| Fichier de base | **Aucun** — la base est reconstruite à chaque session |

### Fonctionnement

Au démarrage de l'application, les fichiers Excel (`dataLHC/*.xlsx`, `dataLHT/*.xlsx`) sont chargés automatiquement par `LHC_Classe_initialisation_cellules.py` et `LHT_Classe_initialisation_cellules.py`. Chaque feuille Excel devient une table SQLite en mémoire :

| Table | Contenu |
|---|---|
| `clef` | Clefs électriques par cellule |
| `partie_mobile` | Parties mobiles par cellule |
| `smalt` | SMALT (Système de Mise à la Terre) |
| `cellule` | Données des cellules HTA |
| `transformateur` | Transformateurs |
| `coffret` | Coffrets électriques |
| `panneau` | Panneaux |

Des index sont créés dynamiquement sur la colonne `cellule` pour optimiser les requêtes (voir `benchmark_sql.py` et résultats dans `benchmark_resultats.json`).

Pour régénérer les résultats du benchmark :

```bash
python benchmark_sql.py
```

---

## Accès administrateur / Tableau de bord

Il n'existe pas de back-office d'administration distinct. Le **Tableau de bord IA** (`tdb_ia.py`) remplit ce rôle de supervision :

- **URL déployée** : https://circuitcle-ml-tdb.streamlit.app/
- **Local** : `streamlit run tdb_ia.py` → `http://localhost:8502`

Le TDB affiche :
- Synthèse des performances du modèle ML retenu
- Comparaison des 5 algorithmes (Logistic Regression, Decision Tree, Random Forest, KNN, MLP)
- Matrice de confusion + importance des variables
- Benchmark SQL avant/après indexation

Aucun identifiant requis.

---

## Compatibilité multi-navigateur

| Navigateur | Statut |
|---|---|
| Google Chrome | ✅ Testé — recommandé |
| Safari (macOS) | ✅ Testé — compatible |
| Mozilla Firefox | ✅ Compatible (Streamlit officiel) |
| Microsoft Edge | ✅ Compatible (Streamlit officiel) |

> L'application utilise Streamlit ≥ 1.28 et Plotly pour les graphiques interactifs. Tout navigateur moderne supportant ES6+ est compatible.

---

## Structure du projet

```
CIRCUIT CLÉ V3.5/
├── DEMO_CircuitCle/
│   ├── LHC_1300_preVD_..._NORMAL.txt    # Log de démonstration — situation normale
│   └── LHC_900_preVD_..._DANGEREUX.txt  # Log de démonstration — situation dangereuse
├── app_ml.py                        # Application principale
├── tdb_ia.py                        # Tableau de bord IA
├── rag_engine.py                    # Moteur RAG (TF-IDF hybride)
├── requirements.txt                 # Dépendances Python
├── style.css                        # Feuille de style Streamlit
├── dataset_logs_ml.csv              # Dataset ML (209 logs)
├── meilleur_modele_ml.joblib        # Modèle ML sérialisé
├── benchmark_resultats.json         # Résultats benchmark SQL
├── etapes_choisies_LHC.json         # Étapes procédurales LHC
├── etapes_choisies_LHT.json         # Étapes procédurales LHT
├── LHC_Classe_*.py                  # Classes métier LHC (20 fichiers)
├── LHT_Classe_*.py                  # Classes métier LHT (15 fichiers)
├── Classe_execution*.py             # Exécution des procédures
├── dataLHC/                         # Données Excel tableaux LHC
├── dataLHT/                         # Données Excel tableaux LHT
├── historiqueLHC/                   # Logs historiques LHC (.txt)
├── historiqueLHT/                   # Logs historiques LHT (.txt)
├── contexte_metier/                 # Documents métier (.txt)
├── train_modele_ml.py               # Entraînement du modèle ML
├── build_dataset_ml.py              # Construction du dataset
├── generate_synthetic_logs.py       # Génération de logs synthétiques
├── benchmark_sql.py                 # Script benchmark SQL
├── .streamlit/
│   └── secrets.toml.example         # Modèle de configuration (clé API)
└── sql_schema.sql                   # Documentation schéma SQLite
```

---

## Liens utiles

- **Dépôt GitHub** : https://github.com/lenpllt/circuitcle-ml
- **App déployée** : https://circuitcle-ml.streamlit.app/
- **TDB déployé** : https://circuitcle-ml-tdb.streamlit.app/
- **Modèle LLM** : Anthropic Claude Haiku (`claude-haiku-4-5-20251001`)
- **Documentation Streamlit** : https://docs.streamlit.io
