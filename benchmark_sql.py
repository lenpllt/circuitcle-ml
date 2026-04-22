import json
import sqlite3
import time
from statistics import mean

from LHC_Classe_initialisation_cellules import InitialisationCellules as InitLHC


def mesurer_temps_requete(conn, requete, repetitions=100):
    cursor = conn.cursor()
    temps = []

    for _ in range(repetitions):
        debut = time.perf_counter()
        cursor.execute(requete)
        cursor.fetchall()
        fin = time.perf_counter()
        temps.append(fin - debut)

    return mean(temps)


def afficher_indexes(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index';")
    indexes = cursor.fetchall()
    print("\nIndexes présents :")
    if not indexes:
        print("Aucun index")
    else:
        for idx in indexes:
            print("-", idx[0])


def main():
    print("===== BENCHMARK SQL =====")

    init = InitLHC("900_preVD_LHC")
    conn = init.conn
    cursor = conn.cursor()

    requetes = {
        "Q1_clef_par_cellule": "SELECT * FROM clef WHERE cellule = 'LHC005JA';",
        "Q2_partie_mobile_par_cellule": "SELECT * FROM partie_mobile WHERE cellule = 'LHC005JA';",
        "Q3_smalt_par_cellule": "SELECT * FROM smalt WHERE cellule = 'LHC005JA';"
    }

    print("\n--- Temps AVANT optimisation ---")
    resultats_avant = {}
    for nom, req in requetes.items():
        t = mesurer_temps_requete(conn, req)
        resultats_avant[nom] = t
        print(f"{nom} : {t:.8f} sec")

    afficher_indexes(conn)

    # Création d’index pour optimiser les recherches par cellule
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_clef_cellule ON clef(cellule);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_partie_mobile_cellule ON partie_mobile(cellule);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_smalt_cellule ON smalt(cellule);")
    conn.commit()

    print("\n--- Temps APRÈS optimisation ---")
    resultats_apres = {}
    for nom, req in requetes.items():
        t = mesurer_temps_requete(conn, req)
        resultats_apres[nom] = t
        print(f"{nom} : {t:.8f} sec")

    afficher_indexes(conn)

    print("\n--- Gain observé ---")
    for nom in requetes.keys():
        avant = resultats_avant[nom]
        apres = resultats_apres[nom]
        gain = ((avant - apres) / avant * 100) if avant > 0 else 0
        print(f"{nom} : gain = {gain:.2f}%")

    # Sauvegarde pour le tableau de bord Streamlit (tdb_ia.py)
    with open("benchmark_resultats.json", "w", encoding="utf-8") as f:
        json.dump({"avant": resultats_avant, "apres": resultats_apres}, f, indent=2)
    print("\nRésultats sauvegardés dans benchmark_resultats.json")

if __name__ == "__main__":
    main()