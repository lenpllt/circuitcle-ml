import pandas as pd
from LHC_Classe_initialisation_cellules import InitialisationCellules as InitLHC


def analyse_valeurs_manquantes(conn):
    print("\n===== ANALYSE DES VALEURS MANQUANTES =====")

    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)

    for table in tables['name']:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)

        missing = df.isnull().sum().sum()

        print(f"\nTable : {table}")
        print(f"Valeurs manquantes totales : {missing}")

        if missing > 0:
            print(df.isnull().sum())


def analyse_incoherences(conn):
    print("\n===== ANALYSE DES INCOHERENCES =====")

    # Exemple simple : vérifier que toutes les clés ont un état valide
    df_clef = pd.read_sql("SELECT * FROM clef", conn)

    valeurs_valides = ["presente", "absente", "prisonniere"]

    incoherences = df_clef[~df_clef.iloc[:,1].isin(valeurs_valides)]

    print("\nIncohérences sur les états des clés :")
    print(incoherences)

    # Vérifier doublons
    doublons = df_clef[df_clef.duplicated()]

    print("\nDoublons détectés :")
    print(doublons)


def main():
    print("===== ANALYSE QUALITE DES DONNEES =====")

    init = InitLHC("900_preVD_LHC")
    conn = init.conn

    analyse_valeurs_manquantes(conn)
    analyse_incoherences(conn)


if __name__ == "__main__":
    main()