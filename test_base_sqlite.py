import sqlite3
from LHC_Classe_initialisation_cellules import InitialisationCellules as InitLHC
from LHT_Classe_initialisation_cellules import InitialisationCellules as InitLHT


def afficher_tables(conn: sqlite3.Connection, titre: str) -> None:
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = cursor.fetchall()

    print("\n" + "=" * 60)
    print(titre)
    print("=" * 60)
    print("Tables présentes dans la base :")
    for table in tables:
        print(f"- {table[0]}")


def afficher_apercu_table(conn: sqlite3.Connection, nom_table: str, limite: int = 5) -> None:
    cursor = conn.cursor()
    print(f"\nAperçu de la table '{nom_table}' (limite {limite}) :")
    try:
        cursor.execute(f"SELECT * FROM {nom_table} LIMIT {limite};")
        rows = cursor.fetchall()

        if not rows:
            print("  -> table vide")
            return

        for row in rows:
            print(row)

    except Exception as e:
        print(f"  -> erreur lors de la lecture de la table {nom_table} : {e}")


def tester_lhc(palier: str = "900_preVD_LHC") -> None:
    print("\nTEST BASE SQLITE - LHC")
    init_lhc = InitLHC(palier)
    conn = init_lhc.conn

    afficher_tables(conn, f"Base SQLite LHC créée depuis : {palier}.xlsx")
    afficher_apercu_table(conn, "clef")
    afficher_apercu_table(conn, "cellule")
    afficher_apercu_table(conn, "partie_mobile")


def tester_lht(palier: str = "900") -> None:
    print("\nTEST BASE SQLITE - LHT")
    init_lht = InitLHT(palier)
    conn = init_lht.conn

    afficher_tables(conn, f"Base SQLite LHT créée depuis : {palier}_LHT.xlsx")
    afficher_apercu_table(conn, "clef")
    afficher_apercu_table(conn, "cellule")
    afficher_apercu_table(conn, "partie_mobile")


if __name__ == "__main__":
    try:
        tester_lhc("900_preVD_LHC")
    except Exception as e:
        print(f"\nErreur test LHC : {e}")

    try:
        tester_lht("900")
    except Exception as e:
        print(f"\nErreur test LHT : {e}")