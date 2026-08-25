#importation de tous les modules permettant d'initialiser les cellules
import sqlite3 #permet de faire des requetes SQL
from LHT_Classe_element_elec import ElementElectrique
from LHT_Classe_partie_mobile import PartieMobile
from LHT_Classe_clef import Clef
from LHT_Classe_SMALT import SMALT
from LHT_Classe_serrureMere import SerrureMere
from LHT_Classe_panneau import Panneau
from LHT_Classe_cellule import Cellule
from LHT_Classe_transformateur import Transformateur
from LHT_Classe_coffret import Coffret
from LHT_Classe_eclisse import Eclisse
from LHT_Classe_boite_eclisse import BoiteEclisse
from LHT_Classe_source import Source
import numpy as np
import pandas as pd #pour extraire les donnees excel
import ast #permet de comprendre les matrice de l'excel et de les convertir en matrice python

class InitialisationCellules:

    def __init__(self, palier):
        self.palier = palier #stocke le palier choisi
        self.file_path = f"dataLHT/{palier}_LHT.xlsx" #chemin du fichier excel correspondant
        self.conn = self.create_sqlite_db_from_excel() #crée une table de données SQL à partir de l'Excel
        self.cursor = self.conn.cursor() #pour pouvoir exécuter des requêtes sur cette table SQL

    #fonction qui charge chaque feuille excel dans un DataFrame
    def load_excel_to_dataframe(self):
        # Charger toutes les feuilles d'un fichier Excel dans des DataFrames
        xls = pd.ExcelFile(self.file_path)
        tables = {}
        
        #xls.sheet_names permet d'obtenir les noms des feuilles
        for nom in xls.sheet_names:
            tables[nom] = pd.read_excel(xls, sheet_name=nom)
        #chaque feuille a une table qui stocke les données
        return tables

    #Idem qu'au dessus mais stocke seulement les données de la feuille celfs_libres : recupere les clefs de la feuille et les converti en une liste
    def extract_elements_from_excel(self, sheet_name):
        # Charge les données de la feuille Excel spécifiée sheet_name
        df = pd.read_excel(self.file_path, sheet_name=sheet_name)
        
        # Convertir les données en une liste
        elements = df.values.flatten().tolist()
        
        return elements

    #converti les dataFrames crées en une table SQL pour faire des requêtes dessus et extraire plus facilement les donnees voulues
    def create_sqlite_db_from_excel(self):
        sheets = self.load_excel_to_dataframe()

        #Crée une base de données SQLite en mémoire
        conn = sqlite3.connect(':memory:')
        
        #Charge chaque feuille Excel comme une table SQLite
        for sheet_name, df in sheets.items():
            df.to_sql(sheet_name, conn, index=False, if_exists='replace')
        
        return conn

    #prend en parametre une requete SQL et retourne le resultat
    def get_data_from_db(self, query, param=()):
        #Exécute la requête SQL passée en argument avec les paramètres fournis
        self.cursor.execute(query, param)
        #Récupère et retourne tous les résultats de la requête
        return self.cursor.fetchall()


    #Récupère les objets clefs crées appartenant à un certain élément électrique (element_elec)
    def get_clefs(self, element_elec):
        query = "SELECT * FROM clef WHERE clef.cellule =  ?" #selectionne toutes les caracts des clefs appartenant a un élément électrique
        return self.get_data_from_db(query, (element_elec,))

    #Récupère les objets partie mobile crées appartenant à une certaine cellule (cellule_name)
    def get_partie_mobile(self, cellule_name):
        query = "SELECT * FROM partie_mobile  WHERE cellule = ?"
        return self.get_data_from_db(query, (cellule_name,))

    #Récupère les objets eclisse crées appartenant à une certaine cellule (cellule_name)
    def get_eclisse(self, cellule_name):
        query = "SELECT * FROM eclisse  WHERE cellule = ?"
        return self.get_data_from_db(query, (cellule_name,))

    #Récupère les objets boite eclisse crées appartenant à une certaine cellule (cellule_name)
    def get_boite_eclisse(self, cellule_name):
        query = "SELECT * FROM boite_eclisse  WHERE cellule = ?"
        return self.get_data_from_db(query, (cellule_name,))

    #Récupère les objets smalt crées appartenant à une certaine cellule (cellule_name)
    def get_smalt(self, cellule_name):
        query = "SELECT * FROM smalt JOIN cellule ON smalt.cellule = ?"
        return self.get_data_from_db(query, (cellule_name,))

    #Récupère les objets serrures crées appartenant à une certaine cellule (cellule_name)
    def get_serrures(self, cellule_name):
        query = "SELECT * FROM serrure WHERE cellule = ?"
        return self.get_data_from_db(query, (cellule_name,))

    #Récupère les objets porte crées appartenant à une certaine cellule (cellule_name)
    def get_porte(self, cellule_name):
        query = "SELECT * FROM panneau WHERE cellule = ?"
        return self.get_data_from_db(query, (cellule_name,))

    #Récupère les objets source crées appartenant à une certaine cellule (cellule_name)
    def get_source(self, cellule_name):
        query = "SELECT * FROM source WHERE cellule = ?"
        return self.get_data_from_db(query, (cellule_name,))

    #Récupère les objets transformateur crées appartenant à une certaine cellule (cellule_name)
    def get_transformateur(self, cellule_name):
        query = "SELECT * FROM transformateur WHERE cellule = ?"
        return self.get_data_from_db(query, (cellule_name,))

    #Récupère les objets coffret crées appartenant à une certaine cellule (cellule_name)
    def get_coffret(self, cellule_name):
        query = "SELECT * FROM coffret WHERE cellule = ?"
        return self.get_data_from_db(query, (cellule_name,))

    #Récupère les objets cellule crées dont le nom est cellule_name
    def get_cellule(self, cellule_name):
        query = "SELECT * FROM cellule WHERE nom = ?"
        return self.get_data_from_db(query, (cellule_name,))

    #cette fonction initialise une cellule en créant un objet de la classe Cellule avec tous ses elements electriques   
    def init_cellule(self, cellule_name):
        
        #ici on recupere toutes les caractéristiques des différents éléments électriques de la cellule via l'excel
        clefs = self.get_clefs(cellule_name)
        partie_mobile_data = self.get_partie_mobile(cellule_name)
        smalt_data = self.get_smalt(cellule_name)
        serrures = self.get_serrures(cellule_name)
        porte_data = self.get_porte(cellule_name)
        transformateur_data = self.get_transformateur(cellule_name)
        cellule_data = self.get_cellule(cellule_name)
        coffret_data = self.get_coffret(cellule_name)
        eclisse_data = self.get_eclisse(cellule_name)
        boite_eclisse_data = self.get_boite_eclisse(cellule_name)
        source_data = self.get_source(cellule_name)
        
        if clefs != []:
            clefs_cellule = [Clef(nom=clef[0], etat=clef[1], penne=clef[2], appartenance=clef[3], type=clef[4], cellule=clef[5]) for clef in clefs]
        else:
            clefs_cellule = []
        
        #on cree l'objet PartieMobile, si il y en a un, à partir des colonnes de l'excel
        if partie_mobile_data != []:
            partie_mobile = PartieMobile(
                nom=partie_mobile_data[0][0],
                clefs=[clef for clef in clefs_cellule if clef.appartenance == partie_mobile_data[0][0]],
                type=partie_mobile_data[0][1],
                cellule=cellule_name,
                etat=partie_mobile_data[0][3],
                position=partie_mobile_data[0][4],
                representation=ast.literal_eval((partie_mobile_data[0][5])), 
                deplacement=partie_mobile_data[0][6],
                appartenance=np.array(ast.literal_eval(cellule_data[0][1]), dtype=object),
            )
        else:
            partie_mobile = []
        
        #on cree l'objet Eclisse, si il y en a un, à partir des colonnes de l'excel
        if eclisse_data != []:
            eclisse = Eclisse(
                nom=eclisse_data[0][0],
                clefs=[clef for clef in clefs_cellule if clef.appartenance == eclisse_data[0][0]],
                type=eclisse_data[0][1],
                cellule=cellule_name,
                etat=eclisse_data[0][3],
                position=eclisse_data[0][4],
                representation=ast.literal_eval((eclisse_data[0][5])), 
                deplacement=eclisse_data[0][6],
                appartenance=np.array(ast.literal_eval(cellule_data[0][1]), dtype=object),# Assuming this is stored as a string in DB
                voie=eclisse_data[0][7],
            )
        else:
            eclisse = []
        
        #on cree l'objet BoiteEclisse, si il y en a un, à partir des colonnes de l'excel
        if boite_eclisse_data != []:
            boite_eclisse = BoiteEclisse(
                nom=boite_eclisse_data[0][0],
                clefs=[clef for clef in clefs_cellule if clef.appartenance == boite_eclisse_data[0][0]],
                type=boite_eclisse_data[0][1],
                cellule=cellule_name,
                position=boite_eclisse_data[0][3],
                representation=ast.literal_eval((boite_eclisse_data[0][4])), 
                deplacement=boite_eclisse_data[0][5],
                appartenance=np.array(ast.literal_eval(cellule_data[0][1]), dtype=object),# Assuming this is stored as a string in DB
                voie=boite_eclisse_data[0][6],
                stock_eclisses = boite_eclisse_data[0][7]
            )
        else:
            boite_eclisse = []
        
        #on cree l'objet SMALT, si il y en a un, à partir des colonnes de l'excel
        if smalt_data != []:    
            smalt = SMALT(
                nom=smalt_data[0][0],
                clefs=[clef for clef in clefs_cellule if clef.appartenance == smalt_data[0][0]],
                type=smalt_data[0][1],
                etat=smalt_data[0][2],
                cellule=cellule_name,
                deplacement=smalt_data[0][3],
                representation=ast.literal_eval((smalt_data[0][5])),
                appartenance=np.array(ast.literal_eval(cellule_data[0][1]), dtype=object),
            )
        else:
            smalt = []
        
        #on cree les objets Serrure, si il y en a, à partir des colonnes de l'excel   
        serrure_objects = []
        if serrures != []:
            for serrure in serrures:
                serrure_objects.append(SerrureMere(
                    nom=serrure[0],
                    type=serrure[2],
                    clefs=[clef for clef in clefs_cellule if clef.appartenance == serrure[0]],
                    cellule=cellule_name
                ))
        
        #on cree l'objet Porte, si il y en a un, à partir des colonnes de l'excel
        if porte_data != []:
            porte = Panneau(
                nom=porte_data[0][0],
                type=porte_data[0][1],
                clefs=[clef for clef in clefs_cellule if clef.appartenance == porte_data[0][0]],
                etat=porte_data[0][2],
                cellule=cellule_name,
            )
        else:
            porte = []
        
        #on cree l'objet Transformateur, si il y en a un, à partir des colonnes de l'excel
        if transformateur_data != []:
            transformateur = Transformateur(
                nom=transformateur_data[0][0],
                type=transformateur_data[0][1],
                clefs=[clef for clef in clefs_cellule if clef.appartenance == transformateur_data[0][0]],
                cellule=cellule_name,
                )
        else:
            transformateur = []
        
        #on cree l'objet Coffret, si il y en a un, à partir des colonnes de l'excel
        if coffret_data != []:
            coffret = Coffret(
                nom=coffret_data[0][0],
                type=coffret_data[0][1],
                cellule=cellule_name,
                clefs=[clef for clef in clefs_cellule if clef.appartenance == coffret_data[0][0]],
                )
        else:
            coffret = []
        
        #on cree l'objet Source, si il y en a un, à partir des colonnes de l'excel
        if source_data != []:
            source = Source(
                nom=source_data[0][0],
                type=source_data[0][1],
                cellule=cellule_name,
                clefs=[clef for clef in clefs_cellule if clef.appartenance == source_data[0][0]],
                )
        else:
            source = []
                
        #on renvoie l'objet Cellule contenant tous les elements (son nom, sa representation matricielle, ses éléments électriques...)
        return Cellule(nom=(cellule_data[0][0]), matrice=np.array(ast.literal_eval(cellule_data[0][1]), dtype=object), voisine=ast.literal_eval(cellule_data[0][2]), position_x=cellule_data[0][3], partie_mobile=partie_mobile, smalt=smalt, serrures=serrure_objects, porte=porte, transformateur=transformateur, longueur=6, coffret=coffret, eclisse=eclisse, boite_eclisse=boite_eclisse, source=source, largeur=3, statut=cellule_data[0][4], position_y=cellule_data[0][5])

    
    #permet de réinitialiser toutes les cellules des tableaux
    def reinitialiser_cellules(self):    
        self.conn = self.create_sqlite_db_from_excel()
        self.cursor = self.conn.cursor()
        
        self.cursor.execute("SELECT nom FROM cellule") #on recupere le nom de toutes les cellules
        noms_cellules = self.cursor.fetchall()
        
        #on initialise toutes les cellules
        liste_cellules = [self.init_cellule(nom[0]) for nom in noms_cellules] 
        
        #on initialise les clefs libres (celle au BdC)
        ElementElectrique.clefs_libres = self.extract_elements_from_excel('clefs_libres')
        
        self.conn.close()
        return liste_cellules
