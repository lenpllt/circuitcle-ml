#on importe ce qui est necessaire
from LHC_Classe_configuration import Configuration # type: ignore
import tkinter as tk
from tkinter import ttk, messagebox, PhotoImage
from LHC_Classe_choix_cellules import ChoixCellules
from LHC_Classe_etapes_choisies import EtapesChoisies
from LHC_Classe_initialisation_cellules import InitialisationCellules  
from LHC_Classe_conditions_fin import ConditionsFin
import os
import sys
from datetime import datetime

# Classe pour rediriger les print dans une fenetre tkinter
class RedirectText(object):
    def __init__(self, text_widget):
        self.output = text_widget

    def write(self, string):
        self.output.insert(tk.END, string)
        self.output.see(tk.END)

    def flush(self):
        pass

class ExecutionLHC:
    def __init__(self, window):
        self.window = window

        #paramètres visuels (taille...) de la représentation si on choisit le dessin du tableau entier
        self.parametres_tableau_entier = {
            'taille': 30, #taille du dessin
            'x_coin': 100, #position horizontale du debut du dessin (premier element de la matrice)
            'y_coin': 130, #position verticale du debut du dessin (premier element de la matrice)
            'taille_serrure_mere': 120, #taille verticale de la serrure mere
            'a': 5*30, #decalage horizontal entre les cellule : un décalage de 5*taille correspondent à coller les matrices les unes à côté des autres
            'b': 30, #decalage vertical entre les cellules
            'largeur_fenetre': 1500, #largueur de la fenetre finale
            'hauteur_fenetre': 700 #hauteur de la fenetre finale
        }

        #parametres visuels si on choisit le dessin d'une seule cellule LHC
        self.parametres_cellule_unique = {
            'taille': 30,
            'x_coin': 100,
            'y_coin': 130,
            'taille_serrure_mere': 120,
            'a': 5*30,
            'b': 30,
            'largeur_fenetre': 800,
            'hauteur_fenetre': 700
        }

    def restore_stdout_stderr(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    #cree la fenêtre finale permettant d'exécuter les étapes, de choisir les étapes et d'ajouter des conditions : elle utilise les paramètres définis au dessus
    def creer_fenetre(self, liste_cellules, liste, params, palier, etapes):
        fenetre = tk.Toplevel(self.window) #création de la fenêtre
        fenetre.title(f"LHC-{palier}")
        icon = PhotoImage(file='Logo_sans_fond.png')
        fenetre.iconphoto(False, icon)
        fenetre.iconbitmap('Logo_avec_fond.ico')
        notebook = ttk.Notebook(fenetre) #création d'un notebook : permet la création des differents onglets

        #crée chaque onglet
        onglet1 = tk.Frame(notebook)
        onglet2 = tk.Frame(notebook)
        onglet3 = tk.Frame(notebook)
        onglet4 = tk.Frame(notebook) #Console

        #ajoute chaque onglet dans le notebook
        notebook.add(onglet1, text="test")
        notebook.add(onglet2, text="choix etapes")
        notebook.add(onglet3, text="ajout conditions")
        notebook.add(onglet4, text="Console")
        notebook.pack(expand=True, fill="both")

        configuration = Configuration()
        #cree la matrice finale, utilisée pour la recherche de danger
        matrice_finale = configuration.matrice_unique(liste_cellules)

        #Ajout des conditions supplementaires et les étapes que l'on peut choisir dans les onglet2 et 3
        conditions = ConditionsFin(matrice_finale,liste_cellules)
        conditions.ajout_conditions(onglet3, liste)
        etapes.choix_etapes(onglet2, liste)
        etapes.reinitialiser_etapes(liste_cellules)

        #Ajoute dans l'onglet 1 le dessin et les boutons d'execution des étapes
        configuration.test_tout(
            onglet1, conditions, etapes, 
            params["a"], params["b"], liste_cellules, liste,
            params["largeur_fenetre"], params["hauteur_fenetre"],
            params["taille"], params["x_coin"], params["taille_serrure_mere"] + 10,
            params["taille_serrure_mere"], 1000, palier
        )

        #fonction qui permet simplement de corriger un problème de déplacement via les flèches 
        def on_tab_change(event):
            selected_tab = notebook.index(notebook.select())
            if selected_tab == 0 and onglet1.winfo_children():
                canvas = onglet1.winfo_children()[0]
                canvas.focus_set()
        notebook.bind("<<NotebookTabChanged>>", on_tab_change)

        #Console redirigée
        text_console = tk.Text(onglet4, wrap="word", height=30)
        text_console.pack(expand=True, fill="both")
        sys.stdout = RedirectText(text_console)
        sys.stderr = RedirectText(text_console)

        def clear_console():
            text_console.delete("1.0", tk.END)

        bouton_clear = tk.Button(onglet4, text="Effacer la console", command=clear_console)
        bouton_clear.pack(pady=5)

        def exporter_console():
            contenu = text_console.get("1.0", tk.END).strip()
            if contenu:
                now = datetime.now()
                date_heure = now.strftime("%d-%m-%Y_%H-%M-%S")

                if "_preVD_" in palier:
                    version = "preVD"
                elif "_postVD_" in palier:
                    version = "postVD"
                else:
                    version = "VD"

                num_palier = palier.split("_")[0]
                nom_fichier = f"LHC_{num_palier}_{version}_{date_heure}.txt"
                dossier = "historiqueLHC"

                os.makedirs(dossier, exist_ok=True)
                chemin_fichier = os.path.join(dossier, nom_fichier)

                with open(chemin_fichier, "w", encoding="utf-8") as f:
                    f.write(contenu)

                messagebox.showinfo("Export réussi", f"Console exportée dans {chemin_fichier}")
            else:
                messagebox.showwarning("Console vide", "Aucun contenu à exporter.")

        bouton_exporter = tk.Button(onglet4, text="Exporter la console", command=exporter_console)
        bouton_exporter.pack(pady=5)

        fenetre.protocol("WM_DELETE_WINDOW", lambda: (self.restore_stdout_stderr(), fenetre.destroy()))

    #fonction permettant de creer le tableau LHC
    def tableauLHC(self, initialisation, palier, etapes):
        liste_cellules = initialisation.reinitialiser_cellules()
        self.creer_fenetre(liste_cellules, liste_cellules, self.parametres_tableau_entier, palier, etapes)

    #fonction permettant de creer dynamiquement une fonction pour chaque cellule
    def generer_fonctions_dynamiques(self, palier, initialisation, etapes):
        cursor = initialisation.cursor
        #Récupére toutes les cellules avec leurs paramètres
        cursor.execute("SELECT nom, voisine, position_x FROM cellule ORDER BY position_x ASC")
        cellules = cursor.fetchall()

        #Itére sur chaque cellule pour créer dynamiquement une fonction
        for cellule in cellules:
            nom, voisine, position_x = cellule
            if nom.startswith('LHC'):
                def fonction(palier_local=palier, position_x_cellule=position_x, nom_voisine=voisine):
                    liste_cellules = initialisation.reinitialiser_cellules()
                    #Filtre les cellules ayant le même position_x
                    liste = [c for c in liste_cellules if c.position_x == position_x_cellule]

                    if str(nom_voisine).startswith('LHC'):
                        cellule_voisine = next((c for c in liste_cellules if c.nom == nom_voisine), None)
                        if cellule_voisine:
                            liste_voisine = [c for c in liste_cellules if c.position_x == cellule_voisine.position_x]
                            liste_finale = liste + liste_voisine if cellule_voisine.position_x - position_x_cellule > 0 else liste_voisine + liste
                            self.creer_fenetre(liste_cellules, liste_finale, self.parametres_cellule_unique, palier, etapes)
                            return
                    self.creer_fenetre(liste_cellules, liste, self.parametres_cellule_unique, palier, etapes)

                #Ajoute la fonction dans l'espace de noms global
                globals()[nom] = fonction

    #crée la deuxieme fenetre pour choisir les cellules a afficher
    def fenetre_principale(self, palier):
        etapes = EtapesChoisies()
        initialisation = InitialisationCellules(palier)
        self.generer_fonctions_dynamiques(palier, initialisation, etapes)

        fenetre = tk.Toplevel(self.window)
        fenetre.title(f"LHC-{palier}")
        icon = PhotoImage(file='Logo_sans_fond.png')
        fenetre.iconphoto(False, icon)
        fenetre.iconbitmap('Logo_avec_fond.ico')
        notebook = ttk.Notebook(fenetre)

        onglet1 = tk.Frame(notebook)
        onglet2 = tk.Frame(notebook)

        notebook.add(onglet1, text="Verrouillage LHC")
        notebook.add(onglet2, text="choix cellules")
        notebook.pack(expand=True, fill="both")

        cursor = initialisation.cursor
        cursor.execute("SELECT nom FROM cellule ORDER BY position_x ASC")
        cellules_noms = cursor.fetchall()

        boutonLHC = tk.Button(onglet1, text="tableau LHC", command=lambda: self.tableauLHC(initialisation, palier, etapes))
        boutonLHC.pack()

        #Crée un bouton pour chaque cellule
        for cellule in cellules_noms:
            cellule_nom = cellule[0]
            if cellule_nom.startswith('LHC'):
                bouton = tk.Button(onglet1, text=cellule_nom, command=lambda nom=cellule_nom: globals()[nom](palier))
                bouton.pack()

        liste_cellules = initialisation.reinitialiser_cellules()
        cellules = ChoixCellules(liste_cellules)
        cellules.choix_cellule(onglet2, liste_cellules)

        boutonchoix = tk.Button(onglet1, text="choix cellules", command=lambda: self.cellules_choisies(cellules, initialisation, palier, etapes))
        boutonchoix.pack()

    #si on choisit des cellules à la main, cette fonction permet d'afficher ces cellules   
    def cellules_choisies(self, cellules, initialisation, palier, etapes):
        liste_cellules = initialisation.reinitialiser_cellules()
        listes = cellules.choix_cellules
        liste_noms = [liste.nom for liste in listes]
        liste_reinit = [cell for cell in liste_cellules if cell.nom in liste_noms]
        if liste_reinit:
            self.creer_fenetre(liste_cellules, liste_reinit, self.parametres_cellule_unique, palier, etapes)
        else:
            messagebox.showwarning("Avertissement", "Aucune cellule n'a été choisie !")

    #permet de choisir si on veut les dernières modifications en cours de déploiement ou la configuration avant les modifications de la VD
    def choix_modif_VD(self, palier):
        fenetre = tk.Toplevel(self.window)
        fenetre.geometry("400x200")
        fenetre.title(f"Choix VD modif palier {palier}")
        icon = PhotoImage(file='Logo_sans_fond.png')
        fenetre.iconphoto(False, icon)
        fenetre.iconbitmap('Logo_avec_fond.ico')
        boutonPreVD = tk.Button(fenetre, text="modifs preVD", command=lambda: self.fenetre_principale(f"{palier}_preVD_LHC"))
        boutonPreVD.pack()
        boutonPostVD = tk.Button(fenetre, text="modifs postVD", command=lambda: self.fenetre_principale(f"{palier}_postVD_LHC"))
        boutonPostVD.pack()

    #Creation de la premiere fenetre avec deux boutons qui permettront d'aller au palier voulu
    def execution(self, window):
        self.window = tk.Toplevel(window)
        self.window.geometry("400x200")
        self.window.title("Choix palier, tableau LHC")
        icon = PhotoImage(file='Logo_sans_fond.png')
        self.window.iconphoto(False, icon)
        self.window.iconbitmap('Logo_avec_fond.ico')
        bouton900 = tk.Button(self.window, text="900", command=lambda: self.choix_modif_VD(900))
        bouton900.pack()
        bouton1300 = tk.Button(self.window, text="1300", command=lambda: self.choix_modif_VD(1300))
        bouton1300.pack()
        bouton1400 = tk.Button(self.window, text="1400", command=lambda: self.choix_modif_VD(1400))
        bouton1400.pack()

#lance l'interface
def main(root):
    window = tk.Toplevel(root)
    app = ExecutionLHC(window)
    app.execution(window)
