#on importe ce qui est necessaire
import tkinter as tk
from tkinter import ttk, messagebox, PhotoImage
from LHT_Classe_conditions_fin import ConditionsFin
from LHT_Classe_initialisation_cellules import InitialisationCellules # type: ignore
from LHT_Classe_etapes_choisies import EtapesChoisies
from LHT_Classe_configuration import Configuration # type: ignore
import sys
from datetime import datetime
import os

# Classe pour rediriger les print dans une fenetre tkinter
class RedirectText(object):
    def __init__(self, text_widget):
        self.output = text_widget
        self.buffer = []

    def write(self, string):
        self.output.insert(tk.END, string)
        self.output.see(tk.END)
        self.buffer.append(string)

    def flush(self):
        pass

    def get_buffer_content(self):
        return ''.join(self.buffer)

class ExecutionLHT:
    #initialise les parametres visuels (taille...) du dessin final
    def __init__(self, window):
        self.window = window

        self.parametres_tableau_entier = {  
            'taille': 30, #taille du dessin
            'x_coin': 0, #position horizontale du debut du dessin (premier element de la matrice)
            'y_coin': 200, #position verticale du debut du dessin (premier element de la matrice)
            'taille_serrure_mere': 120, #taille verticale de la serrure mere 
            'a': 5*30, #decalage horizontal entre les cellule : un décalage de 5*taille correspondent à coller les matrices les unes à côté des autres
            'b':30, #decalage vertical entre les cellules
            'largeur_fenetre': 1500, #largueur de la fenetre finale
            'hauteur_fenetre': 700 #hauteur de la fenetre finale
        }

    def restore_stdout_stderr(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    #cree la fenêtre finale permettant d'exécuter les étapes, de choisir les étapes et d'ajouter des conditions : elle utilise les paramètres définis au dessus
    def creer_fenetre(self, liste_cellules, liste, params, palier, etapes):
        fenetre = tk.Toplevel(self.window) #création de la fenêtre
        fenetre.title(f"LHT-{palier}")
        icon = PhotoImage(file='Logo_sans_fond.png')
        fenetre.iconphoto(False, icon)
        fenetre.iconbitmap('Logo_avec_fond.ico') 
        notebook = ttk.Notebook(fenetre)    #creation d'un notebook : permet la création des differents onglets

        #créé chaque onglet
        onglet1 = tk.Frame(notebook)
        onglet2 = tk.Frame(notebook)
        onglet3 = tk.Frame(notebook)
        onglet4 = tk.Frame(notebook)

        #ajoute chaque onglet dans le notebook
        notebook.add(onglet1, text="test")
        notebook.add(onglet2, text="choix etapes")
        notebook.add(onglet3, text="ajout conditions")
        notebook.add(onglet4, text="Console")
        notebook.pack(expand=True, fill="both")

        configuration = Configuration()
        #cree la matrice finale, utilisée pour la recherche de danger
        matrice_finale = configuration.matrice_unique(liste_cellules)

        # Ajout des conditions supplementaires et les étapes que l'on peut choisir dans les onglet2 et 3
        conditions = ConditionsFin(matrice_finale)
        conditions.ajout_conditions(onglet3, liste)
        etapes.choix_etapes(onglet2, liste)
        etapes.reinitialiser_etapes(liste_cellules)

        #Ajoute dans l'onglet 1 le dessin et les boutons d'execution des étapes
        configuration.test_tout(onglet1, conditions, etapes, configuration.etapes_aleatoire(liste, conditions),
                params["a"], params["b"], liste_cellules, liste,
                params["largeur_fenetre"], params["hauteur_fenetre"],
                params["taille"], params["x_coin"], params["y_coin"],
                params["taille_serrure_mere"], 40, palier)

        #fonction qui permet simplement de corriger un problème de déplacement via les flèches 
        def on_tab_change(event):
            selected_tab = notebook.index(notebook.select())
            if selected_tab == 0:
                canvas = onglet1.winfo_children()[0]
                canvas.focus_set()
        notebook.bind("<<NotebookTabChanged>>", on_tab_change)

        text_console = tk.Text(onglet4, wrap="word", height=30)
        text_console.pack(expand=True, fill="both")
        redirect = RedirectText(text_console)
        sys.stdout = redirect
        sys.stderr = redirect

        def clear_console():
            text_console.delete("1.0", tk.END)

        bouton_clear = tk.Button(onglet4, text="Effacer la console", command=clear_console)
        bouton_clear.pack(pady=5)

        def exporter_console():
            now = datetime.now()
            timestamp = now.strftime("%d-%m-%Y_%H-%M-%S")
            dossier = "historiqueLHT"
            if not os.path.exists(dossier):
                os.makedirs(dossier)
            filename = f"{dossier}/LHT_{palier}_{timestamp}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(redirect.get_buffer_content())
            messagebox.showinfo("Exportation réussie", f"Les logs ont été exportés dans :\n{filename}")

        bouton_exporter = tk.Button(onglet4, text="Exporter la console", command=exporter_console)
        bouton_exporter.pack(pady=5)

        fenetre.protocol("WM_DELETE_WINDOW", lambda: (self.restore_stdout_stderr(), fenetre.destroy()))

    #fonction permettant de creer le tableau LHT    
    def tableauLHT(self, palier, etapes):
        initialisation = InitialisationCellules(palier)
        liste_cellules = initialisation.reinitialiser_cellules()
        self.creer_fenetre(liste_cellules, liste_cellules, self.parametres_tableau_entier, palier, etapes)

    #Creation de la premiere fenetre avec deux boutons qui permettront d'aller au palier voulu
    def execution(self, window):
        window = tk.Toplevel(window)
        window.geometry("400x200")
        window.title("Choix palier, tableau LHT")
        icon = PhotoImage(file='Logo_sans_fond.png')
        window.iconphoto(False, icon)
        window.iconbitmap('Logo_avec_fond.ico')
        etapes = EtapesChoisies()   #initialisation de l'objet de la classe EtapesChoisies
        bouton900 = tk.Button(window, text="900", command=lambda : self.tableauLHT(900, etapes))
        bouton900.pack()
        bouton1300 = tk.Button(window, text="1300 2 tranches", command=lambda : self.tableauLHT("1300_2tranches", etapes))
        bouton1300.pack()
        bouton1300_4tranches = tk.Button(window, text="1300 4 tranches", command=lambda : self.tableauLHT("1300_4tranches", etapes))
        bouton1300_4tranches.pack()
        bouton1400 = tk.Button(window, text="1400", command=lambda : self.tableauLHT(1400, etapes))
        bouton1400.pack()

#lance l'interface
def main(root):
    window = tk.Toplevel(root)
    app = ExecutionLHT(window)
    app.execution(window)
