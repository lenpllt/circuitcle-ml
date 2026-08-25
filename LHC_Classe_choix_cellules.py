import tkinter as tk #pour l'affiche graphique
from tkinter import ttk, messagebox
from LHC_Classe_cellule import Cellule 

#classe permettant de choisir les cellules a afficher
class ChoixCellules():
    
    #caractéristiques de la classe
    def __init__(self, liste_cellules):
        self.liste_cellules = liste_cellules #liste des cellules qu'on peut choisir d'afficher
        self.choix_cellules = [] #liste qui stockera les différentes cellules choisies (que l'on veut afficher)

    #permet d'ajouter la cellule sélectionnée à la liste des cellules choisies 
    def ajouter_cellule(self, cellule_combobox, condition_text, liste_cellules):
        nom_cellule = cellule_combobox.get() #récupère la cellule sélectionnée parmis le bandeau déroulant
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, nom_cellule)
        liste = cellule.cellules_communes(liste_cellules) #on récupère la cellule sélectionnée, ainsi que la cellule avec laquelle elle est reliée le cas échéant
       
        for cellules in liste:
            cellule_obj = next((c for c in liste_cellules if c.nom == cellules), None)
            self.choix_cellules.append(cellule_obj)

         #on écrit dans l'interface le nom des cellules ajoutées    
        ecriture_cellule = f"{liste}"
        current_text = condition_text.get("1.0", tk.END).strip()
        new_text = f"{current_text}\n{ecriture_cellule}" if current_text else ecriture_cellule
        condition_text.delete("1.0", tk.END)
        condition_text.insert(tk.END, new_text + "\n")

    #supprime la dernière cellule choisie   
    def supprimer_derniere_cellule(self, liste_cellules, condition_text):
        if self.choix_cellules:
            cellule = self.choix_cellules[-1]
            liste = cellule.cellules_communes(liste_cellules)
            for nom in liste:
                element = Cellule.chercher_cellule_par_nom(liste_cellules, nom)
                self.choix_cellules.remove(element)  
            current_text = condition_text.get("1.0", tk.END).strip().splitlines()
            current_text.remove(f"{liste}")
            condition_text.delete("1.0", tk.END)
            condition_text.insert(tk.END, "\n".join(current_text) + "\n")
        else:
            messagebox.showwarning("Avertissement", "Aucune cellule à supprimer.")
           
    #crée l'interface pour choisir les cellules
    def choix_cellule(self, root, liste_cellules):
        
        condition_label = tk.Label(root, text="Choix des cellules :")
        condition_label.pack()

        condition_text = tk.Text(root, height=15, width=80)
        condition_text.pack()
        
        cellule_label = tk.Label(root, text="cellule")
        cellule_label.pack()
        
        cellule_combobox = ttk.Combobox(root, values=[cell.nom for cell in liste_cellules if cell.nom.startswith('LHC')], width=15)
        cellule_combobox.pack()

        ajouter_button = tk.Button(root, text="Ajouter Cellule", command=lambda : self.ajouter_cellule(cellule_combobox, condition_text, liste_cellules))
        ajouter_button.pack() 
        
        supprimer_button = tk.Button(root, text="Supprimer Dernière Cellule", command=lambda: self.supprimer_derniere_cellule(liste_cellules, condition_text))
        supprimer_button.pack()
