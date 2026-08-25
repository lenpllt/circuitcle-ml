import json #type de fichier qui va sauvegarder les différentes procédures créées
import tkinter as tk
from tkinter import messagebox

#classe EtapesChoisies qui permet d'écrire des procédures et de les sauveggarder jusqu'à suppression
class EtapesChoisies:
    
    listes_etapes = {} #liste les procédures que l'on a écrit
    compteur = 1
    donnees_etapes = {} #stocke les données sur les étapes choisies (noms de la cellule et de l'étape)
    fichier_sauvegarde = "etapes_choisies_LHT.json" #fichier dans lequel les procédures écrites sont sauvegardées
    
    #caractéristiques de la classe
    def __init__(self):
        self.etapes_choisies = [] #stocke les étpaes choisies de la procédure en cours
        self.nouvelle_entree = []
        #self.vider_dictionnaires()
        self.charger_etapes() #charge les étapes sauvegardées dans le fichier json
        
    #supprime toutes les procédures écrites, à n'utiliser qu'en cas de besoin
    def vider_dictionnaires(self):
        EtapesChoisies.donnees_etapes.clear()
        EtapesChoisies.listes_etapes.clear()
        self.sauvegarder_etapes()  #sauvegarde les changements pour que le fichier soit bien mis à jour
        print("Les dictionnaires ont été vidés.")

     #sauvegarde les procédures dans un fichier json pour qu'elles ne soient pas supprimées quand on ferme la fenêtre ni même python   
    def sauvegarder_etapes(self):
        with open(EtapesChoisies.fichier_sauvegarde, 'w') as fichier:
            json.dump({
                'donnees_etapes': EtapesChoisies.donnees_etapes,
                'compteur': EtapesChoisies.compteur  
            }, fichier)
    
    #charge les données contenues dans le fichier json 
    def charger_etapes(self):
        try:
            with open(EtapesChoisies.fichier_sauvegarde, 'r') as fichier:
                data = json.load(fichier)
                EtapesChoisies.donnees_etapes = data.get('donnees_etapes', {})
                EtapesChoisies.compteur = data.get('compteur', 1)  
        except FileNotFoundError:
            EtapesChoisies.donnees_etapes = {}
            EtapesChoisies.compteur = 1
        
    #associe dans un dictionnaire un nom à chaque étape possible, pour pouvoir retrouver une étape via son nom
    def etapes_possibles(self, cellule):
        etapes = {}
        partie_mobile = cellule.partie_mobile
        smalt = cellule.smalt
        porte = cellule.porte
        serrures = cellule.serrures
        transformateur = cellule.transformateur
        coffret = cellule.coffret
        eclisse = cellule.eclisse
        if partie_mobile:
            etapes["insertion_extraction_manivelle partie mobile"] = lambda pm=partie_mobile: pm.insertion_extraction_manivelle()
            etapes["embrochage_debrochage partie mobile"] = lambda pm=partie_mobile, s=smalt, c=cellule: pm.embrochage_debrochage(s, c)
            etapes["verrouillage_deverrouillage_tiroir"] = lambda pm=partie_mobile: pm.verrouillage_deverrouillage_tiroir()
        if eclisse:
            etapes["suppression_ajoute eclisse"] = lambda e=eclisse, c=cellule: e.ajout_suppression_eclisse(c)
            etapes["verrouillage_deverrouillage_tiroir"] = lambda e=eclisse: e.verrouillage_deverrouillage_tiroir()
        if smalt:    
            etapes["ouverture_fermeture_SMALT"] = lambda s=smalt, pm=partie_mobile, c=cellule: s.ouverture_fermeture_SMALT(pm, c)
            etapes["verrouillage_deverrouillage_SMALT"] = lambda s=smalt: s.verrouillage_deverrouillage_SMALT()
        if porte != []:
            etapes["ouverture_panneau"] = lambda p=porte, s=smalt: p.ouverture_panneau(s)
        if coffret != []:
            etapes["ouverture_fermeture_coffret"] = lambda c=coffret: c.ouverture_fermeture_coffret()
        if transformateur != []:
            etapes["consignation_deconsignation_transformateur"] = lambda t=transformateur: t.consignation_deconsignation_transformateur()
        for serrure in serrures:
            etapes[f"clefs_serrure_mere {serrure.nom}"] = lambda sr=serrure, c=cellule: sr.clefs_serrure_mere(c)
        return etapes
    
    #fonction qui réinitialise les étapes de chaque procédure, pour repartir des états initiaux
    def reinitialiser_etapes(self, liste_cellules):
        #print(EtapesChoisies.listes_etapes)
        #print(EtapesChoisies.compteur)
        for nom_liste, etapes in EtapesChoisies.donnees_etapes.items():
            #print(nom_liste)
            if nom_liste not in EtapesChoisies.listes_etapes:
                EtapesChoisies.listes_etapes[nom_liste] = []  
            for i, (etape_nom, cellule_nom) in enumerate(etapes):
                #print(EtapesChoisies.donnees_etapes)
                cellule_obj = next((c for c in liste_cellules if c.nom == cellule_nom), None)
                if cellule_obj:
                    nouvelles_etapes = self.etapes_possibles(cellule_obj)
                    
                    if etape_nom in nouvelles_etapes:
                        if EtapesChoisies.listes_etapes[nom_liste] == []:
                            EtapesChoisies.listes_etapes[nom_liste].append(nouvelles_etapes[etape_nom])
                        else:
                            #print(i)
                            EtapesChoisies.listes_etapes[nom_liste] = []
                            for (etape_nom, cellule_nom) in etapes:
                                cellule_obj = next((c for c in liste_cellules if c.nom == cellule_nom), None)
                                if cellule_obj:
                                    nouvelles_etapes = self.etapes_possibles(cellule_obj)
                                    if etape_nom in nouvelles_etapes:
                                        EtapesChoisies.listes_etapes[nom_liste].append(nouvelles_etapes[etape_nom])
        #print(EtapesChoisies.listes_etapes)
                            
     #liste les cellules qui sont possibles à choisir                   
    def cellules_possibles(self, liste_cellules):
        liste = []
        for cellule in liste_cellules:
            if self.etapes_possibles(cellule) != {}:
                liste.append(cellule.nom)
        return liste
    
    #met à jour les choix possibles d'étapes en fonction de la cellule sélectionnée
    def update_etapes(self, cellule_var, etape_var, etape_menu, liste_cellules):
        cellule_nom = cellule_var.get()
        cellule_obj = next((c for c in liste_cellules if c.nom == cellule_nom), None)
        
        #Efface les anciennes options dans le menu déroulant des étapes
        etape_menu["menu"].delete(0, "end")
        
        if cellule_obj:
            etapes = self.etapes_possibles(cellule_obj)
            for nom, etape in etapes.items():
                etape_menu["menu"].add_command(label=nom, command=lambda e=nom: etape_var.set(e))
        else:
            etape_var.set("")  
                
     #fonction qui ajoute une étape à la liste des étapes                   
    def ajouter_etape(self, cellule_var, etape_var, label_etapes_choisies, liste_cellules):
        cellule = cellule_var.get()
        nom_etape = etape_var.get()
        cellule_obj = next((c for c in liste_cellules if c.nom == f"{cellule}"), None)
        if cellule_obj:
            etapes = self.etapes_possibles(cellule_obj)
            for nom, etape in etapes.items():
                if nom == nom_etape:
                    self.etapes_choisies.append(etape) #ajoute l'étape à la liste
                    liste_key = f"liste {EtapesChoisies.compteur}"
                    if liste_key not in EtapesChoisies.donnees_etapes:
                        EtapesChoisies.donnees_etapes[liste_key] = []
                    EtapesChoisies.donnees_etapes[liste_key].append((nom_etape, cellule))
                    self.sauvegarder_etapes()
                    
        if nom_etape:
            self.nouvelle_entree.append(nom_etape + " " + f"{cellule}")
            label_etapes_choisies.config(text="Étapes choisies :\n" + "\n".join(self.nouvelle_entree))
        else:
            messagebox.showwarning("Avertissement", "Veuillez choisir une étape.")

           
    #créé l'interface et l'affichage avec les différents boutons
    def choix_etapes(self, fenetre_ouverte, liste_cellules):
        
        font_grand = ("Arial", 11)
       
        cellule_var = tk.StringVar(fenetre_ouverte, value="")
        etape_var = tk.StringVar(fenetre_ouverte, value="")

        cellule_label = tk.Label(fenetre_ouverte, text="Choisir une cellule :", font=font_grand)
        cellule_label.pack()

        # Liste déroulante pour choisir une cellule
        cellule_menu = tk.OptionMenu(fenetre_ouverte, cellule_var, *self.cellules_possibles(liste_cellules))
        cellule_menu.config(font=font_grand)
        cellule_menu.pack()

        etape_label = tk.Label(fenetre_ouverte, text="Choisir une étape :", font=font_grand)
        etape_label.pack()

        # Liste déroulante pour choisir une étape
        etape_menu = tk.OptionMenu(fenetre_ouverte, etape_var, "")
        etape_menu.config(font=font_grand)
        etape_menu.pack()
        
        #affiche les étapes choisies
        label_etapes_choisies = tk.Label(fenetre_ouverte, text="Étapes choisies :", font=font_grand)
        label_etapes_choisies.pack()
        
        ajouter_button = tk.Button(fenetre_ouverte, text="Ajouter Étape", command=lambda: self.ajouter_etape(cellule_var, etape_var, label_etapes_choisies, liste_cellules), font=font_grand)
        ajouter_button.pack()
        
        supprimer_button = tk.Button(fenetre_ouverte, text="Supprimer la dernière étape", command=lambda: self.supprimer_derniere_etape(label_etapes_choisies), font=font_grand)
        supprimer_button.pack()

        bouton_finaliser_etapes = tk.Button(fenetre_ouverte, text="Finaliser liste", command= lambda: self.finaliser_liste(label_etapes_choisies))
        bouton_finaliser_etapes.pack()
        
        bouton_supprimer_liste = tk.Button(fenetre_ouverte, text="Supprimer la dernière liste", command=self.supprimer_derniere_liste, font=font_grand)
        bouton_supprimer_liste.pack()

        # Mettre à jour les étapes disponibles lorsque la cellule est sélectionnée
        cellule_var.trace("w", lambda *args: self.update_etapes(cellule_var, etape_var, etape_menu, liste_cellules))

        
    #permet de supprimer la dernière étape choisie
    def supprimer_derniere_etape(self, label_etapes_choisies):
        if self.nouvelle_entree:
            self.nouvelle_entree.pop()  
            self.etapes_choisies.pop()  
            label_etapes_choisies.config(text="Étapes choisies :\n" + "\n".join(self.nouvelle_entree))
        else:
            messagebox.showwarning("Avertissement", "Aucune étape à supprimer.")

    #finalise la procédure d'étapes écrite
    def finaliser_liste(self, label_etapes_choisies):
        if self.etapes_choisies:
            nom_liste = f"liste {EtapesChoisies.compteur}"  # nom liste 1, liste 2...
            EtapesChoisies.listes_etapes[nom_liste] = self.etapes_choisies.copy()
            self.sauvegarder_etapes()
            
            print(f"\nListe finalisée : {nom_liste}")
            for (etape_nom, cellule_nom) in EtapesChoisies.donnees_etapes[nom_liste]:
                print(f"- {etape_nom} {cellule_nom}")
            
            EtapesChoisies.compteur += 1
            self.sauvegarder_etapes()
            self.etapes_choisies.clear()
            self.nouvelle_entree.clear()
            
            label_etapes_choisies.config(text="Aucune etape choisie")
            
            if hasattr(self, 'update_menu_callback'):
                self.update_menu_callback()
    
    #supprime la dernière procédure créée
    def supprimer_derniere_liste(self):
        if EtapesChoisies.donnees_etapes and EtapesChoisies.listes_etapes:
            derniere_clef = max(EtapesChoisies.donnees_etapes.keys(), key=lambda x: int(x.split()[1]))
            del EtapesChoisies.donnees_etapes[derniere_clef]
            del EtapesChoisies.listes_etapes[derniere_clef]
            EtapesChoisies.compteur -= 1
            self.sauvegarder_etapes()
            print(f"La dernière liste ({derniere_clef}) a été supprimée.")
        else:
            print("Aucune liste à supprimer.")
                
    def set_update_menu_callback(self, callback):
        self.update_menu_callback = callback
