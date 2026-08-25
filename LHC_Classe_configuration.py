#importation des modules necessaires
import tkinter as tk #cree des fenetres pour l'interface
from tkinter import messagebox
from LHC_Classe_cellule import Cellule
from LHC_Classe_dessin import Dessin
from LHC_Classe_initialisation_cellules import InitialisationCellules
import numpy as np #pratique pour l'utilisation de matrices
import random #utile pour l'aleatoire

#création de la classe Configuration, qui regroupe des fonctions indépendantes qui configurent l'exécution des étapes et l'affichage du tableau
class Configuration:
    #caractéristiques de la classe
    def __init__(self):
        #variables permettant de gerer correctement les mises en pause
        self.en_marche = True #variable qui se met à jour quand on est en pause
        self.test_en_cours = None #variable qui stocke le test actuel pour le reprendre quand on finit la mise en pause

    #cree la feuille de dessin 
    def creation_canvas(self, frame, largeur, hauteur):
        #scrollregion delimite la zone du dessin que l'on pourra voir en se deplacant
        canvas = tk.Canvas(frame, width=largeur, height=hauteur, bg='white', scrollregion=(-largeur, -hauteur, largeur * 2, hauteur * 3))
        
        # Scrollbar verticale
        vbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
        vbar.config(command=canvas.yview)
        canvas.config(yscrollcommand=vbar.set)

        # Scrollbar horizontale
        hbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        hbar.config(command=canvas.xview)
        canvas.config(xscrollcommand=hbar.set)

        # Placement des widgets
        canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")

        # Configurer le frame pour qu'il étende les widgets
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        return canvas

    #fonction permettant de mettre en pause/play : cette fonction s'execute quand on appuie sur la touche espace
    def mettre_en_pause(self, root, dessin, conditions, a, b, liste_cellules, liste, etapes, index, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, nombre_execution):
        global en_marche, test_en_cours
        en_marche = not en_marche #permet de mettre en pause si on est en marche et inversement
        if not en_marche:
            print("Simulation mise en pause.")
        else:
            print("Simulation reprise.")
            if test_en_cours == "classique":
                dessin.canvas.after(100, self.executer_etape, root, dessin, conditions, a, b, liste_cellules, liste, etapes, index, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere)
            elif test_en_cours == "aleatoire":
                dessin.canvas.after(100, self.etape_aleatoire, root, dessin, conditions, a, b, liste_cellules, liste, etapes, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, nombre_execution)
            
    #configure la fenetre pour qu'elle detecte la mise en pause via l'appuie de la touche espace                        
    def configurer_reactions_clavier(self, root, dessin, conditions, a, b, liste_cellules, liste, etapes, index, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, nombre_execution):
        root.focus_set() 
        root.bind("<space>", lambda event: self.mettre_en_pause(root, dessin, conditions, a, b, liste_cellules, liste, etapes, index, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, nombre_execution)) 
    
    def dessin_final(self, dessin, a, b, taille, cellule, x_coin, y_coin, taille_serrure_mere, largeur, hauteur):
        partie_mobile = cellule.partie_mobile
        smalt = cellule.smalt
        serrures = cellule.serrures
        porte = cellule.porte
        transformateur = cellule.transformateur
        matrice = cellule.matrice
        coffret = cellule.coffret

        zoom = dessin.facteur_zoom
        taille = round(taille * zoom)
        a = round(a * zoom)
        b = round(b * zoom)
        taille_serrure_mere = round(taille_serrure_mere * zoom)

        for i in range(len(matrice)):
            for j in range(len(matrice[0])):
                x = j * taille + x_coin
                y = i * taille + y_coin

                element = matrice[i][j]

                if element == 'disjoncteur':
                    dessin.dessiner_disjoncteur(x + a + taille / 2, y + b, taille)
                    if partie_mobile and hasattr(partie_mobile, 'representation'):
                        dessin.dessiner_elements_disjoncteur(partie_mobile, i, j, partie_mobile.representation, taille, a, b, x_coin, y_coin)

                if element == 'tiroir':
                    dessin.dessiner_tiroir(x + a + taille / 2, y + b, taille)
                    if partie_mobile and hasattr(partie_mobile, 'representation'):
                        dessin.dessiner_elements_disjoncteur(partie_mobile, i, j, partie_mobile.representation, taille, a, b, x_coin, y_coin)

                if element == 'contacteur':
                    dessin.dessiner_contacteur(x + a + taille / 2, y + b, taille)
                    if partie_mobile and hasattr(partie_mobile, 'representation'):
                        dessin.dessiner_elements_disjoncteur(partie_mobile, i, j, partie_mobile.representation, taille, a, b, x_coin, y_coin)

                if element == 'transformateur':
                    dessin.dessiner_transformateur(transformateur, x + a + taille / 2, y + b, taille)

                if element == 'coffret':
                    if coffret and hasattr(coffret, 'clefs'):
                        dessin.dessiner_coffret(coffret, x + a, y + b, taille)
                        
                if element == 'fil_h':
                    dessin.dessiner_fil_h(x + a, y + b + taille / 2, taille)

                if element == 'fil_v':
                    dessin.dessiner_fil_v(x + a + taille / 2, y + b, taille)

                if element == 'smalt':
                    dessin.dessiner_terre(x + a, y + b + taille / 2, taille)
                    if smalt and hasattr(smalt, 'representation'):
                        dessin.dessiner_elements_smalt(smalt, i, j, smalt.representation, taille, a, b, x_coin, y_coin)

                if element == 'fil_h_dep':
                    dessin.dessiner_fil_h_dep(x + a, y + b, taille)

                if element == 'fil_dep_smalt':
                    dessin.dessiner_fil_dep_smalt(x + a, y + b, taille)

                if element == 'angle_haut_gauche':
                    dessin.dessiner_angle_haut_gauche(x + a, y + b, taille)

                if element == 'angle_haut_droit':
                    dessin.dessiner_angle_haut_droit(x + a, y + b, taille)

                if element == 'angle_bas_droit':
                    dessin.dessiner_angle_bas_droit(x + a, y + b, taille)

                if element == 'angle_bas_gauche':
                    dessin.dessiner_angle_bas_gauche(x + a, y + b, taille)

                if element == 'croisement':
                    dessin.dessiner_croisement(x + a, y + b, taille)

                if element == 'source':
                    dessin.dessiner_source(x + a, y + b, taille)

                if element == 'fusible':
                    dessin.dessiner_fusible(x + a + taille / 2, y + b, taille)

        dessin.dessiner_rectangle(x_coin + a, y_coin - taille / 10 + b, cellule.largeur * taille + x_coin + a, cellule.longueur * taille + y_coin + b)

        if 'LHC' in cellule.nom:
            dessin.dessiner_rectangle(x_coin + a, y_coin - taille_serrure_mere + b, cellule.largeur * taille + x_coin + a, y_coin - taille / 10 + b)
            dessin.canvas.create_text(x_coin + taille + a, y_coin - taille / 4 + b, text=cellule.nom, font=("Arial", taille_serrure_mere // 14))
            position = 0
            for serrure in serrures:
                dessin.dessiner_serrure_mere(x_coin + a, y_coin - taille_serrure_mere + b, cellule, matrice, serrure, len(cellule.serrures), position, taille, taille_serrure_mere)
                position += 1
        else:
            dessin.dessiner_rectangle(x_coin + a, cellule.longueur * taille + y_coin + b, cellule.largeur * taille + x_coin + a, cellule.longueur * taille + y_coin + b + taille_serrure_mere)
            dessin.canvas.create_text(x_coin + a + taille, cellule.longueur * taille + y_coin + b + taille_serrure_mere - taille / 5, text=cellule.nom, font=("Arial", taille_serrure_mere // 14))
            position = 0
            for serrure in serrures:
                dessin.dessiner_serrure_mere(x_coin + a, cellule.longueur * taille + y_coin + b, cellule, matrice, serrure, len(cellule.serrures), position, taille, taille_serrure_mere)
                position += 1

        if porte:
            if cellule.nom.startswith('LHC'):
                dessin.canvas.create_rectangle(x_coin + taille / 5 + a, cellule.longueur * taille + y_coin - 3 * taille / 4 + b, cellule.largeur * taille + x_coin - taille / 5 + a, cellule.longueur * taille + y_coin - taille / 10 + b, dash=(4, 2))
                if porte.clefs:
                    dessin.dessiner_carre(x_coin + taille / 5 + a + taille / 4, cellule.longueur * taille + y_coin - taille / 2 + b, porte.clefs[0], taille)
                    dessin.canvas.create_text(x_coin + taille / 5 + a + taille / 4, cellule.longueur * taille + y_coin - taille / 2 + b, text=porte.clefs[0].nom, font=("Arial", taille // 5))
            else:
                dessin.canvas.create_rectangle(x_coin + taille / 5 + a, y_coin + taille / 10 + b, cellule.largeur * taille + x_coin - taille / 5 + a, y_coin + 2 * taille / 3 + b, dash=(4, 2))
                if porte.clefs:
                    dessin.dessiner_carre(x_coin + taille / 5 + a + taille / 4, y_coin + taille / 10 + b + taille / 3, porte.clefs[0], taille)
                    dessin.canvas.create_text(x_coin + taille / 5 + a + taille / 4, y_coin + taille / 10 + b + taille / 3, text=porte.clefs[0].nom, font=("Arial", taille // 5))

        liste = 2 * taille
        dessin.canvas.create_text(largeur / 15, cellule.longueur * taille + y_coin + taille / 5, text='clefs libres : ', font=("Arial", taille // 4))
        for clefs in cellule.smalt.clefs_libres if hasattr(cellule.smalt, 'clefs_libres') else []:
            dessin.canvas.create_text(largeur / 15 + liste, cellule.longueur * taille + y_coin + taille / 5, text=clefs + ',', font=("Arial", taille // 4))
            liste += taille

    '''
    #fonction qui dessine une cellule en utilisant les fonctions de la classe Dessin       
    def dessin_final(self, dessin, a, b, taille, cellule, x_coin, y_coin, taille_serrure_mere, largeur, hauteur):
        #on recupere tous les éléments électriques d'une cellule
        partie_mobile = cellule.partie_mobile
        smalt = cellule.smalt
        serrures = cellule.serrures
        porte = cellule.porte
        transformateur = cellule.transformateur
        matrice = cellule.matrice
        coffret = cellule.coffret
        
        zoom = dessin.facteur_zoom #échelle de zoom
        taille = round(taille*zoom)
        a = round(a*zoom)
        b = round(b*zoom)
        taille_serrure_mere =round(taille_serrure_mere*zoom)
        for i in range(len(matrice)):
                for j in range(len(matrice[0])):
                    x = j*taille + x_coin 
                    y = i*taille + y_coin
                    if matrice[i][j] == 'disjoncteur':
                        dessin.dessiner_disjoncteur(x + a + taille/2, y + b , taille)
                        if partie_mobile:
                            dessin.dessiner_elements_disjoncteur(partie_mobile, i, j, partie_mobile.representation, taille, a, b, x_coin, y_coin)
                                            
                    if matrice[i][j] == 'tiroir':
                        dessin.dessiner_tiroir(x + a + taille/2, y + b , taille)
                        if partie_mobile:
                            dessin.dessiner_elements_disjoncteur(partie_mobile, i, j, partie_mobile.representation, taille, a, b, x_coin, y_coin)
                    
                    if matrice[i][j] == 'contacteur':
                        dessin.dessiner_contacteur(x + a + taille/2, y + b , taille)
                        if partie_mobile:
                            dessin.dessiner_elements_disjoncteur(partie_mobile, i, j, partie_mobile.representation, taille, a, b, x_coin, y_coin)
                    
                    if matrice[i][j] == 'transformateur':
                        dessin.dessiner_transformateur(transformateur, x + a + taille/2, y + b, taille)
                        
                    if matrice[i][j] == 'coffret':
                        dessin.dessiner_coffret(coffret, x + a, y + b, taille)
                            
                    if matrice[i][j] == 'fil_h':    
                        dessin.dessiner_fil_h(x + a, y + b + taille/2 , taille)
                            
                    if matrice[i][j] == 'fil_v':
                        dessin.dessiner_fil_v(x + a + taille/2, y + b , taille)

                    if matrice[i][j] == 'smalt':  
                        dessin.dessiner_terre(x + a, y + b + taille/2 , taille)
                        dessin.dessiner_elements_smalt( smalt, i, j, smalt.representation, taille, a, b, x_coin, y_coin)
                    
                    if matrice[i][j] == 'fil_h_dep':
                        dessin.dessiner_fil_h_dep(x + a, y + b , taille)
                        
                    if matrice[i][j] == 'fil_dep_smalt':
                        dessin.dessiner_fil_dep_smalt(x + a, y + b , taille)
                        
                    if matrice[i][j] == 'angle_haut_gauche':
                        dessin.dessiner_angle_haut_gauche(x + a, y + b , taille)
                    
                    if matrice[i][j] == 'angle_haut_droit':
                        dessin.dessiner_angle_haut_droit(x + a, y + b , taille)
                    
                    if matrice[i][j] == 'angle_bas_droit':
                        dessin.dessiner_angle_bas_droit(x + a, y + b , taille)
                    
                    if matrice[i][j] == 'angle_bas_gauche':
                        dessin.dessiner_angle_bas_gauche(x + a, y + b , taille)
                    
                    if matrice[i][j] == 'croisement':
                        dessin.dessiner_croisement(x + a, y + b , taille)
                        
                    if matrice[i][j] == 'source':
                        dessin.dessiner_source(x + a, y + b, taille)
                        
                    if matrice[i][j] == 'fusible':
                        dessin.dessiner_fusible(x + a + taille/2, y + b, taille)
        #Rectangle de la cellule      
        dessin.dessiner_rectangle( x_coin + a, y_coin -taille/10 + b , cellule.largeur*taille+ x_coin + a, cellule.longueur*taille + y_coin + b )
        #création de la serrure mere
        if 'LHC' in cellule.nom:
            dessin.dessiner_rectangle( x_coin + a, y_coin - taille_serrure_mere + b, cellule.largeur*taille + x_coin + a, y_coin -taille/10 + b )
            dessin.canvas.create_text(x_coin + taille + a, y_coin -taille/4 + b, text=cellule.nom, font=("Arial", taille_serrure_mere//14))
            position = 0
            for serrure in serrures: #ajout de toutes les serrures mères pour chaque cellule
                dessin.dessiner_serrure_mere( x_coin + a,  y_coin - taille_serrure_mere + b, cellule, matrice, serrure, len(cellule.serrures), position, taille, taille_serrure_mere)
                position += 1

        #si la cellule n'est pas du tableau LHC, la serrure ce mettra en dessous de la cellule et pas au dessus        
        else:
            dessin.dessiner_rectangle(x_coin + a, cellule.longueur*taille + y_coin + b, cellule.largeur*taille + x_coin + a, cellule.longueur*taille + y_coin + b + taille_serrure_mere)
            dessin.canvas.create_text(x_coin + a + taille, cellule.longueur*taille + y_coin + b + taille_serrure_mere - taille/5 , text=cellule.nom, font=("Arial", taille_serrure_mere//14))
            position = 0
            for serrure in serrures:
                dessin.dessiner_serrure_mere( x_coin + a, cellule.longueur*taille + y_coin + b, cellule, matrice, serrure, len(cellule.serrures), position, taille, taille_serrure_mere)
                position += 1
        if porte != []:
            if cellule.nom.startswith('LHC'):
                dessin.canvas.create_rectangle(x_coin + taille/5 + a, cellule.longueur*taille  + y_coin - 3*taille/4 + b , cellule.largeur*taille +x_coin - taille/5 + a,cellule.longueur*taille  + y_coin -taille/10 + b, dash=(4,2))
                if porte.clefs != []:
                    dessin.dessiner_carre( x_coin + taille/5 + a+ taille/4, cellule.longueur*taille  + y_coin - taille/2 + b, porte.clefs[0], taille)
                    dessin.canvas.create_text(x_coin + taille/5 + a+ taille/4, cellule.longueur*taille  + y_coin - taille/2 + b, text=porte.clefs[0].nom, font=("Arial", taille//5))
            else:
                dessin.canvas.create_rectangle(x_coin + taille/5 + a, y_coin + taille/10 + b , cellule.largeur*taille +x_coin - taille/5 + a,y_coin + 2*taille/3 + b, dash=(4,2))
                if porte.clefs != []:
                    dessin.dessiner_carre( x_coin + taille/5 + a+ taille/4, y_coin + taille/10 + b + taille/3, porte.clefs[0], taille)
                    dessin.canvas.create_text(x_coin + taille/5 + a+ taille/4, y_coin + taille/10 + b + taille/3, text=porte.clefs[0].nom, font=("Arial", taille//5))
        
        liste = 2*taille
        dessin.canvas.create_text(largeur/15,cellule.longueur*taille + y_coin + taille/5, text='clefs libres : ', font=("Arial", taille//4))
        for clefs in cellule.smalt.clefs_libres:
            dessin.canvas.create_text(largeur/15 + liste, cellule.longueur*taille + y_coin + taille/5, text=clefs +',', font=("Arial", taille//4))
            liste+= taille
    '''

    #construit un dictionnaire mettant pour chaque cellule les mises a la terre nécessaires avant d'ouvrir la porte
    def construire_dictionnaire_smalt(self, liste_cellules):
        liste_smalt = {}
        
        for cellule1 in liste_cellules:
            if cellule1.voisine != 'False' and cellule1.voisine != 'BC':
                for cellule in liste_cellules:
                    if cellule.nom == cellule1.voisine:
                        voisine = cellule
                        liste_smalt[cellule1.nom] = [cellule1.smalt, voisine.smalt]
                if cellule1.nom not in liste_smalt.keys():
                    liste_smalt[cellule1.nom] = [cellule1.smalt]
            else:
                liste_smalt[cellule1.nom] = [cellule1.smalt]
        
        return liste_smalt
    
    #dessin final du tableau : une cellule LHC est positionnée en y=0 et une cellule d'un autre tableau commence à la fin de la longueur de la cellule LHC à laquelle elle est reliée
    def dessin_tableau(self, dessin, liste_cellules, a, b, taille, x_coin,y_coin, taille_serrure_mere, largeur, hauteur):
        for i, cellule in enumerate(liste_cellules):
            if i == 0:
                position_premiere_cellule = cellule.position_x
            if 'LHC' in cellule.nom:
                    self.dessin_final(dessin, (cellule.position_x-position_premiere_cellule)*a, 0, taille, cellule, x_coin,y_coin, taille_serrure_mere, largeur, hauteur)  
            else:
                voisine = cellule.chercher_cellule_par_nom(liste_cellules, cellule.voisine)
                self.dessin_final(dessin, (cellule.position_x-position_premiere_cellule)*a, len(voisine.matrice)*b, taille, cellule, x_coin, y_coin, taille_serrure_mere, largeur, hauteur)

    #regarde le dessin de la plus bas en hauteur, pour savoir le nombre de lignes que doit avoir la matrice finale
    def max_hauteur(self, liste_cellules):
        max_hauteur = 0
        for cellule in liste_cellules:
            if cellule.voisine == 'non':
                max_hauteur = max(len(cellule.matrice), max_hauteur)
            else:
                voisine = cellule.chercher_cellule_par_nom(liste_cellules, cellule.voisine)
                max_hauteur = max((len(cellule.matrice)+len(voisine.matrice)), max_hauteur)
        return max_hauteur + 10

    #cree la matrice finale (unique) permettant de verifier l'existence d'un chemin entre deux elements electriques        
    def matrice_unique(self, liste_cellules):
        matrice_finale = np.zeros((self.max_hauteur(liste_cellules), 5 * (len(liste_cellules) + 3)), dtype=object)

        for cellule in liste_cellules:
            matrice = cellule.matrice
            if matrice is None:
                continue  # sécurité

            ligne, colonne = matrice.shape

            x_offset = 5 * cellule.position_x

            if cellule.nom.startswith('LHC'):
                y_offset = 0
            else:
                voisine = Cellule.chercher_cellule_par_nom(liste_cellules, cellule.voisine)
                y_offset = len(voisine.matrice) if voisine and voisine.matrice is not None else 0

            # Placement dans la matrice finale
            matrice_finale[y_offset:y_offset + ligne, x_offset:x_offset + colonne] = matrice

            # Calcul des positions absolues
            positions_absolues = [
                (y_offset + i, x_offset + j)
                for i in range(ligne)
                for j in range(colonne)
                if matrice[i][j] != 0
            ]
            cellule.positions_absolues = positions_absolues

        return matrice_finale

    #fonction qui execute des etapes choisies    
    def executer_etape(self, root, dessin, conditions, a, b, liste_cellules, liste, etapes, index, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, palier):
        nombre_execution = 1
        self.configurer_reactions_clavier(root, dessin, conditions, a, b, liste_cellules, liste, etapes, index, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, nombre_execution) 
        global test_en_cours
        test_en_cours = "classique"  
        conditions.tableau = self.matrice_unique(liste_cellules)
        liste_smalt = self.construire_dictionnaire_smalt(liste_cellules)
        liste_porte = [cellule.porte for cellule in liste if cellule.porte.clefs != []]
        global en_marche
        
        if not en_marche: #si on est en pause, on stoppe l'exécution de la fonction
            return
        
        if index < len(etapes):
            
            print(f"Index actuel : {index} / {len(etapes)}")

            dessin.canvas.delete("all")
                
            #Exécute l'étape actuelle
            if en_marche:
                etapes[index]()
            
            self.dessin_tableau(dessin, liste, a, b, taille, 15, y_coin, taille_serrure_mere, largeur, hauteur)
            
            message = conditions.verifier_conditions(liste_smalt, liste_porte, liste_cellules, palier) #vérifie si il y a des conditions d'arrêt
            #verifier_conditions renvoit soit False, soit un message
            
            #si message n'est pas False, ca veut dire qu'il y a un un danger donc on arrête l'exécution et on affiche le message 
            if message != False:
                dessin.canvas.create_text(largeur / 2, hauteur / 2, text=message, fill='red', font=("Arial", 20))
                return
            
            #passe à l'étape suivante après 500 ms si il n'y a pas de danger et qu'on n'est pas en pause
            dessin.canvas.after(500, self.executer_etape, root, dessin, conditions, a, b, liste_cellules, liste, etapes, index + 1, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere)
            
        else:
            dessin.canvas.delete("all")
            self.dessin_tableau(dessin, liste, a, b, taille, 15, y_coin, taille_serrure_mere, largeur, hauteur)
            message = conditions.verifier_conditions(liste_smalt, liste_porte, liste_cellules)
            if message == False:
                message = "Fin de l'execution des etapes"
                print(f"Fin de l'execution des etapes")

            dessin.canvas.create_text(largeur / 2, hauteur / 2, text=message, fill='red', font=("Arial", 20)) 

    #fonction qui permet, quand on clique sur le bouton "test_classique", de choisir une des procédures écrites avant de l'exécuter
    def executer_etape_avec_selection(self, root, dessin, conditions, a, b, liste_cellules, liste, etapes, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere):
        selection_window = tk.Toplevel(root)
        selection_window.title("Selectionner une etape")
        
        liste_etapes_var = tk.StringVar(selection_window, value="")
        
        #on vérifie si on a des procédure et on affiche la liste
        if not etapes.listes_etapes:
            messagebox.showerror("Erreur", "La liste des étapes est vide.")
        else:
            liste_etapes_menu = tk.OptionMenu(selection_window, liste_etapes_var, *etapes.listes_etapes.keys())
            liste_etapes_menu.pack(pady=10)
        
        def valider_selection():
            selected_step = liste_etapes_var.get()
            if selected_step:
                etapes_choisies = etapes.listes_etapes.get(selected_step, [])
                self.executer_etape(root, dessin, conditions, a, b, liste_cellules, liste, etapes_choisies, 0, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere)
                selection_window.destroy()
            else:
                tk.messagebox.showwarning("selection manquante", "veuillez selectionner une etape avant de continuer")
                
        bouton_valider = tk.Button(selection_window, text="Valider", command=valider_selection)
        bouton_valider.pack(pady=10)
        
        bouton_annuler = tk.Button(selection_window, text="Annuler", command=selection_window.destroy)
        bouton_annuler.pack(pady=10)
        
    #execute des etapes aleatoirement      
    def executer_etape2(self, root, dessin, conditions, a, b, liste_cellules, liste, etapes, index, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, nombre_execution, callback=None):
        
        self.configurer_reactions_clavier(root, dessin, conditions, a, b, liste_cellules, liste, etapes, index, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, nombre_execution) 
        conditions.tableau = self.matrice_unique(liste_cellules)
        liste_smalt = self.construire_dictionnaire_smalt(liste_cellules)
        liste_porte = [cellule.porte for cellule in liste if cellule.porte.clefs != []]
        message = conditions.verifier_conditions(liste_smalt, liste_porte, liste_cellules)
        global en_marche
        
        if not en_marche:
            return
        if index < len(etapes):
            dessin.canvas.delete("all")
            if en_marche:
                etapes[index]()
            self.dessin_tableau(dessin, liste, a, b, taille, 15, y_coin, taille_serrure_mere, largeur, hauteur)
            message = conditions.verifier_conditions(liste_smalt, liste_porte, liste_cellules)
            #si message n'est pas False, ca veut dire qu'il y a un un danger donc on arrête l'exécution et on affiche le message
            if message != False:
                dessin.canvas.create_text(largeur / 2, hauteur / 2, text=message, fill='red', font=("Arial", 20))
                return
            
            dessin.canvas.after(1, self.executer_etape2, root, dessin, conditions, a, b, liste_cellules, liste, etapes, index + 1, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, nombre_execution, callback)
        else:
            dessin.canvas.delete("all")
            self.dessin_tableau(dessin, liste, a, b, taille, 15, y_coin, taille_serrure_mere, largeur, hauteur)
            conditions.tableau = self.matrice_unique(liste_cellules)
            message = conditions.verifier_conditions(liste_smalt, liste_porte, liste_cellules)
            if message != False:
                dessin.canvas.create_text(largeur / 2, hauteur / 2, text=message, fill='red', font=("Arial", 20))
                return
            if callback is not None:
                callback()  #callback de fin appelé quand on a réalisé toute les exécutions
  
    #mélange les etapes et appelle la fonction "executer_etape2" (qui exécute les étapes) un nombre de fois defini    
    def etape_aleatoire(self, root, dessin, conditions, a, b, liste_cellules, liste, etapes, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, nombre_execution):
        global en_marche
        global test_en_cours
        
        test_en_cours = "aleatoire" 
        liste_smalt = self.construire_dictionnaire_smalt(liste_cellules)
        liste_porte = [cellule.porte for cellule in liste if  cellule.porte.clefs != []]  
        print()
        
        def execute_iteration(i):
            if i < nombre_execution :
                random.shuffle(etapes)  #mélange les étapes
                self.executer_etape2(root, dessin, conditions, a, b, liste_cellules, liste, etapes, 0, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, nombre_execution, callback=lambda: execute_iteration(i + 1))
            else:
                callback_fin()  #callback de fin appelé quand on a réalisé toute les exécutions

        def callback_fin():
            conditions.tableau = self.matrice_unique(liste_cellules)
            message = conditions.verifier_conditions(liste_smalt, liste_porte, liste_cellules)
                
            if message == False:
                message = "Fin de l'exécution des étapes"
                print(f"Fin de l'execution des etapes")
                
            for clef, element in conditions.clefs_non_utilisees(liste_cellules):
                print(f"Clef non utilisée : {clef} (associée à l’élément {element})")

            dessin.canvas.create_text(largeur / 2, hauteur / 2, text=message, fill='red', font=("Arial", 20))
            
        execute_iteration(0)  #démarre la première exécution

    """
    def permute(nums, k):
        return list(itertools.permutations(nums, k))


    def test_permutations(root, dessin, conditions, a, b, liste_cellules, liste, etapes, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, palier):
        etapes = etapes_aleatoire(liste)
        liste_indices = [i for i in range(2*len(etapes))]
        etapes_posibles = etapes + etapes
        for k in range(1, len(liste_indices) + 1):
            liste_permutation = permute(liste_indices, k)

            for permutation in liste_permutation:
                print(permutation)
                print()
                liste = reinitialiser_liste(liste, palier)
                liste_cellules = reinitialiser_cellules(palier)
                etapes = etapes_aleatoire(liste)
                etapes_possibles = etapes + etapes
                conditions.tableau = matrice_unique(liste_cellules)
                liste_smalt = construire_dictionnaire_smalt(liste_cellules)
                liste_porte = [cellule.porte for cellule in liste if cellule.porte.clefs != []]
            
                for k in permutation:
                    etapes_possibles[k]()          
                
                message = conditions.verifier_conditions(liste_smalt, liste_porte, liste_cellules)
                if message == False:
                    message = "ok"
                else:
                    print(message)
                    break
                print(message)
                
    """ 

    #crée le dessin du tableau et les boutons permettant d'exécuter les étapes choisies ou aléatoires
    def test_tout(self, root, conditions, etapes, a, b, liste_cellules, liste, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, nombre_execution, palier):
        global en_marche
        en_marche = True

        # Frame principal avec 2 lignes : boutons (0), canvas (1)
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # Frame pour les boutons
        frame_boutons = tk.Frame(root)
        frame_boutons.grid(row=0, column=0, sticky="ew", pady=5)

        # Frame pour le canvas
        frame_canvas = tk.Frame(root)
        frame_canvas.grid(row=1, column=0, sticky="nsew")

        # Création du canvas dans le frame_canvas
        canvas = self.creation_canvas(frame_canvas, largeur, hauteur)
        dessin = Dessin(canvas)

        self.dessin_tableau(dessin, liste, a, b, taille, x_coin, y_coin, taille_serrure_mere, largeur, hauteur)

        # Fonction pour test aléatoire
        def demander_nombre_execution():
            popup = tk.Toplevel(root)
            popup.title("Nombre d'exécutions")
            tk.Label(popup, text="Entrez le nombre d'exécutions :").pack(pady=5)
            entry = tk.Entry(popup)
            entry.pack(pady=5)

            def valider():
                try:
                    nb = int(entry.get())
                    popup.destroy()

                    etapes_aleatoire = self.etapes_aleatoire(liste, conditions)

                    self.etape_aleatoire(root, dessin, conditions, a, b, liste_cellules, liste, etapes_aleatoire,
                                        largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, nb)

                except ValueError:
                    tk.messagebox.showerror("Erreur", "Veuillez entrer un nombre entier valide.")

            tk.Button(popup, text="Valider", command=valider).pack(pady=5)
            tk.Button(popup, text="Annuler", command=popup.destroy).pack(pady=5)

        # Boutons dans le frame_boutons
        bouton_test_classique = tk.Button(frame_boutons, text="test par étapes", command=lambda: self.executer_etape_avec_selection(
            root, dessin, conditions, a, b, liste_cellules, liste, etapes, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere))
        bouton_test_classique.grid(row=0, column=0, padx=5)

        bouton_start = tk.Button(frame_boutons, text="test aléatoire", command=demander_nombre_execution)
        bouton_start.grid(row=0, column=1, padx=5)

        #Réinitialise le canvas et les cellules
        def reset_cellules():
            nonlocal liste_cellules, liste
            # Réinitialisation complète
            initialisation = InitialisationCellules(palier)
            liste_cellules = initialisation.reinitialiser_cellules()

            # Vider les clefs utilisées
            conditions.clefs_utilisees.clear()

            # Filtrage des cellules visibles
            noms_cellules_visibles = [cellule.nom for cellule in liste]
            liste = [cellule for cellule in liste_cellules if cellule.nom in noms_cellules_visibles]

            # Régénération des étapes avec les nouveaux objets
            etapes_aleatoire = self.etapes_aleatoire(liste, conditions)

            # Nettoyage et redessin
            dessin.canvas.delete("all")
            self.dessin_tableau(dessin, liste, a, b, taille, x_coin, y_coin, taille_serrure_mere, largeur, hauteur)

        bouton_reset = tk.Button(frame_boutons, text="Réinitialiser", command=reset_cellules)
        bouton_reset.grid(row=0, column=2, padx=5)

        canvas.focus_set()

    #crée une liste avec toutes les etapes possibles pour chaque cellule de la liste des cellules
    #ce sont ces étapes qui vont être mélangées et exécutées aléatoirement      
    def etapes_aleatoire(self, liste_cellules, conditions):
        etapes = []
        for cellule in liste_cellules:
            partie_mobile = cellule.partie_mobile
            smalt = cellule.smalt
            porte = cellule.porte
            serrures = cellule.serrures
            transformateur = cellule.transformateur
            coffret = cellule.coffret

            if partie_mobile != []:
                etapes.append(lambda pm=partie_mobile: pm.insertion_extraction_manivelle())
                etapes.append(lambda pm=partie_mobile, s=smalt, c=cellule: pm.embrochage_debrochage(s, c))
                etapes.append(lambda pm=partie_mobile: pm.verrouillage_deverrouillage_tiroir())

            if smalt != []:
                etapes.append(lambda s=smalt: (
                    conditions.enregistrer_clef_utilisee(s.clefs[0], s) if s.clefs else None
                ) or s.verrouillage_deverrouillage_SMALT())

                etapes.append(lambda s=smalt, pm=partie_mobile, p=porte, c=cellule: (
                    conditions.enregistrer_clef_utilisee(s.clefs[0], s) if s.clefs else None
                ) or s.ouverture_fermeture_SMALT(pm, p, c))

            if porte != []:
                etapes.append(lambda p=porte, s=smalt: (
                    conditions.enregistrer_clef_utilisee(p.clefs[0], p) if p.clefs else None
                ) or p.ouverture_fermeture_panneau(s))

            if coffret != []:
                etapes.append(lambda c=coffret: (
                    conditions.enregistrer_clef_utilisee(c.clefs[0], c) if c.clefs else None
                ) or c.ouverture_fermeture_coffret())

            if transformateur != []:
                etapes.append(lambda t=transformateur: t.consignation_deconsignation_transformateur())

            for serrure in serrures:
                etapes.append(lambda sr=serrure: (
                    [conditions.enregistrer_clef_utilisee(clef, sr) for clef in sr.clefs],
                    sr.clefs_serrure_mere()
                ))

        return etapes
