#importation des modules necessaires
import tkinter as tk
from tkinter import messagebox
import numpy as np 
import random #pour exécuter les étapes aléatoirement   
from LHT_Classe_dessin import Dessin
from LHT_Classe_eclisse import Eclisse
from LHT_Classe_element_elec import ElementElectrique
from LHT_Classe_initialisation_cellules import InitialisationCellules


class Configuration:

    def __init__(self):
        #variables permettant de gerer correctement les mises en pause
        en_marche = True #variable qui se met à jour quand on est en pause
        test_en_cours = None #variable qui stocke le test actuel pour le reprendre quand on finit la mise en pause

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
                dessin.canvas.after(100, self.execution_aleatoire, root, dessin, conditions, a, b, liste_cellules, liste, etapes, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, nombre_execution)
            
    #configure la fenetre pour qu'elle detecte la mise en pause via l'appuie de la touche espace                        
    def configurer_reactions_clavier(self, root, dessin, conditions, a, b, liste_cellules, liste, etapes, index, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, nombre_execution):
        root.focus_set() 
        root.bind("<space>", lambda event: self.mettre_en_pause(root, dessin, conditions, a, b, liste_cellules, liste, etapes, index, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, nombre_execution)) 

    #fonction qui dessine une cellule en utilisant les fonctions de la classe Dessin       
    def dessin_final(self, dessin, a, b, taille, cellule, x_coin, y_coin, taille_serrure_mere):
        #on récupère tous les éléments électriques d'une cellule
        partie_mobile = cellule.partie_mobile
        smalt = cellule.smalt
        serrures = cellule.serrures
        porte = cellule.porte
        transformateur = cellule.transformateur
        matrice = cellule.matrice
        coffret = cellule.coffret
        eclisse = cellule.eclisse
        boite_eclisse = cellule.boite_eclisse
        source = cellule.source
        
        scale = dessin.scale_factor #échelle de zoom
        taille = round(taille*scale)
        a = round(a*scale)
        b = round(b*scale)
        taille_serrure_mere =round(taille_serrure_mere*scale)
        for i in range(len(matrice)):
                for j in range(len(matrice[0])):
                    x = j*taille + x_coin
                    y = i*taille + y_coin
                    if matrice[i][j] == 'disjoncteur':
                        dessin.dessiner_disjoncteur(x + a + taille/2, y + b , taille)
                        dessin.dessiner_elements_disjoncteur(partie_mobile, i, j, partie_mobile.representation, taille, a, b, x_coin, y_coin)
                                            
                    if matrice[i][j] == 'tiroir':      
                        dessin.dessiner_tiroir(x + a + taille/2, y + b , taille) 
                        dessin.dessiner_elements_disjoncteur(partie_mobile, i, j, partie_mobile.representation, taille, a, b, x_coin, y_coin)
                    
                    if matrice[i][j] == 'contacteur':      
                        dessin.dessiner_contacteur(x + a + taille/2, y + b , taille) 
                        dessin.dessiner_elements_disjoncteur(partie_mobile, i, j, partie_mobile.representation, taille, a, b, x_coin, y_coin)
                    
                    if matrice[i][j] == 'eclisse':
                        if eclisse.position == 'presente':
                            dessin.dessiner_eclisse_presente(x + a + taille/2, y + b, taille)
                            dessin.dessiner_elements_disjoncteur(eclisse, i, j, eclisse.representation, taille, a, b, x_coin, y_coin)
                        else :
                            dessin.dessiner_eclisse_absente(x + a + taille/2, y + b, taille)
                            dessin.dessiner_elements_disjoncteur(eclisse, i, j, eclisse.representation, taille, a, b, x_coin, y_coin)
                            
                    if matrice[i][j] == 'boite_eclisse':
                        dessin.dessiner_boite(boite_eclisse, x + a, y + b, taille)
                        dessin.dessiner_elements_disjoncteur(boite_eclisse, i, j, boite_eclisse.representation, taille, a, b, x_coin, y_coin)
                    
                    if matrice[i][j] == 'transformateur':
                        dessin.dessiner_transformateur(transformateur, x + a + taille/2, y + b, taille)
                        
                    if matrice[i][j] == 'coffret':
                        dessin.dessiner_coffret(coffret, x + a, y + b, taille)
                            
                    if matrice[i][j] == 'fil_h':    
                        dessin.dessiner_fil_h(x + a, y + b  + taille/2, taille)
                            
                    if matrice[i][j] == 'fil_v':
                        dessin.dessiner_fil_v(x + a + taille/2, y + b , taille)

                    if matrice[i][j] == 'smalt':  
                        dessin.dessiner_terre(x + a, y + b  + taille/2, taille)
                        dessin.dessiner_elements_smalt( smalt, i, j, smalt.representation, taille, a, b, x_coin, y_coin)
                    
                    if matrice[i][j] == 'fil_h_dep':
                        dessin.dessiner_fil_h_dep(x + a, y + b , taille)

                    if matrice[i][j] == 'fil_h_arrivee':
                        dessin.dessiner_fil_h_arrivee(x + a, y + b , taille)

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
                        dessin.dessiner_source(source, x + a, y + b, taille)
                        
                    if matrice[i][j] == 'fusible':
                        dessin.dessiner_fusible(x + a + taille/2, y + b, taille)
                        
        dessin.ecrire_mot(cellule.position_x*taille*5 + x_coin + 20, cellule.position_y*taille + y_coin + 10, taille, cellule.nom)  

        #on distingue si on est un depart ou une arrivée :
        #pour chaque tableau, une ligne horizontale représente le tableau : un départ sera dessiné au dessous du tableau, et une arrivée au dessus   
        if cellule.statut == "depart": 
            if serrures != []:
                dessin.dessiner_serrure_mere( 
                    x_coin + a, 
                    y_coin + b,
                    taille,
                    taille_serrure_mere,
                    cellule
                    )
            # Rectangle de la porte d'acces dessiné en pointillé avec sa clef
            if porte != []:
                dessin.canvas.create_rectangle(
                    x_coin + taille/5 + a,
                    cellule.longueur*taille  + y_coin - 3*taille/4 + b ,
                    cellule.largeur*taille +x_coin - taille/5 + a,
                    cellule.longueur*taille  + y_coin -taille/10 + b,
                    dash=(4,2)
                    )
                if porte.clefs != []:
                    dessin.dessiner_carre( 
                        x_coin + taille/5 + a+ taille/4,
                        cellule.longueur*taille  + y_coin - taille/2 + b,
                        porte.clefs[0], 
                        taille
                        )
                    dessin.canvas.create_text(
                        x_coin + taille/5 + a+ taille/4,
                        cellule.longueur*taille  + y_coin - taille/2 + b,
                        text=porte.clefs[0].nom,
                        font=("Arial", taille//5)
                        )

        elif cellule.statut == "arrivee":
            if serrures != []:
                dessin.dessiner_serrure_mere( 
                    x_coin + a, 
                    y_coin + b +taille_serrure_mere + len(cellule.matrice)*taille,
                    taille,
                    taille_serrure_mere,
                    cellule
                    )
            if porte != []:
                dessin.canvas.create_rectangle(
                    x_coin + taille/5 + a,
                    (len(cellule.matrice)-cellule.longueur)*taille + y_coin + taille/10 + b ,
                    cellule.largeur*taille + x_coin - taille/5 + a, 
                    (len(cellule.matrice)-cellule.longueur)*taille + y_coin + 2*taille/3 + b,
                    dash=(4,2)
                    )
                if porte.clefs != []:
                    dessin.dessiner_carre( 
                        x_coin + taille/5 + a+ taille/4, 
                        (len(cellule.matrice)-cellule.longueur)*taille + y_coin + taille/10 + b + taille/3,
                        porte.clefs[0], 
                        taille
                        )
                    dessin.canvas.create_text(
                        x_coin + taille/5 + a+ taille/4,
                        (len(cellule.matrice)-cellule.longueur)*taille + y_coin + taille/10 + b + taille/3,
                        text=porte.clefs[0].nom,
                        font=("Arial", taille//5)
                        )            
        
    #dessin final du tableau : chaque cellule est positionnée correctement grâce aux caractéristique position_x et position_y
    def dessin_tableau(self, dessin, liste_cellules, a, b, taille, x_coin,y_coin, taille_serrure_mere):
        for cellule in liste_cellules:
            if cellule.statut == 'depart' :
                self.dessin_final(dessin, cellule.position_x*a, cellule.position_y*b, taille, cellule, x_coin,y_coin, taille_serrure_mere)
            elif cellule.statut == 'arrivee':
                self.dessin_final(dessin, cellule.position_x*a, (cellule.position_y-(len(cellule.matrice)-1))*b, taille, cellule, x_coin,y_coin, taille_serrure_mere)    

        liste = 2*taille
        dessin.canvas.create_text(30, taille/5, text='clefs libres : ', font=("Arial", taille//4))
        dessin.canvas.create_text(50, 7*taille/10, text='eclisses A dispos : ' + f"{Eclisse.eclisseA}", font=("Arial", taille//4))
        dessin.canvas.create_text(50, 12*taille/10, text='eclisses B dispos : ' + f"{Eclisse.eclisseB}", font=("Arial", taille//4))    
        for clefs in ElementElectrique.clefs_libres:
            dessin.canvas.create_text(30 + liste, taille/5, text=clefs +',', font=("Arial", taille//4))
            liste+= taille
            
    #fonction qui récupère la position_y négative la plus importante : utilisée pour placer correctement les éléments dans la matrice finale 
    def max_negatif_position_y(self, liste_cellules):
        liste_nombre = [cellule.position_y for cellule in liste_cellules]
        max_neg = 0
        for i in range(len(liste_nombre)):
            nombre = liste_nombre[i]
            if nombre < 0 and abs(nombre) > max_neg:
                max_neg = abs(nombre)
        return max_neg

    #fonction qui recupere la position_x negative la plus importante : utilisée pour placer correctement les éléments dans la matrice finale 
    def max_negatif_position_x(self, liste_cellules):
        liste_nombre = [cellule.position_x for cellule in liste_cellules]
        max_neg = 0
        for i in range(len(liste_nombre)):
            nombre = liste_nombre[i]
            if nombre < 0 and abs(nombre) > max_neg:
                max_neg = abs(nombre)
        return max_neg

    #fonction qui récupère la position_x positive la plus importante : utilisée pour connaitre la taille que doit avoir la matrice finale
    def max_positif_position_x(self, liste_cellules):
        liste_nombre = [cellule.position_x for cellule in liste_cellules]
        max_pos = 0
        for i in range(len(liste_nombre)):
            nombre = liste_nombre[i]
            if nombre > 0 and nombre > max_pos:
                max_pos = nombre
        return max_pos

    #fonction qui récupère la position_y positive la plus importante : utilisée pour connaitre la taille que doit avoir la matrice finale
    def max_positif_position_y(self, liste_cellules):
        liste_nombre = [cellule.position_y for cellule in liste_cellules]
        max_pos = 0
        for i in range(len(liste_nombre)):
            nombre = liste_nombre[i]
            if nombre > 0 and nombre > max_pos:
                max_pos = nombre
        return max_pos

    #cree la matrice finale (unique) permettant de verifier l'existence d'un chemin entre deux elements electriques        
    def matrice_unique(self, liste_cellules):
        a = self.max_negatif_position_y(liste_cellules) +10
        b = self.max_negatif_position_x(liste_cellules)
        aa = self.max_positif_position_y(liste_cellules)
        bb = self.max_positif_position_x(liste_cellules)
        matrice_finale = np.zeros((a + aa + 20, 5*(b+bb+2)), dtype=object)

        for cellule in liste_cellules:
            matrice = cellule.matrice  
            ligne, colonne = matrice.shape   
            
            if cellule.statut =='depart':  
                matrice_finale[a+cellule.position_y:a+cellule.position_y+ligne, 5*(cellule.position_x + b):5*(cellule.position_x + b)+colonne] = matrice
            elif cellule.statut =='arrivee':
                matrice_finale[a+cellule.position_y - ligne:a+cellule.position_y, 5*(cellule.position_x + b):5*(cellule.position_x + b)+colonne] = matrice
        
        return matrice_finale            

    #fonction qui execute des etapes choisies    
    def executer_etape(self, root, dessin, conditions, a, b, liste_cellules, liste, etapes, index, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere):
        nombre_execution = 1
        self.configurer_reactions_clavier(root, dessin, conditions, a, b, liste_cellules, liste, etapes, index, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, nombre_execution) 
        global test_en_cours
        test_en_cours = "classique"  
        conditions.tableau = self.matrice_unique(liste_cellules)
        
        global en_marche
        
        if not en_marche: #si on est en pause, on stoppe l'exécution de la fonction
            return
        
        if index < len(etapes):
            
            dessin.canvas.delete("all")
                
            #Exécute l'étape actuelle
            if en_marche:
                etapes[index]() 
            
            self.dessin_tableau(dessin, liste, a, b, taille, 15, y_coin, taille_serrure_mere)
            
            message = conditions.verifier_conditions(liste_cellules) #vérifie si il y a des conditions d'arrêt
            #verifier_conditions renvoit soit False, soit un message
            
            #si message n'est pas False, ca veut dire qu'il y a un un danger donc on arrête l'exécution et on affiche le message 
            if message != False:
                dessin.canvas.create_text(largeur / 2, hauteur / 2, text=message, fill='red', font=("Arial", 20))
                return
            
            #passe à l'étape suivante après 500 ms si il n'y a pas de danger et qu'on n'est pas en pause
            dessin.canvas.after(500, self.executer_etape, root, dessin, conditions, a, b, liste_cellules, liste, etapes, index + 1, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere)
            
        else:
            dessin.canvas.delete("all")
            self.dessin_tableau(dessin, liste, a, b, taille, 15, y_coin, taille_serrure_mere)
            message = conditions.verifier_conditions(liste_cellules)
            if message == False:
                message = "Fin de l'execution des etapes"
                print(f"Fin de l'execution des etapes")
            dessin.canvas.create_text(largeur / 2, hauteur / 2, text=message, fill='red', font=("Arial", 20)) 
 
    #execute les étapes aléatoirement     
    def executer_etape2(self, root, dessin, conditions, a, b, liste_cellules, liste, etapes, index, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, nombre_execution, callback=None): 
        
        self.configurer_reactions_clavier(root, dessin, conditions, a, b, liste_cellules, liste, etapes, index, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, nombre_execution) 
        conditions.tableau = self.matrice_unique(liste_cellules)
        
        global en_marche
        
        if not en_marche:
            return
        
        if index < len(etapes):
            
            dessin.canvas.delete("all")
                
            if en_marche:
                etapes[index]()
            self.dessin_tableau(dessin, liste, a, b, taille, 15, y_coin, taille_serrure_mere)
            
            message = conditions.verifier_conditions(liste_cellules)
            
            #si message n'est pas False, ca veut dire qu'il y a un un danger donc on arrête l'exécution et on affiche le message
            if message != False:
                dessin.canvas.create_text(largeur / 2, hauteur / 2, text=message, fill='red', font=("Arial", 20))
                return
            
            dessin.canvas.after(1, self.executer_etape2, root, dessin, conditions, a, b, liste_cellules, liste, etapes, index + 1, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, nombre_execution, callback)
                
        else:
            dessin.canvas.delete("all")
            self.dessin_tableau(dessin, liste, a, b, taille, 15, y_coin, taille_serrure_mere)
            conditions.tableau = self.matrice_unique(liste_cellules)
            message = conditions.verifier_conditions(liste_cellules)

            if message != False:
                dessin.canvas.create_text(largeur / 2, hauteur / 2, text=message, fill='red', font=("Arial", 20))
                return
                        
            if callback is not None:
                callback()  #callback de fin appelé quand on a réalisé toute les exécutions
   
    #mélange les etapes et appelle la fonction "executer_etape2" (qui exécute les étapes) un nombre de fois defini    
    def execution_aleatoire(self, root, dessin, conditions, a, b, liste_cellules, liste, etapes, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, nombre_execution):
        global en_marche
        global test_en_cours
        
        test_en_cours = "aleatoire" 
        
        def execute_iteration(i):
            if i < nombre_execution :
                random.shuffle(etapes)  #mélange les étapes
                self.executer_etape2(root, dessin, conditions, a, b, liste_cellules, liste, etapes, 0, largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, nombre_execution, callback=lambda: execute_iteration(i + 1))
            else:
                callback_fin()  #callback de fin appelé quand on a réalisé toute les exécutions

        def callback_fin():
            conditions.tableau = self.matrice_unique(liste_cellules)
            message = conditions.verifier_conditions(liste_cellules)
                
            if message == False:
                message = "Fin de l'exécution des étapes"
                print(f"Fin de l'execution des etapes")

            for clef, element in conditions.clefs_non_utilisees(liste_cellules):
                print(f"Clef non utilisée : {clef} (associée à l’élément {element})")

            dessin.canvas.create_text(largeur / 2, hauteur / 2, text=message, fill='red', font=("Arial", 20))
            
        execute_iteration(0)  #on démarre la première exécution

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

    #crée le dessin du tableau et les boutons permettant d'exécuter les étapes choisies ou aléatoires
    def test_tout(self, root, conditions, etapes, etapes_aleatoire, a, b, liste_cellules, liste,
                  largeur, hauteur, taille, x_coin, y_coin, taille_serrure_mere, nombre_execution, palier):
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

        self.dessin_tableau(dessin, liste, a, b, taille, x_coin, y_coin, taille_serrure_mere)

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

                    self.execution_aleatoire(root, dessin, conditions, a, b, liste_cellules, liste, etapes_aleatoire,
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
            self.dessin_tableau(dessin, liste, a, b, taille, x_coin, y_coin, taille_serrure_mere)

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
            eclisse = cellule.eclisse
            
            if partie_mobile:        
                etapes.append(lambda pm=partie_mobile: pm.insertion_extraction_manivelle())
                etapes.append(lambda pm=partie_mobile, s=smalt, c=cellule: pm.embrochage_debrochage(s, c))
                etapes.append(lambda pm=partie_mobile: pm.verrouillage_deverrouillage_tiroir())
            if eclisse !=[]:
                etapes.append(lambda e=eclisse, c=cellule: (
                    conditions.enregistrer_clef_utilisee(e.clefs[0], e) if e.clefs else None
                ) or e.ajout_suppression_eclisse(c))
                etapes.append(lambda e=eclisse: (
                    conditions.enregistrer_clef_utilisee(e.clefs[0], e) if e.clefs else None
                ) or e.verrouillage_deverrouillage_tiroir())
                    
            if smalt != []:
                etapes.append(lambda s=smalt: (
                    conditions.enregistrer_clef_utilisee(s.clefs[0], s) if s.clefs else None
                ) or s.verrouillage_deverrouillage_SMALT())

                etapes.append(lambda s=smalt, pm=partie_mobile, c=cellule: (
                    conditions.enregistrer_clef_utilisee(s.clefs[0], s) if s.clefs else None
                ) or s.ouverture_fermeture_SMALT(pm, c))

            if porte != []:
                etapes.append(lambda p=porte, s=smalt: (
                    conditions.enregistrer_clef_utilisee(p.clefs[0], p) if p.clefs else None
                ) or p.ouverture_panneau(s))

            if coffret != []:
                etapes.append(lambda c=coffret, ce=cellule: (
                    conditions.enregistrer_clef_utilisee(c.clefs[0], c) if c.clefs else None
                ) or c.ouverture_fermeture_coffret(ce))

            if transformateur != []:
                etapes.append(lambda t=transformateur: t.consignation_deconsignation_transformateur())

            for serrure in serrures:
                etapes.append(lambda sr=serrure,  c=cellule: (
                    [conditions.enregistrer_clef_utilisee(clef, sr) for clef in sr.clefs],
                    sr.clefs_serrure_mere(c)
                ))
        return etapes

    '''
    #cree la liste des etapes pour realimenter 1LHB par LHT
    def etapes_choisies_essai(liste_cellules):
        etapes = []
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '1LHA2')
        partie_mobile = cellule.partie_mobile
        smalt = cellule.smalt
        etapes.append(lambda pm=partie_mobile: pm.insertion_extraction_manivelle())
        etapes.append(lambda pm=partie_mobile, s=smalt, c=cellule: pm.embrochage_debrochage(s, c))
        etapes.append(lambda pm=partie_mobile: pm.insertion_extraction_manivelle())
        etapes.append(lambda pm=partie_mobile: pm.verrouillage_deverrouillage_tiroir())
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '1LHA4')
        partie_mobile = cellule.partie_mobile
        smalt = cellule.smalt
        etapes.append(lambda pm=partie_mobile: pm.insertion_extraction_manivelle())
        etapes.append(lambda pm=partie_mobile, s=smalt, c=cellule: pm.embrochage_debrochage(s, c))
        etapes.append(lambda pm=partie_mobile: pm.insertion_extraction_manivelle())
        etapes.append(lambda pm=partie_mobile: pm.verrouillage_deverrouillage_tiroir())
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '1LHA5')
        partie_mobile = cellule.partie_mobile
        smalt = cellule.smalt
        etapes.append(lambda pm=partie_mobile: pm.insertion_extraction_manivelle())
        etapes.append(lambda pm=partie_mobile, s=smalt, c=cellule: pm.embrochage_debrochage(s, c))
        etapes.append(lambda pm=partie_mobile: pm.insertion_extraction_manivelle())
        etapes.append(lambda pm=partie_mobile: pm.verrouillage_deverrouillage_tiroir())
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '0LHT121JC')
        coffret = cellule.coffret
        etapes.append(lambda c=coffret, ce=cellule: c.ouverture_fermeture_coffret(cellule))
        partie_mobile = cellule.partie_mobile
        smalt = cellule.smalt
        serrure = cellule.serrures[0]
        etapes.append(lambda pm=partie_mobile: pm.verrouillage_deverrouillage_tiroir())
        etapes.append(lambda pm=partie_mobile: pm.insertion_extraction_manivelle())
        etapes.append(lambda pm=partie_mobile, s=smalt, c=cellule: pm.embrochage_debrochage(s, c))
        etapes.append(lambda pm=partie_mobile: pm.insertion_extraction_manivelle())
        etapes.append(lambda pm=partie_mobile: pm.verrouillage_deverrouillage_tiroir())
        etapes.append(lambda sr=serrure, c=cellule: sr.clefs_serrure_mere(c))
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT())
        etapes.append(lambda s=smalt, pm=partie_mobile, c=cellule: s.ouverture_fermeture_SMALT(pm, c))
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT())
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '0LHT3')
        partie_mobile = cellule.partie_mobile
        smalt = cellule.smalt
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT())
        etapes.append(lambda s=smalt, pm=partie_mobile, c=cellule: s.ouverture_fermeture_SMALT(pm, c))
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT())  
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '1LHB5')
        partie_mobile = cellule.partie_mobile
        smalt = cellule.smalt
        etapes.append(lambda pm=partie_mobile: pm.insertion_extraction_manivelle())
        etapes.append(lambda pm=partie_mobile, s=smalt, c=cellule: pm.embrochage_debrochage(s, c))
        etapes.append(lambda pm=partie_mobile: pm.insertion_extraction_manivelle())
        etapes.append(lambda pm=partie_mobile: pm.verrouillage_deverrouillage_tiroir())
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '1LHB4')
        partie_mobile = cellule.partie_mobile
        smalt = cellule.smalt
        etapes.append(lambda pm=partie_mobile: pm.insertion_extraction_manivelle())
        etapes.append(lambda pm=partie_mobile, s=smalt, c=cellule: pm.embrochage_debrochage(s, c))
        etapes.append(lambda pm=partie_mobile: pm.insertion_extraction_manivelle())
        etapes.append(lambda pm=partie_mobile: pm.verrouillage_deverrouillage_tiroir())
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '2LHT6')
        serrure = cellule.serrures[0]
        etapes.append(lambda sr=serrure, c=cellule: sr.clefs_serrure_mere(c))
        serrure = cellule.serrures[1]
        etapes.append(lambda sr=serrure, c=cellule: sr.clefs_serrure_mere(c))
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '1LHT2')
        partie_mobile = cellule.partie_mobile
        smalt = cellule.smalt
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT())
        etapes.append(lambda s=smalt, pm=partie_mobile, c=cellule: s.ouverture_fermeture_SMALT(pm, c))
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT()) 
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '2LHT3')
        partie_mobile = cellule.partie_mobile
        smalt = cellule.smalt
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT())
        etapes.append(lambda s=smalt, pm=partie_mobile, c=cellule: s.ouverture_fermeture_SMALT(pm, c))
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT()) 
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '2LHT6')
        serrure = cellule.serrures[2]
        etapes.append(lambda sr=serrure, c=cellule: sr.clefs_serrure_mere(c))
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '1LHA3')
        partie_mobile = cellule.partie_mobile
        smalt = cellule.smalt
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT())
        etapes.append(lambda s=smalt, pm=partie_mobile, c=cellule: s.ouverture_fermeture_SMALT(pm, c))
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT()) 
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '1LHT1')
        eclisse = cellule.eclisse
        etapes.append(lambda e=eclisse: e.verrouillage_deverrouillage_tiroir())
        etapes.append(lambda e=eclisse, c=cellule: e.ajout_suppression_eclisse(c))
        etapes.append(lambda e=eclisse: e.verrouillage_deverrouillage_tiroir())
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '1LHA3')
        partie_mobile = cellule.partie_mobile
        smalt = cellule.smalt
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT())
        etapes.append(lambda s=smalt, pm=partie_mobile, c=cellule: s.ouverture_fermeture_SMALT(pm, c))
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT()) 
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '1LHT4')
        boite_eclisse = cellule.boite_eclisse
        etapes.append(lambda be=boite_eclisse: be.verrouillage_deverrouillage_tiroir())
        etapes.append(lambda be=boite_eclisse, c=cellule: be.ouverture_fermeture_boite(c))
        etapes.append(lambda be=boite_eclisse, c=cellule: be.ouverture_fermeture_boite(c))
        etapes.append(lambda be=boite_eclisse: be.verrouillage_deverrouillage_tiroir())
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '1LHT5')
        boite_eclisse = cellule.boite_eclisse
        etapes.append(lambda be=boite_eclisse: be.verrouillage_deverrouillage_tiroir())
        etapes.append(lambda be=boite_eclisse, c=cellule: be.ouverture_fermeture_boite(c))
        etapes.append(lambda be=boite_eclisse, c=cellule: be.ouverture_fermeture_boite(c))
        etapes.append(lambda be=boite_eclisse: be.verrouillage_deverrouillage_tiroir())
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '1LHB2')
        partie_mobile = cellule.partie_mobile
        smalt = cellule.smalt
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT())
        etapes.append(lambda s=smalt, pm=partie_mobile, c=cellule: s.ouverture_fermeture_SMALT(pm, c))
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT())
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '1LHT6')
        eclisse = cellule.eclisse
        etapes.append(lambda e=eclisse: e.verrouillage_deverrouillage_tiroir())
        etapes.append(lambda e=eclisse, c=cellule: e.ajout_suppression_eclisse(c))
        etapes.append(lambda e=eclisse: e.verrouillage_deverrouillage_tiroir())
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '1LHB2')
        partie_mobile = cellule.partie_mobile
        smalt = cellule.smalt
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT())
        etapes.append(lambda s=smalt, pm=partie_mobile, c=cellule: s.ouverture_fermeture_SMALT(pm, c))
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT())
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '2LHT6')
        serrure = cellule.serrures[2]
        etapes.append(lambda sr=serrure, c=cellule: sr.clefs_serrure_mere(c))
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '2LHT3')
        partie_mobile = cellule.partie_mobile
        smalt = cellule.smalt
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT())
        etapes.append(lambda s=smalt, pm=partie_mobile, c=cellule: s.ouverture_fermeture_SMALT(pm, c))
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT())
        etapes.append(lambda sr=serrure, c=cellule: sr.clefs_serrure_mere(c))
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '1LHT2')
        partie_mobile = cellule.partie_mobile
        smalt = cellule.smalt
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT())
        etapes.append(lambda s=smalt, pm=partie_mobile, c=cellule: s.ouverture_fermeture_SMALT(pm, c))
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT())
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '2LHT6')
        serrure = cellule.serrures[1]
        etapes.append(lambda sr=serrure, c=cellule: sr.clefs_serrure_mere(c))
        serrure = cellule.serrures[0]
        etapes.append(lambda sr=serrure, c=cellule: sr.clefs_serrure_mere(c))
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '1LHA3')
        serrure = cellule.serrures[0]
        etapes.append(lambda sr=serrure, c=cellule: sr.clefs_serrure_mere(c))
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '1LHB4')
        serrure = cellule.serrures[0]
        etapes.append(lambda sr=serrure, c=cellule: sr.clefs_serrure_mere(c))
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '1LHB2')
        partie_mobile = cellule.partie_mobile
        smalt = cellule.smalt
        etapes.append(lambda pm=partie_mobile: pm.verrouillage_deverrouillage_tiroir())
        etapes.append(lambda pm=partie_mobile: pm.insertion_extraction_manivelle())
        etapes.append(lambda pm=partie_mobile, s=smalt, c=cellule: pm.embrochage_debrochage(s, c))
        etapes.append(lambda pm=partie_mobile: pm.insertion_extraction_manivelle())
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '0LHT3')
        partie_mobile = cellule.partie_mobile
        smalt = cellule.smalt
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT())
        etapes.append(lambda s=smalt, pm=partie_mobile, c=cellule: s.ouverture_fermeture_SMALT(pm, c))
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT()) 
        cellule = Cellule.chercher_cellule_par_nom(liste_cellules, '0LHT121JC')
        partie_mobile = cellule.partie_mobile
        smalt = cellule.smalt
        serrure = cellule.serrures[0]
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT())
        etapes.append(lambda s=smalt, pm=partie_mobile, c=cellule: s.ouverture_fermeture_SMALT(pm, c))
        etapes.append(lambda s=smalt: s.verrouillage_deverrouillage_SMALT())
        etapes.append(lambda sr=serrure, c=cellule: sr.clefs_serrure_mere(c))
        etapes.append(lambda pm=partie_mobile: pm.verrouillage_deverrouillage_tiroir())
        etapes.append(lambda pm=partie_mobile: pm.insertion_extraction_manivelle())
        etapes.append(lambda pm=partie_mobile, s=smalt, c=cellule: pm.embrochage_debrochage(s, c))
        etapes.append(lambda pm=partie_mobile: pm.insertion_extraction_manivelle())
        etapes.append(lambda pm=partie_mobile: pm.verrouillage_deverrouillage_tiroir())
        coffret = cellule.coffret
        etapes.append(lambda c=coffret, ce=cellule: c.ouverture_fermeture_coffret(ce))
        return etapes
    '''    
    