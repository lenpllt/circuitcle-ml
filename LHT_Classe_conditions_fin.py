import tkinter as tk
from tkinter import ttk
from LHT_Classe_cellule import Cellule

#création de la classe ConditionsFin, permettant d'écrive les conditions qui arrêteront l'exécution des étapes
class ConditionsFin:

    #caractéristiques de la classe
    def __init__(self, tableau):
        self.tableau = tableau
        self.starts, self.ends = self.rechercher_sources_smalt()  #Pré-calcule les positions des sources et smalts dans le tableau
        self.conditions = []  #liste d'ajout de nouvelles conditions
        self.operateur = {"est": "==", "n'est pas": "!="}  
        self.attribut_selon_etat = {"prisonniere": "etat", "presente": "etat", "absente": "etat", "penne rentre": "penne", "penne sorti": "penne"}
        self.clef_combobox = None #variable qui sera True si on ajoute une condition sur une clef
        self.condition_result = False
        self.clefs_utilisees = set()

    #fonction qui construit un dictionnaire des smalts des cellules voisines
    #permet de regarder pour un cellule donnée si on doit se soucier de l'état de smalts de cellules voisines
    #c'est la cas lorsqu'on ouvre un panneau, ou lorsque l'on veut récupérer une éclisse : les deux tableaux voisins doivent être mis à la terre
    @staticmethod
    def construire_dictionnaire_smalt(liste_cellules):
        liste_smalt = {}
        
        for cellule in liste_cellules:
            if 'LHC' in cellule.nom: #si la cellule appartient au tableau LHC, on regarde si elle a une voisine
                if cellule.voisine:
                    voisine = Cellule.chercher_cellule_par_nom(liste_cellules, cellule.voisine[0])
                    liste_smalt[cellule.nom] = [cellule.smalt, voisine.smalt]
            '''       
            if cellule.eclisse: #si la cellule possède une éclisse, on regarde ses voisines
                if cellule.voisine:
                    liste_smalt[cellule.nom] = []
                    for nom_voisine in cellule.voisine:
                        voisine = Cellule.chercher_cellule_par_nom(liste_cellules, nom_voisine)
                        liste_smalt[cellule.nom].append(voisine.smalt)
                    '''
        
        return liste_smalt

    #crée une liste de tous les éléments électriques d'une cellules
    def elements_possibles(self, cellule):
        elements = []
        partie_mobile = cellule.partie_mobile
        smalt = cellule.smalt
        porte = cellule.porte
        serrures = cellule.serrures
        transformateur = cellule.transformateur
        coffret = cellule.coffret
        if partie_mobile != []:
            elements.append(partie_mobile)
        if smalt != []:
            elements.append(smalt)
        if porte != []:
           elements.append(porte)
        if coffret != []:
            elements.append(coffret)
        if transformateur != []:
            elements.append(transformateur)
        for serrure in serrures:
            elements.append(serrure) 
        return elements

    #défnit les attributs (ce sur quoi on peut ajouter des conditions) pour chaque élément électrique
    def attributs_possible(self, element):     
        attribut = []
        if element.type == 'smalt':
            attribut.append('etat')
            attribut.append('clefs')
        elif element.type in ['transformateur', 'panneau', 'coffret', 'serrure_mere']:
            attribut.append('clefs')
        else :
            attribut.append('etat')
            attribut.append('position')
            attribut.append('clefs')
        return attribut

    #définit les états possibles une fois l'attribut choisi
    def etats_possible(self, attribut):
        etat = []
        if attribut == 'etat':
            etat.append('ferme')
            etat.append('ouvert')
        if attribut == 'position':
            etat.append('embroche')
            etat.append('debroche')
        return etat
            
    #met à jour les éléments électriques possibles une fois la cellule sélectionnée        
    def mettre_a_jour_elements(self, liste_cellules, cellule_combobox, element_combobox, attribut_combobox, etat_combobox):
        element_combobox.set('')  
        attribut_combobox.set('')  
        etat_combobox.set('')  # Efface la sélection de l'état
        if self.clef_combobox:
            self.clef_combobox.set('')
        cellule_nom = cellule_combobox.get()
        cellule_obj = next((c for c in liste_cellules if c.nom == cellule_nom), None)
        if cellule_obj:
            elements = self.elements_possibles(cellule_obj) #récupère les éléments électriques de la cellule
            element_combobox['values'] = [element.nom for element in elements]
        else:
            element_combobox.set("")  

    #met à jour les attributs possibles une fois l'élément électrique sélectionné      
    def mettre_a_jour_attributs(self, liste_cellules, cellule_combobox, element_combobox, attribut_combobox, etat_combobox):
        cellule_nom = cellule_combobox.get()
        element_nom = element_combobox.get()
        attribut_combobox.set('')  
        etat_combobox.set('')  
        if self.clef_combobox:
            self.clef_combobox.set('')
        cellule_obj = next((c for c in liste_cellules if c.nom == cellule_nom), None)
        if cellule_obj:
            elements = self.elements_possibles(cellule_obj)
            element_obj = next((e for e in elements if e.nom == element_nom), None)
            attributs_possibles = self.attributs_possible(element_obj)
            attribut_combobox['values'] = attributs_possibles
                      
    #ajoute la conditon que l'on vient d'écrire      
    def ajouter_condition(self, liste_cellules, cellule_combobox, element_combobox, operator_combobox, attribut_combobox, etat_combobox, condition_text):
        cellule_nom = cellule_combobox.get()
        element_nom = element_combobox.get()
        cellule_obj = next((c for c in liste_cellules if c.nom == cellule_nom), None)
        elements = self.elements_possibles(cellule_obj)
        element_obj = next((e for e in elements if e.nom == element_nom), None)
        op = operator_combobox.get()
        attribut = attribut_combobox.get()
        etat = etat_combobox.get()
        operateur_python = self.operateur[op]
        
        if attribut == "clefs":
            self.clefs_utilisees.add(clef_obj.nom)

        if not cellule_nom or not element_nom or not op or not attribut or not etat:
            tk.messagebox.showerror("Erreur", "Veuillez sélectionner une valeur dans chaque menu déroulant.")
            return

        if cellule_obj:
            if attribut == "clefs":
                if self.clef_combobox is None or not self.clef_combobox.get():
                    tk.messagebox.showerror("Erreur", "Veuillez sélectionner une clef.")
                    return
        
                clef_nom = self.clef_combobox.get()
                clef_obj = next((clef for clef in element_obj.clefs if clef.nom == clef_nom), None)
        
                if not clef_obj:
                    tk.messagebox.showerror("Erreur", "Clef non trouvée.")
                    return
                
                condition =[clef_obj, operateur_python, etat]
                ecriture_condition = f"(clef {clef_obj.nom} {element_obj.nom} {element_obj.cellule} {op} {etat})"
            else:
                condition =[element_obj, attribut, operateur_python, etat]
                ecriture_condition = f"({element_obj.nom} {element_obj.cellule} {op} {etat})"
                
            #Ajoute la condition à la liste
            self.conditions.append(condition) #ajoute la condition à la liste de conditions
            print(self.conditions)

            #ajoute la condition au texte existant
            current_text = condition_text.get("1.0", tk.END).strip()
            new_text = f"{current_text} \n {ecriture_condition}" if current_text else ecriture_condition
            condition_text.delete("1.0", tk.END)
            condition_text.insert(tk.END, new_text + "\n")

    #fonction qui permet de supprimer la dernière condition écrite        
    def supprimer_derniere_condition(self, condition_text):
        current_text = condition_text.get("1.0", tk.END).strip()
        lignes = current_text.split("\n")
        
        #vérifie qu'il y a des lignes à supprimer
        if lignes:
            lignes.pop() 
            new_text = "\n".join(lignes)
            condition_text.delete("1.0", tk.END)
            #Réinsére le nouveau texte sans la dernière ligne
            condition_text.insert(tk.END, new_text + "\n")
        if self.conditions:
            self.conditions.pop()
            print(self.conditions)
            
            
    #ajoute le texte de la dernière condition
    def ajouter_texte_et_condition(self, condition_text, texte_a_ajouter):
        current_text = condition_text.get("1.0", tk.END).strip()
        new_text = f"{current_text} \n {texte_a_ajouter}".strip()
        condition_text.delete("1.0", tk.END)
        condition_text.insert(tk.END, new_text + "\n")
        
        #ajoute le texte à la liste de conditions
        self.conditions.append(texte_a_ajouter)
        
    #crée l'interface pour ajouter les conditions souhaitées    
    def ajout_conditions(self, root, liste_cellules):

        condition_label = tk.Label(root, text="Construire la condition:")
        condition_label.pack()

        condition_text = tk.Text(root, height=15, width=80)
        condition_text.pack()

        buttons_frame = tk.Frame(root)
        buttons_frame.pack()

        tk.Button(buttons_frame, text="(", command=lambda: self.ajouter_texte_et_condition(condition_text, "(")).pack(side=tk.LEFT)
        tk.Button(buttons_frame, text=")", command=lambda: self.ajouter_texte_et_condition(condition_text, ")")).pack(side=tk.LEFT)
        tk.Button(buttons_frame, text="AND", command=lambda: self.ajouter_texte_et_condition(condition_text, "AND")).pack(side=tk.LEFT)
        tk.Button(buttons_frame, text="OR", command=lambda: self.ajouter_texte_et_condition(condition_text, "OR")).pack(side=tk.LEFT)

        frame_bandeaux = tk.Frame(root)
        frame_bandeaux.pack(pady=10)

        bandeau_frame = tk.Frame(frame_bandeaux)
        bandeau_frame.pack(pady=5)
        
        
        cellule_label = tk.Label(bandeau_frame, text="cellule")
        cellule_label.grid(row=0, column=0)
        
        cellule_combobox = ttk.Combobox(bandeau_frame, values=[cell.nom for cell in liste_cellules], width=15)
        cellule_combobox.grid(row=1, column=0)
        cellule_combobox.bind("<<ComboboxSelected>>", lambda event: self.mettre_a_jour_elements(liste_cellules, cellule_combobox, element_combobox, attribut_combobox, etat_combobox))

        element_label = tk.Label(bandeau_frame, text="element electrique")
        element_label.grid(row=0, column=1)
        
        element_combobox = ttk.Combobox(bandeau_frame, width=15)
        element_combobox.grid(row=1, column=1)
        element_combobox.bind("<<ComboboxSelected>>", lambda event: self.mettre_a_jour_attributs(liste_cellules, cellule_combobox, element_combobox, attribut_combobox, etat_combobox))
        
        attribut_label = tk.Label(bandeau_frame, text="attribut")
        attribut_label.grid(row=0, column=2)
        
        attribut_combobox = ttk.Combobox(bandeau_frame, width=15)
        attribut_combobox.grid(row=1, column=2)
        
        operator_label = tk.Label(bandeau_frame, text="comparaison")
        operator_label.grid(row=0, column=4)

        operator_combobox = ttk.Combobox(bandeau_frame, values=list(self.operateur.keys()), width=10)
        operator_combobox.grid(row=1, column=4)
        operator_combobox.set("est")

        etat_label = tk.Label(bandeau_frame, text="etat")
        etat_label.grid(row=0, column=5)
        
        etat_combobox = ttk.Combobox(bandeau_frame, width=15)
        etat_combobox.grid(row=1, column=5)
       
       #met à jour les états possibles selon l'attribut choisi
        def mettre_a_jour_etat(root, liste_cellules, cellule_combobox, element_combobox, attribut_combobox, etat_combobox):
            attribut_nom = attribut_combobox.get()
            cellule_nom = cellule_combobox.get()
            cellule_obj = next((c for c in liste_cellules if c.nom == cellule_nom), None)
            if cellule_obj:
                elements = self.elements_possibles(cellule_obj)
                for element in elements:
                    if element.nom == element_combobox.get():
                        element_nom = element
            if attribut_nom == "clefs":
                clef_label = tk.Label(root, text="clef")
                clef_label.grid(row=0, column=3)
                #Si l'attribut est "clef", on ajoute un nouveau bandeau pour sélectionner la clef
                self.clef_combobox = ttk.Combobox(root, values=[clef.nom for clef in element_nom.clefs], width=15)  # Liste des clefs disponibles
                self.clef_combobox.grid(row=1, column=3)
                if self.clef_combobox["values"]:
                    self.clef_combobox.set(self.clef_combobox["values"][0])
                etat_combobox['values'] = ['prisonniere', 'presente', 'absente', 'penne rentre', 'penne sorti']
            else :
                if self.clef_combobox:
                    self.clef_combobox.grid_forget()  
                    self.clef_combobox.destroy()      
                    self.clef_combobox = None
                etats_possibles = self.etats_possible(attribut_nom)
                etat_combobox['values'] = etats_possibles

        attribut_combobox.bind("<<ComboboxSelected>>", lambda event: mettre_a_jour_etat(bandeau_frame, liste_cellules, cellule_combobox, element_combobox, attribut_combobox, etat_combobox))

        ajouter_button = tk.Button(root, text="Ajouter Condition", command=lambda : self.ajouter_condition(liste_cellules, cellule_combobox, element_combobox, operator_combobox, attribut_combobox, etat_combobox, condition_text))
        ajouter_button.pack()  
        
        supprimer_button = tk.Button(root, text="Supprimer la dernière condition", command=lambda: self.supprimer_derniere_condition(condition_text))
        supprimer_button.pack()
    
    #fonction qui cherche les positions des différentes sources et des différents smalts dans le tableau final
    def rechercher_sources_smalt(self):
        starts = []
        ends = []
        
        for i in range(len(self.tableau)):
            for j in range(len(self.tableau[i])):
                if self.tableau[i][j] == 'source':
                    starts.append((i, j))
                elif self.tableau[i][j] == 'smalt':
                    ends.append((i, j))
        
        return starts, ends

    #parcours de graphe en profondeur qui recherche s'il existe un chemin entre une source et un smalt
    #pour cela, on suppose qu'un chemin s'arrête lorsque l'on tombe sur un 0 dans la matrice
    def existe_chemin(self):
        self.starts, self.ends = self.rechercher_sources_smalt()
        if not self.starts or not self.ends:
            return False
        
        rows = len(self.tableau)
        cols = len(self.tableau[0])

        def dfs(start, end):
            stack = [(start, None)]  
            visited = set([start])  # Ensemble des positions déjà visitées
            parents = {start: None}  # Dictionnaire pour suivre le chemin
            
            directions = {
                'haut': (-1, 0),
                'bas': (1, 0),
                'gauche': (0, -1),
                'droite': (0, 1)
            }
                       
            #Fonction pour détecter un carrefour (les 4 directions sont possibles)
            def est_carrefour(x, y):
                #Vérifie si on a des chemins dans les 4 directions (haut, bas, gauche, droite)
                return ((self.tableau[x-1][y] != 0 and x-1 >=0) and  
                        (self.tableau[x+1][y] != 0 and x+1 < len(self.tableau)) and  
                        (self.tableau[x][y-1] != 0 and y-1 >=0) and  
                        (self.tableau[x][y+1] != 0 and y+1 < len(self.tableau[0])))     
        
            while stack:
                (x, y), direction_precedente = stack.pop()
                
                if (x, y) == end:
                    return True
                
                if not est_carrefour(x, y):
                    visited.add((x, y))
                
                #on explore les 4 directions
                for direction, (dx, dy) in directions.items():
                    nx, ny = x + dx, y + dy
                    
                    if 0 <= nx < rows and 0 <= ny < cols and self.tableau[nx][ny] != 0 and (nx, ny) not in visited:
                        
                        #Si on est à un carrefour, on ne peut continuer le chemin qu'en face (qu'en continuant tout droit)
                        if est_carrefour(x, y):
                            if direction_precedente != None and direction == direction_precedente:
                                stack.append(((nx, ny), direction))
                                parents[(nx, ny)] = (x, y)
                        else:
                            stack.append(((nx, ny), direction))
                            parents[(nx, ny)] = (x, y)
                        
        
            return False

        #Vérification des chemins entre toutes les sources et SMALTs trouvés
        for start in self.starts:
            for end in self.ends:
                chemin_trouve = dfs(start, end)
                if chemin_trouve:
                    print(f"Chemin trouvé entre {start} et {end}")
                    return True

        return False
    
    
    #on suppose qu'il y a un danger pour le matériel si il existe un chemin entre un smalt et une source
    def danger_materiel(self):
        return self.existe_chemin()
        
    #il y a un danger pour les personnes si un panneau d'accès aux câbles peut être ouvert sans être à la terre des deux côtés du câble
    def danger_personne1(self, liste_smalt, porte_acces):
        if liste_smalt != [] and porte_acces != []:
            return (porte_acces.etat == 'ouvert' and any(smalt.etat == 'ouvert' for smalt in liste_smalt))

     #il y a un danger pour les personnes si une éclisse peut être récupérée sans être sur les tableaux reliés à cette éclisse   
    def danger_personne2(self, liste_smalt, clef):
        if liste_smalt != []:
            return (clef.penne == 'rentre' and any(smalt.etat == 'ouvert' for smalt in liste_smalt))
        return False
        
    #construit la condition ajoutée pour qu'elle soit traduite en code
    def construire_expression(self):
        #Transforme la liste de conditions et opérateurs en une chaîne d'expression logique
        expression = ""
        
        for element in self.conditions:
            if isinstance(element, list):
                if len(element) == 4:
                    element_obj = element[0]
                    attribut = element[1]
                    operateur_python = element[2]
                    etat = element[3]
                    attribut_valeur = getattr(element_obj, attribut, None)

                
                    if operateur_python == "==":
                        condition_result = (attribut_valeur == etat)
                    elif operateur_python == "!=":
                        condition_result = (attribut_valeur != etat)
                    else:
                        raise ValueError(f"Opérateur inconnu: {operateur_python}")
                    
                    expression += f"{condition_result}"
                    
                else:
                    clef_obj = element[0]
                    operateur_python = element[1]
                    etat = element[2]
                    attribut = self.attribut_selon_etat[etat]
                    mots = etat.split()  
                    dernier_mot = mots[-1]
                    attribut_valeur = getattr(clef_obj, attribut, None)
                    
                    if operateur_python == "==":
                        condition_result = (attribut_valeur == dernier_mot)
                    elif operateur_python == "!=":
                        condition_result = (attribut_valeur != dernier_mot)
                    else:
                        raise ValueError(f"Opérateur inconnu: {operateur_python}")
                    
                    #Transforme la condition en chaîne de caractères
                    expression += f"{condition_result}"
            
            elif element in ["AND", "OR"]:
                #Ajoute les opérateurs logiques
                expression += f" {element.lower()} "
            
            elif element == "(":
                #Ajoute la parenthèse ouvrante
                expression += "("
            
            elif element == ")":
                #Ajoute la parenthèse fermante
                expression += ")"
        
        return expression
        
    #Cette fonction permet de lister les clés non utilisées
    def clefs_non_utilisees(self, liste_cellules):
        toutes_les_clefs = set()

        for cellule in liste_cellules:
            elements = self.elements_possibles(cellule)
            for element in elements:
                if hasattr(element, 'clefs'):
                    for clef in element.clefs:
                        toutes_les_clefs.add((clef.nom, element.nom))

        non_utilisees = toutes_les_clefs - self.clefs_utilisees
        return list(non_utilisees)

    #Cette fonction enregistre les clés utilisées
    def enregistrer_clef_utilisee(self, clef, element):
        if clef and hasattr(clef, "nom") and element and hasattr(element, "nom"):
            self.clefs_utilisees.add((clef.nom, element.nom))

    #cette fonction vérifie si une des conditions est vérifiée et returne un message de danger le cas échéant : c'est ce message qui sera affiché en gros sur le dessin
    def verifier_conditions(self, liste_cellules):
        
        liste_smalt = ConditionsFin.construire_dictionnaire_smalt(liste_cellules)
        
        #regarde si la condition ajoutée est vérifiée
        expression = self.construire_expression()
        if expression != "":
            if eval(expression) == True:
                print(f"Danger ! La condition ajoutee est verifiee")

                non_utilisees = self.clefs_non_utilisees(liste_cellules)
                for clef, element in non_utilisees:
                    print(f"Clé non utilisée : {clef} (élément : {element})")

                return "Danger ! La condition ajoutee est verifiee"
        
        if self.danger_materiel():
            print("Danger détecté pour le materiel ! Arrêt immédiat.")
            print("Il existe un chemin entre une source et un smalt")

            non_utilisees = self.clefs_non_utilisees(liste_cellules)
            for clef, element in non_utilisees:
                print(f"Clé non utilisée : {clef} (élément : {element})")

            return "Danger détecté pour le materiel ! Arrêt immédiat."
        
        for cellule in liste_cellules:
            if cellule.porte and cellule.nom in liste_smalt and self.danger_personne1(liste_smalt[cellule.nom], cellule.porte):  
                print(f"Danger détecté pour les personnes au niveau de la cellule {cellule.nom} ! Arrêt immédiat.")
                print("Un panneau d'accès aux câbles peut être ouvert sans être à la terre des deux côtés du câble")

                non_utilisees = self.clefs_non_utilisees(liste_cellules)
                for clef, element in non_utilisees:
                    print(f"Clé non utilisée : {clef} (élément : {element})")

                return "Danger détecté pour les personnes car on peut ouvrir une porte sans avoir tout mis a la terre ! Arrêt immédiat."
            
            #si la cellule possède une éclisse, on récupère la clef permettant de la verrouiller et on regarde si il y a un danger
            elif cellule.eclisse : 
                clef = cellule.eclisse.clefs[0]
                if cellule.nom in liste_smalt:
                    if self.danger_personne2(liste_smalt[cellule.nom], clef):
                        return "Danger détecté pour les personnes car on peut prendre une eclisse sans avoir tout mis à la terre ! Arrêt immédiat."                 
            
        return False
 