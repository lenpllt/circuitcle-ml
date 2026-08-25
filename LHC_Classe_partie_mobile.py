#On importe la classe ElemlentElectrique car c'est la classe mere de PartieMobile 
from LHC_Classe_element_elec import ElementElectrique

#Creation d'une classe partie mobile, qui permettra de créer les objets parties mobiles 
class PartieMobile(ElementElectrique):
    
    #caractéristiques d'un objet partie mobile
    def __init__(self, nom,clefs, type, etat, cellule, representation, deplacement, position, appartenance): 
        super().__init__(nom, clefs, cellule, type) #caracteristiques de la classe mere
        self.deplacement = deplacement
        self.position = position
        self.etat = etat
        self.representation = representation
        self.appartenance = appartenance
        self.manivelle_inseree = False

    #retourne True si les contraintes mecaniques s'exercent sur le tiroir de la partie mobile : dans ce cas, on ne pourra pas réaliser l'étape     
    def contraintes_mecaniques_tiroir(self, SMALT):
        return (self.etat == 'ferme' or SMALT.etat == 'ferme') 

    #retourne True si les contraintes mecaniques s'exercent sur le smalt : dans ce cas, on ne pourra pas fermer le smalt
    def contraintes_mecaniques_SMALT(self):
        return self.position == 'embroche' #retourne True si la partie mobile est embrochée, False sinon
    
    #fonction permettant d'embrocher ou de débrocher la partie mobile si les conditions (contraintes mécaniques et de verrouillage) le permettent
    def embrochage_debrochage(self, SMALT, cellule):
        #condition pour continuer : ne pas avoir de contraintes mécaniques sur la partie mobile  
        if SMALT == [] or not self.contraintes_mecaniques_tiroir(SMALT): 
            #la premiere condition est vérifiée
            M = self.representation #on recupere la matrice représentant la serrurerie de la partie mobile
            x = self.position_element(cellule.matrice)[0] #position x (abscisse) de la partie mobile dans la matrice représentant la cellule
            y = self.position_element(cellule.matrice)[1] #position y (ordonnée) de la partie mobile dans la matrice représentant la cellule
            
            #deuxième condition pour continuer : tiroir déverrouillé (pas de pêne sorti) et manivelle insérée
            if self.verifier_penne_sorti() == [] and self.manivelle_inseree == True: 
                if self.position == 'embroche': 
                    self.position = 'debroche' 
                    cellule.matrice[x][y + 1] = self.type #on decale la partie mobile d'un cran vers la droite, synonyme du debrochage
                    cellule.matrice[x][y] = 0 #l'emplacement precedent de la partie mobile devient vide
                    print("débrochage", self.nom, self.cellule) 
                    
                elif self.position == 'debroche': 
                    self.position = 'embroche' 
                    cellule.matrice[x][y-1] = self.type #on decale la partie mobile d'un cran vers la gauche, synonyme de l'embrochage
                    cellule.matrice[x][y] = 0 #l'emplacement precedent de la partie mobile devient vide
                    print("embrochage", self.nom, self.cellule) 
                    
                #on decale la representation du verrouillage selon la direction désignée par la caractéristique déplacement
                #ette fonction decale seulement les elements mecaniques (blocages, creux) de la serrurerie, changeant ainsi l'etat des clefs
                self.representation = self.decaler_matrice()
                
                self.changer_deplacement() #si on s'est décalé vers la droite (débrochage), change le deplacement pour gauche (embrochage)
                
                #traite la modification de l'état des clefs grâce la représentation du verrouillage
                for i in range(len(M)): 
                    for j in range(len(M[i])): 
                        #on regarde chaque clef dans la matrice et si au dessus de son pêne il y a un blocage (prisonnière) ou un creux (présente)
                        clef_nom = M[i][j] 
                        if clef_nom in self.verifier_penne_rentre(): 
                            if M[i - 1][j] == 1 :
                                self.changer_etat_clef(clef_nom, 'prisonniere') 
                            if M[i - 1][j] == 2:
                                self.changer_etat_clef(clef_nom, 'presente')                                 
                return True
        
        # Retourne False si l'etape n'a pas pu se réaliser
        return False
    
    #fonction permettant de verrouiller ou de déverrouiller le tiroir si les conditions (contraintes mécaniques et de verrouillage) le permettent
    def verrouillage_deverrouillage_tiroir(self):
        
        clefs_sorties = self.verifier_penne_sorti() #stocke la liste des clefs dont le pêne est sorti

        if clefs_sorties != []: #si la liste des clefs pêne sorti est non vide, alors on va essayer de verrouiller le tiroir
           for clef_nom in clefs_sorties: 
               if clef_nom in self.clefs_libres: 
                   self.changer_etat_penne(clef_nom, 'rentre') 
                   self.changer_etat_clef(clef_nom, 'presente') 
                   self.clefs_libres.remove(clef_nom) 
                   clefs_sorties.remove(clef_nom) 

            #si la liste des clefs pêne sorti est maintenant vide, alors on pas déverrouillé le tiroir
           if clefs_sorties == []: 
               print(f"Déverrouillage position {self.position}, {self.nom}, {self.cellule}") 
               return True 
           
        else: #dans le cas où la liste des clefs sorties est nulle, on va regarder l'inverse : est-il possible de verrouiller ?
           verrouillage = False #cette variable permet de savoir si on a verrouillé 
           M = self.representation #matrice représentant la serrurerie de la partie mobile        
           for i in range(len(M)): 
               for j in range(len(M[i])): 
                   clef_nom = M[i][j] 
                   if M[i][j] in self.verifier_penne_rentre(): 
                       clef = self.chercher_clef_par_nom(clef_nom) 
                       if M[i-1][j] in [0,2] and clef.etat == "presente": 
                           self.changer_etat_penne(clef_nom, 'sorti') 
                           self.changer_etat_clef(clef_nom, 'absente') 
                           self.clefs_libres.append(clef_nom) 
                           verrouillage = True 
           
           #une fois qu'on a finit de parcourir toutes les clefs, on regarde on regarde si on en a verrouillé au moins une
           if verrouillage: 
                print(f"Verrouillage position {self.position}, {self.nom}, {self.cellule}") 
                return True 
           
        return False #on renvoie False si on a ni verrouillé ni déverrouillé
   
    #fonction permettant de d'insérer ou sortir la manivelle si les conditions de verrouillage le permettent
    def insertion_extraction_manivelle(self):
        clefs_sorties = self.verifier_penne_sorti() #stocke la liste des clefs dont le pêne est sorti
        
        #conditions pour pouvoir insérer la manivelle : ne pas avoir de pêne sorti et qu'aucune manivelle ne soit pour l'instant insérée
        if clefs_sorties == [] and self.manivelle_inseree == False:
            #dans la representation de la serrurerie, on rajoute une ligne entiere de 4 representant la manivelle
            nouvelle_ligne = [4 for _ in range(len(self.representation[0]))] 
            self.representation.insert(1, nouvelle_ligne) 
            self.manivelle_inseree = True 
            for clef in self.clefs: 
                clef.changer_etat('prisonniere') 
            print("manivelle inserree pour", self.position, self.nom, self.cellule) 
            return True 
        
        #conditions pour pouvoir extraire la manivelle : ne pas avoir de pêne sorti et avoir la manivelle actuellement insérée
        elif clefs_sorties == [] and self.manivelle_inseree == True: 
            self.representation.pop(1) #on supprime la ligne de 4 représentant la manivelle dans la matrice
            self.manivelle_inseree = False 
            M = self.representation 
            #l'etat des clefs redevient celui d'avant l'insertion de la manivelle
            for i in range(len(M)): 
                for j in range(len(M[i])): 
                    clef_nom = M[i][j] 
                    if clef_nom in self.verifier_penne_rentre(): 
                        if M[i - 1][j] == 1 : 
                            self.changer_etat_clef(clef_nom, 'prisonniere') 
                        if M[i - 1][j] != 1: 
                            self.changer_etat_clef(clef_nom, 'presente') 
                            
            print("manivelle extraite", self.nom, self.cellule, "en position", self.position)
            return True 

        return False #on renvoie false si on a pu ni inserer, ni extraire la manivelle
