#on importe la classe ElementElectrique classe mere de SMALT 
from LHC_Classe_element_elec import ElementElectrique

#creation de la classe SMALT : cree des objets smalt
class SMALT(ElementElectrique):
    
     #caracteristiques d'un objet smalt
    def __init__(self, nom,type, clefs, etat, cellule, representation, deplacement, appartenance):
        super().__init__(nom, clefs, cellule, type) #caracts communes a tout element elec
        self.deplacement = deplacement
        self.etat = etat
        self.representation = representation
        self.appartenance = appartenance
        
    #retourne True si la contrainte mecanique sur la porte est presente 
    def contraintes_mecaniques_porte(self):
        return self.etat == 'ouvert'

    #fonction permettant d'ouvrir ou fermer le smalt si les contraintes (mecaniques et de verrouillage) le permettent
    def ouverture_fermeture_SMALT(self, disjoncteur, porte,  cellule):
        
        #la première condition pour pouvoir continuer est de ne pas avoir de contraintes mecaniques
        if disjoncteur == []  or not disjoncteur.contraintes_mecaniques_SMALT() and porte.etat == 'ferme' :
            M = self.representation  #représentation du verrouillage à clefs du smalt
            x = self.position_element(cellule.matrice)[0] #position x (abscisse) du smalt dans la matrice représentant la cellule
            y = self.position_element(cellule.matrice)[1] #position y (ordonnee) du smalt dans la matrice représentant la cellule
            
            #si le smalt n'est pas vérrouillé, on peut executer l'etape et realiser l'ouverture/fermeture
            if not self.verifier_penne_sorti():
                if self.etat == 'ouvert': 
                    self.etat = 'ferme'
                    cellule.matrice[x][y - 1] = 'fil_h' #la fermeture du smalt est representée par l'ajout d'un fil connectant le smalt au reste de la cellule
                    print("fermeture", self.nom, self.cellule)
                elif self.etat == 'ferme': 
                    self.etat = 'ouvert'
                    cellule.matrice[x][y - 1] = 0 #on supprime le fil connectant le smalt au reste de la cellule
                    print("ouverture", self.nom, self.cellule)

                #on decale la representation du verrouillage selon la direction désignée par la caractéristique déplacement
                #les clefs elle ne bouge pas de position ce qui fait qu'elles deviennent libres si elles etaient face a un blocageet vice-versa : elles changent d'etat
                self.representation = self.decaler_matrice()
                self.changer_deplacement() #indique dans quel sens la representation du verrouillage va se deplacer lorsque l'on réalisera à nouveau l'étape

                #on modifie l'etat des clefs en consequence
                for i in range(len(M)):
                    for j in range(len(M[i])):
                        #on regarde chaque clef dans la matrice et si au dessus de sa penne il y a un blocage (prisonnière) ou un creux (présente)
                        clef_nom = M[i][j]
                        if clef_nom in self.verifier_penne_rentre():
                            if j > 0 and M[i][j - 1] == 1:
                                self.changer_etat_clef(clef_nom, 'prisonniere')
                            if j > 0 and (M[i][j - 1] == 2 or M[i][j - 1] == 0):
                                self.changer_etat_clef(clef_nom, 'presente')
                return True
            
        #si une penne est sortie ou les contraintes mecaniques emphechent la réalisation de l'étape, on marque l'echec de l'etape
        return False

    #fonction permettant de verrouiller ou de déverrouiller le smalt si les conditions (contraintes mécaniques et de verrouillage) le permettent
    def verrouillage_deverrouillage_SMALT(self):
        
        #on stocke la liste des clefs sorties
        clefs_sorties = self.verifier_penne_sorti()
        
        # Si la liste des clefs sorties est non nulle, on procéde au déverrouillage des clefs disponibles : on rentre leur pêne
        if clefs_sorties != []:
            for clef_nom in clefs_sorties:
                if clef_nom in self.clefs_libres:
                    self.changer_etat_penne(clef_nom, 'rentre')
                    self.changer_etat_clef(clef_nom, 'presente')
                    self.clefs_libres.remove(clef_nom) 
                    clefs_sorties.remove(clef_nom) 
                    
            #si a la fin plus aucun pêne n'est sorti, on a déverrouillé le smalt : on affiche un message et renvoie True
            if clefs_sorties == []:
                print("Deverrouillage", self.nom, self.cellule, "de la position", self.etat)
                return True
            
        #si aucune clef n'est sortie, on vérifie si on peut verrouiller : pour cela, il faut simplement pouvoir sortir un pêne
        else:
            #dans la représentation du verrouillage, on regarde si une clef est présente pêne rentré : dans ce cas, on sort le pêne
            verrouillage = False
            M = self.representation
            for i in range(len(M)):
                for j in range(len(M[i])):
                    clef_nom =M[i][j]
                    if clef_nom in self.verifier_penne_rentre():
                        clef = self.chercher_clef_par_nom(clef_nom) #récupère l'objet clef avec ses caractéristiques (nom, état et penne) à partir de son nom seulement
                        if j > 0 and M[i][j - 1] == 2 and  clef.etat == "presente": 
                            self.changer_etat_penne(clef_nom, 'sorti')
                            self.changer_etat_clef(clef_nom, 'absente')
                            self.clefs_libres.append(clef_nom)
                            verrouillage = True #variable qui permet de savoir qu'on a pu verrouiller tout en laissant la possibilité de verrouiller plusieurs clefs à la fois
                            
            if verrouillage: #si on a pu verrouiller au moins une clef
                 print(f"Verrouillage position {self.etat}, {self.nom}, {self.cellule}")
                 return True
        #on renvoie False si on a ni verrouillé ni déverrouillé

        return False
