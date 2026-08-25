#on importe la classe ElementElectrique classe mere de BoiteEclisse : BoiteEclisse est un element de cette classe
from LHT_Classe_element_elec import ElementElectrique
from LHT_Classe_eclisse import Eclisse #la boite a eclisse est liee au nombre d'eclisses, donc on importe la classe Eclisse car on l'utilisera ici

#creation de la classe BoiteEclisse : cree des objets boite eclisse
class BoiteEclisse(ElementElectrique):
    
    #caracteristiques d'un objet boite eclisse
    def __init__(self, nom, clefs, type, cellule, voie, stock_eclisses, position, representation, deplacement, appartenance): 
        super().__init__(nom, clefs, cellule, type) #caracts communes a tout element elec
        self.voie = voie
        self.stock_eclisses = stock_eclisses
        self.position = position
        self.representation = representation
        self.deplacement = deplacement
        self.appartenance = appartenance
        
     #fonction permettant d'ouvrir la boite pour récupérer les eclisses ou de la fermer : en gros on recupere les eclisses ou on les depose   
    def ouverture_fermeture_boite(self, cellule):
        
        M = self.representation #représentation du verrouillage à clefs de la boite
        x = self.position_element(cellule.matrice)[0] #position x (abscisse) de la boite dans la matrice représentant la cellule
        y = self.position_element(cellule.matrice)[1] #position y (ordonnee) de la boite dans la matrice représentant la cellule
        
        #Si aucune penne n'est sortie, la boite n'est pas verrouillée : on peut realiser l'ouverture ou la fermeture de la boite à eclisses et insérer ou récupérer les éclisses
        if self.verifier_penne_sorti() == [] :
            
            if self.position == 'embroche': #si on est embroche : 
                self.position = 'debroche' #on débroche
                cellule.matrice[x][y + 1] = self.type #la partie mobile se décale d'un cran representant l'ouverture de la boite
                cellule.matrice[x][y] = 0
                
                #on distingue si on est en voie A ou B
                if self.voie == 'A': #si on est en voie A, alors:
                    #On insère les éclisses voie A dans la boite à éclisses ou on récupère les éclisses de la boite voie A
                    if self.stock_eclisses > 0: #si la boite contient des eclisses, alors : 
                        Eclisse.eclisseA = self.stock_eclisses #on les récup^ère, Eclisse.eclisseA representant le nombre d'eclisse voie A disponibles
                        self.stock_eclisses = 0 #la boite est vidée de ses éclisses en voie A
                    elif Eclisse.eclisseA > 0: #si des éclisses voie A sont disponibles, alors : 
                        self.stock_eclisses = Eclisse.eclisseA
                        Eclisse.eclisseA = 0
                
                elif self.voie == 'B':
                    if self.stock_eclisses > 0:
                        Eclisse.eclisseB = self.stock_eclisses
                        self.stock_eclisses = 0
                    elif Eclisse.eclisseB > 0:
                        self.stock_eclisses = Eclisse.eclisseB
                        Eclisse.eclisseB = 0

                #on affiche la realisation de l'ouverture
                print("ouverture", self.nom, self.cellule)
                
            elif self.position == 'debroche': #si on est debroche, on embroche
                self.position = 'embroche'
                cellule.matrice[x][y-1] = self.type
                cellule.matrice[x][y] = 0

                #on affiche la realisation de la fermeture
                print("fermeture", self.nom, self.cellule)
            
            #on decale la representation du verrouillage selon la direction désignée par la caractéristique déplacement
            #les clefs ne bougent pas de position ce qui fait qu'elles deviennent libres si elles etaient face a un blocage et vice-versa : elles changent d'etat
            self.representation = self.decaler_matrice().representation
            self.changer_deplacement() #indique dans quel sens la representation du verrouillage va se deplacer
            
            #on modifie l'etat des clefs en consequence
            for i in range(len(M)):
                for j in range(len(M[i])):
                    #on regarde chaque clef dans la matrice et si au dessus de sa penne il y a un blocage ou un creux. Cela détermine l'état prisonnière ou présente de la clef
                    clef_nom = M[i][j]
                    if clef_nom in self.verifier_penne_rentre():
                        if M[i - 1][j] == 1 :
                            self.changer_etat_clef(clef_nom, 'prisonniere')
                        if M[i - 1][j] == 2:
                            self.changer_etat_clef(clef_nom, 'presente')
                            
            #on renvoie True car l'étape s'est exécutée
            return True
        
        # Retourne False si le processus n'a pas pu se réaliser
        return False
     
    #fonction permettant de verrouiller ou de déverrouiller le la boite si les conditions (contraintes mécaniques et de verrouillage) le permettent
    def verrouillage_deverrouillage_tiroir(self):
        
       clefs_sorties = self.verifier_penne_sorti() #on stocke la liste des clefs dont le pêne est sorti

       # Si la liste des clefs sorties est non nulle, on procéde au déverrouillage des clefs disponibles
       if clefs_sorties != []:
           for clef_nom in clefs_sorties[:]:
               #pour chauqe clef, on regarde sur elle est libre et on rentre son pêne le cas échéant
               if clef_nom in self.clefs_libres: 
                   self.changer_etat_penne(clef_nom, 'rentre')
                   self.changer_etat_clef(clef_nom, 'presente')
                   self.clefs_libres.remove(clef_nom) #la clef n'est pas libre
                   clefs_sorties.remove(clef_nom) #et le pêne n'est plus sorti
                   
           #si a la fin plus aucune penne n'est sortie, on a déverrouille le tiroir : on affiche un message et renvoie True
           if clefs_sorties == []:
               print(f"Déverrouillage position {self.position}, {self.nom}, {self.cellule}")
               return True

       # Si aucune clef n'est sortie, on vérifie si le verrouillage de la boite
       else:
           #dans la représentation du verrouillage, on regarde si une clef et présente pêne rentré : le cas échéant, on sort la penne
           verrouillage = False
           M = self.representation  #représentation de la serrurerie via la matrice      
           for i in range(len(M)):
               for j in range(len(M[i])):
                   if M[i][j] in self.verifier_penne_rentre():
                       clef_nom = M[i][j]
                       clef = self.chercher_clef_par_nom(clef_nom)
                       if j > 0 and M[i-1][j] in [0,2] and clef.etat == "presente": #on regarde si la clef est présente et que le pêne n'est pas bloqué
                           self.changer_etat_penne(clef_nom, 'sorti')
                           self.changer_etat_clef(clef_nom, 'absente')
                           self.clefs_libres.append(clef_nom)
                           verrouillage = True #permet de savoir qu'on a pu verrouiller mais permet de verrouiller plusieurs clefs à la fois si possible
                           
           #on a verrouillé au moins une clef donc on affiche un message et renvoie True        
           if verrouillage:
                print(f"Verrouillage position {self.position}, {self.nom}, {self.cellule}")
                return True
       #on renvoie False si on a ni pû verrouiller, ni déverrouiller
       return False
   
        
