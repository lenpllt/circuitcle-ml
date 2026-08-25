#on importe la classe ElementElectrique classe mere de Panneau 
from LHC_Classe_element_elec import ElementElectrique

#creation de la classe panneau : cree des objets panneau
class Panneau(ElementElectrique):
    
    #caracteristiques d'un objet panneau
    def __init__(self, nom, type, clefs, cellule, etat):
        super().__init__(nom, clefs, cellule, type) #caractéristiques communes a tout élément électrique
        self.etat = etat
        
    #fonction permettant d'ouvrir ou de fermer le panneau si les conditions le permettent
    def ouverture_fermeture_panneau(self, smalt):

        #la première condition est de ne pas avoir de contrainte mécanique sur la porte
        if not smalt.contraintes_mecaniques_porte():
            #on liste les clefs absentes et prisonnieres via une fonction de la classe ElementElectrique
            liste_clefs_absentes = self.clef_absente()[0]
            liste_clefs_prisonnieres = self.clef_prisonniere()[0]
            
            for clef in liste_clefs_absentes: 
                clef_nom = clef.nom
                if clef_nom not in self.clefs_libres: #si la clef est absente du panneau et n'est pas disponible, on ne pourra pas l'ouvrir
                    return False                  
            
            #si on arrive ici, toutes les clef absentes sont disponibles : on peut ouvrir ou fermer le panneau 
            for clef in liste_clefs_absentes: #on change l'état des clefs absentes
                clef_nom = clef.nom
                self.clefs_libres.remove(clef_nom)
                self.changer_etat_clef(clef_nom, 'prisonniere')
            for clef in liste_clefs_prisonnieres: #on change l'état des clefs prisonnières
                clef_nom = clef.nom
                self.clefs_libres.append(clef_nom)
                self.changer_etat_clef(clef_nom, 'absente')
            if self.etat == 'ouvert':
                self.etat = 'ferme'
                print("Fermeture", self.nom, self.cellule)
            elif self.etat == 'ferme':
                self.etat = 'ouvert'
                print("Ouverture", self.nom, self.cellule) 
            return True
