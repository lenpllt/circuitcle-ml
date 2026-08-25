#on importe la classe ElementElectrique car coffret est une classe fille de celle-ci
from LHC_Classe_element_elec import ElementElectrique

#classe coffret : cree des objets coffret
class Coffret(ElementElectrique):
    
    #initialise toutes les caracteristiques que doit avoir un coffret
    def __init__(self, nom, type, clefs, cellule):
        super().__init__(nom, clefs, cellule, type) #caractéristiques de la classe mère
        
    #réalise l'etape d'ouverture ou de fermeture d'un coffret si les conditions permettent la réalisation
    def ouverture_fermeture_coffret(self):
        #on liste les clefs absentes et prisonnieres via une fonction de la classe ElementElectrique
        liste_clefs_absentes = self.clef_absente()[0]
        liste_clefs_prisonnieres = self.clef_prisonniere()[0]
        
        for clef in liste_clefs_absentes: 
            clef_nom = clef.nom
            if clef_nom not in self.clefs_libres: #si la clef est absente du coffret et n'est pas disponible, on ne pourra pas l'ouvrir
                return False                  
        
        #si on arrive ici, toutes les clef absentes sont disponibles : on peut ouvrir ou fermer le coffret
        for clef in liste_clefs_absentes: #les clefs absentes deviennent prisonnieres
            clef_nom = clef.nom
            self.clefs_libres.remove(clef_nom)
            self.changer_etat_clef(clef_nom, 'prisonniere')
        for clef in liste_clefs_prisonnieres: #les clefs prisonnieres deviennent absentes
            clef_nom = clef.nom
            self.clefs_libres.append(clef_nom)
            self.changer_etat_clef(clef_nom, 'absente')
        print("deverrouillage clefs coffret", self.nom)
        return True
    