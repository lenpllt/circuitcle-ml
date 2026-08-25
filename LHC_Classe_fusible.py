#on importe la classe ElementElectrique car fusible est une classe fille de celle-ci 
from LHC_Classe_element_elec import ElementElectrique

#classe Fusible : cree des objets fusible
class Fusible(ElementElectrique):
    
    #initialise toutes les caracteristiques que doit avoir un fusible
    def __init__(self, nom, type, clefs, cellule):
        super().__init__(nom, clefs, cellule, type) #caractéristiques que doit avoir tout élément électrique
        