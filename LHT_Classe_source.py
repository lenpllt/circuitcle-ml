#on importe la classe ElementElectrique classe mere de Source
from LHT_Classe_element_elec import ElementElectrique

#creation de la classe Source : cree des objets source
class Source(ElementElectrique):
      
     #caracteristiques d'un objet source
    def __init__(self, nom, type, clefs, cellule):
        super().__init__(nom, clefs, cellule, type) #caractéristiques identiques à tout élément électrique
        