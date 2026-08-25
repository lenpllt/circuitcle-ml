#on importe la classe ElementElectrique classe mere de Transformateur 
from LHT_Classe_element_elec import ElementElectrique

#creation de la classe Transformateur : cree des objets transformateur
class Transformateur(ElementElectrique):
    
    #caracteristiques d'un objet transformateur
    def __init__(self, nom, type, clefs, cellule):
        super().__init__(nom, clefs, cellule, type)  #caractéristiques communes à tout element électrique

    
      
    #réalise l'étape de consignation ou déconsignation du transformateur si les contraintes le permettent
    def consignation_deconsignation_transformateur(self):
        #on liste les clefs absentes et prisonnieres 
        liste_clefs_absentes = self.clef_absente()[0]
        liste_clefs_prisonnieres = self.clef_prisonniere()[0]
        
        #on regarde si une des clefs absentes du transformateur n'est pas disponible : dans ce cas, l'étape ne peut pas etre executée
        for clef in liste_clefs_absentes: 
            clef_nom = clef.nom
            if clef_nom not in self.clefs_libres: 
                return False                  
        
        #si on arrive ici, la fonction n'a pas retournée False donc on peut realiser l'etape
        for clef in liste_clefs_absentes: 
            clef_nom = clef.nom
            self.clefs_libres.remove(clef_nom) 
            self.changer_etat_clef(clef_nom, 'prisonniere')
        for clef in liste_clefs_prisonnieres: 
            clef_nom = clef.nom
            self.clefs_libres.append(clef_nom)
            self.changer_etat_clef(clef_nom, 'absente')
        print("deverrouillage clefs", self.nom)
        return True
        
  


