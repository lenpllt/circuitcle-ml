#on importe la classe ElementElectrique classe mere de SerrureMere 
from LHC_Classe_element_elec import ElementElectrique

#creation de la classe SerrureMere : cree des objets serrure mere
class SerrureMere(ElementElectrique):
    
    #caracteristiques d'un objet serrure mere
    def __init__(self, nom, clefs, cellule, type):
        super().__init__(nom, clefs, cellule, type) #caractéristiques communes à tout élément électrique

    #fonction qui libère les clefs prisonnières de le serrure mère si les conditions le permettent    
    def clefs_serrure_mere(self):
        liste_clefs_absentes = []
        liste_clefs_prisonnieres = []
        compte_clefs_absentes = {} #dictionnaire recensant l'occurence de chaque clef
        
        #on parcourt chaque clef, met à jour les variables et on regarde si les clefs absentes sont bien toutes disponibles pour l'exécution de l'étape
        for clef in self.clefs:
            clef_nom = clef.nom
            if clef.etat == 'absente':
                if clef_nom in compte_clefs_absentes:
                    compte_clefs_absentes[clef_nom] += 1
                else:
                    compte_clefs_absentes[clef_nom] = 1
                nombre_fois_libre = self.clefs_libres.count(clef_nom)
                if clef_nom not in self.clefs_libres or  compte_clefs_absentes[clef_nom] > nombre_fois_libre: #permet d'être sur que si on a besoin de deux mêmes clefs, on l'a bien deux fois
                    return False
                else :
                    liste_clefs_absentes.append(clef)
            elif clef.etat == 'prisonniere' :
                liste_clefs_prisonnieres.append(clef)                 
        
        #si on arrive ici, la fonction n'a pas retourne False donc on peut libérer les clefs prisonnières de la serrure (et rendre prisonnières les autres) 
        for clef in liste_clefs_absentes: 
            clef_nom = clef.nom
            self.clefs_libres.remove(clef_nom)
            self.changer_etat_clef(clef_nom, 'prisonniere')
        for clef in liste_clefs_prisonnieres: 
            clef_nom = clef.nom
            self.clefs_libres.append(clef_nom)
            self.changer_etat_clef(clef_nom, 'absente')
        print("deverrouillage clefs serrure mere", self.nom)

        return True
