#on importe la classe ElementElectrique car coffret est une classe fille de celle-ci
from LHT_Classe_element_elec import ElementElectrique

#classe coffret : cree des objets coffret
class Coffret(ElementElectrique):
    
    #regarde si le controle commande sur LHT est autorise : concerne le coffret 0LHT110AR (avec la clef XP)
    autorisation_CC = True
    
    #initialise toutes les caracteristiques que doit avoir un coffret
    def __init__(self, nom, type, clefs, cellule):
        super().__init__(nom, clefs, cellule, type) #caractéristiques de la classe mère
        
    #réalise l'etape d'ouverture ou de fermeture d'un coffret si les conditions permettent la réalisation
    def ouverture_fermeture_coffret(self, cellule):
        #on liste les clefs absentes et prisonnieres via une fonction de la classe ElementElectrique
        liste_clefs_absentes = self.clef_absente()[0]
        liste_clefs_prisonnieres = self.clef_prisonniere()[0]
        
        for clef in liste_clefs_absentes: 
            clef_nom = clef.nom
            if clef_nom not in self.clefs_libres: #si la clef est absente du coffret et n'est pas disponible, on ne pourra pas l'ouvrir
                return False                  
        
        #si on arrive ici, toutes les clef absentes sont disponibles : on peut ouvrir ou fermer le coffret 
        for clef in liste_clefs_absentes: #on change l'état des clefs absentes
            clef_nom = clef.nom
            self.clefs_libres.remove(clef_nom)
            self.changer_etat_clef(clef_nom, 'prisonniere')
        for clef in liste_clefs_prisonnieres: #on change l'état des clefs prisonnières
            clef_nom = clef.nom
            self.clefs_libres.append(clef_nom)
            self.changer_etat_clef(clef_nom, 'absente')
        print("deverrouillage clefs coffret", self.nom)
        
        #on regarde si le coffret est le 0LHT110AR : dans ce cas, l'ouverture ou la fermeture de ce coffret autorise ou non le CC de LHT
        if self.type == "CC": 
            #on suppose que les sources sont toujours placées au début de la matrice, soit en position (0,2)
            x = 0
            y = 2
            #si la clef du coffret est absente, alors le CC est en mode interdiction : le diesel ne peut pas démarrer
            if self.clefs[0].etat == 'absente':
                autorisation_CC= False
                cellule.matrice[x][y] = 0 #on cache la source car elle ne pourra pas demarrer
                
            else:
                autorisation_CC = True
                cellule.matrice[x][y] = 'source'
        
        return True