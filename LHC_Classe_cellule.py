#classe cellule : cree des objets cellules
class Cellule:
    
    #initialise toutes les caracteristiques que doit avoir une cellule
    def __init__(self, nom, matrice, voisine, position_x, partie_mobile, smalt, serrures, porte, transformateur, longueur, coffret, fusible, largeur):
        self.nom = nom
        self.matrice = matrice
        self.partie_mobile = partie_mobile
        self.smalt = smalt 
        self.serrures = serrures
        self.porte = porte 
        self.voisine = voisine
        self.transformateur = transformateur
        self.coffret = coffret
        self.fusible=fusible
        self.longueur = longueur
        self.largeur = largeur
        self.position_x = position_x
    
    #permet de chercher une cellule via son nom parmi une liste de cellules
    @staticmethod
    def chercher_cellule_par_nom(liste_cellules, cellule_nom):
        for cellule in liste_cellules: 
            if cellule.nom == cellule_nom: 
                return cellule
        return None 
    
    #regroupe les cellules LHC reliées avec des cellules d'un autre tableau. Exemple : LHC003JA et LG pour le CPY
    def cellules_communes(self, liste_cellules):
        liste = []
        #on regarde les cellules de la liste qui ont la meme position_x, c'est-à-dire affichées dans une m^me colonne l'une en dessous de l'autre
        for cellules in liste_cellules:
            if cellules.position_x == self.position_x:
                liste.append(cellules.nom)
        return liste
     