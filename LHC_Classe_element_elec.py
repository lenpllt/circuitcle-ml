#classse ElementElectrique : classe mère : chaque classe d'un élément électrique créé (smalt par exemple) pourra utiliser ce qui appartient à cette classe mère
class ElementElectrique:
    
    #cette liste recense les clefs libres à chaque étapes : les clefs libres sont initialisées dans le tableau Excel
    clefs_libres = []    
    
    #chaque élément électrique devra avoir ces caractéristiques
    def __init__(self, nom, clefs, cellule, type):
        self.nom = nom
        self.clefs = [clef for clef in clefs]
        self.cellule = cellule
        self.type = type
    
    #décale la representation du verrouillage (pour smalt ou partie mobile) quand on leur position évolue(embrochage, fermeture du smalt...)
    def decaler_matrice(self): #sera utilisée seulement par les élément ayant une serrurerie (partie mobile, smalt)
        direction = self.deplacement
        M = self.representation  #représente la serrurerie de l'élément électrique  
        if direction == 'droite':
            for i in range(len(M)):
                # Parcourir chaque élément de la ligne de droite à gauche
                for j in range(len(M[i]) - 2, -1, -1):
                    # Si l'élément est 1 (blocage) ou 2 (creux), le décaler à droite
                    if M[i][j] in [1,2]:
                        M[i][j + 1] = M[i][j]
                        M[i][j] = 0

        elif direction == 'gauche':
            for i in range(len(M)):
                # Parcourir chaque élément de la ligne de gauche à droite
                for j in range(1, len(M[i])):
                    # Si l'élément est 1 ou 2, le décaler à gauche
                    if M[i][j] in [1,2]:
                        M[i][j - 1] = M[i][j]
                        M[i][j] = 0

        elif direction == 'bas':
            for j in range(len(M[0])):
                # Parcourir chaque élément de la colonne du bas vers le haut
                for i in range(len(M) - 2, -1, -1):
                    # Si l'élément est 1 ou 2, le décaler vers le bas
                    if M[i][j] in [1,2]:
                        M[i + 1][j] = M[i][j]
                        M[i][j] = 0

        elif direction == 'haut':
            for j in range(len(M[0])):
                # Parcourir chaque élément de la colonne du haut vers le bas
                for i in range(1, len(M)):
                    # Si l'élément est 1 ou 2, le décaler vers le haut
                    if M[i][j] in [1,2]:
                        M[i - 1][j] = M[i][j]
                        M[i][j] = 0

        return self.representation
    
    #prend la direction oppposée à celle que l'on vient d'effectuer
    def changer_deplacement(self):
        if self.deplacement == 'droite':
            self.deplacement = 'gauche'
        elif self.deplacement == 'gauche':
            self.deplacement = 'droite'
        elif self.deplacement == 'haut':
            self.deplacement = 'bas'
        elif self.deplacement == 'bas':
            self.deplacement = 'haut'
    
    #retourne une liste des noms des clefs dont le pêne est sorti
    def verifier_penne_sorti(self):
        return [clef.nom for clef in self.clefs if clef.penne == 'sorti']

    #retourne une liste des noms des clefs dont le pêne est rentré
    def verifier_penne_rentre(self):
        return [clef.nom for clef in self.clefs if clef.penne == 'rentre']
    
    #cherche une clef via son nom
    def chercher_clef_par_nom(self, clef_nom):
        for clef in self.clefs:
            if clef.nom == clef_nom:
                return clef
        return None
    
    #cherche une clef via son nom et lui change son état
    def changer_etat_clef(self, clef_nom, nouvel_etat):
        clef_trouvee = None
        for clef in self.clefs:
            if clef.nom == clef_nom and clef.etat != nouvel_etat:
                clef_trouvee = clef
                break
        
        #Vérifie si la clef a été trouvée et changer son état
        if clef_trouvee != None:
            clef_trouvee.changer_etat(nouvel_etat)
     
    #cherche une clef via son nom et lui change son pêne
    def changer_etat_penne(self, clef_nom, nouvel_etat):
        clef_trouvee = None
        for clef in self.clefs:
            if clef.nom == clef_nom and clef.penne != nouvel_etat:
                clef_trouvee = clef
                break
        
        #Vérifie si la clef a été trouvée et changer son état
        if clef_trouvee != None:
            clef_trouvee.changer_penne(nouvel_etat)
            
    #renvoie deux listes : une contenant les (objets) clefs absentes et l'autre le nom des clefs absentes
    def clef_absente(self):
        clef_absente = [clef for clef in self.clefs if clef.est_absente()]
        nom_clef_absente = [clef.nom for clef in self.clefs if clef.est_absente()]
        return clef_absente, nom_clef_absente
    
    #renvoie deux listes : une contenant les clefs prisonnières et l'autre le nom des clefs prisonnières
    def clef_prisonniere(self):
        clef_prisonniere = [clef for clef in self.clefs if clef.est_prisonniere()]
        nom_clef_prisonniere = [clef.nom for clef in self.clefs if clef.est_prisonniere()]
        return clef_prisonniere, nom_clef_prisonniere
    
    #renvoie la position de l'élément électrique dans une certaine matrice
    def position_element(self, matrice):
        x=0
        y=0
        for i in range(len(matrice)):
            for j in range(len(matrice[0])):
                if matrice[i][j] == self.type:
                    x = i
                    y = j
                    return (x,y)
