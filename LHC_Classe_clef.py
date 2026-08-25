#classe clef : cree des objets clef
class Clef:
    
    #initialise toutes les caracteristiques que doit avoir une clef
    def __init__(self, nom, etat, penne, appartenance, type, cellule):
        self.nom = nom
        self.etat = etat
        self.penne = penne
        self.appartenance = appartenance
        self.type = type
        self.cellule = cellule
        self.utilisee = False
            
    #change l'etat du pêne de la clef : il devient rentré
    def rentrer_penne(self):
        self.penne = 'rentre'
        
     #change l'etat du pêne de la clef : il devient sorti
    def sortir_penne(self):
        self.penne = 'sorti'
        
    #retourne True si le pêne de la clef est rentré
    def est_rentre(self):
        return self.penne == 'rentre'
    
    #retourne True si le pêne de la clef est sorti
    def est_sorti(self):
        return self.penne == 'sorti'
    
    #retourne True si la clef est prisonnière
    def est_prisonniere(self):
        return self.etat == 'prisonniere'
    
    #retourne True si la clef est présente
    def est_presente(self):
        return self.etat == 'presente'
    
    #retourne True si la clef est absente
    def est_absente(self):
        return self.etat == 'absente'
    
    #change l'etat (prisonnière, présente, absente) d'une clef par un nouvel etat
    def changer_etat(self, nouvel_etat):
        self.etat = nouvel_etat
        self.utilisee = True
    
    #change l'etat du pêne (rentré, sorti) d'une clef
    def changer_penne(self, nouvel_etat_penne):
        self.penne = nouvel_etat_penne
        self.utilisee = True
