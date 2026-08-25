#on importe la classe PartieMobile classe mere de Eclisse
from LHT_Classe_partie_mobile import PartieMobile

#Creation d'une classe Eclisse, qui crée des objets éclisses 
class Eclisse(PartieMobile):
    
    #recense le nom d'eclisses disponibles a un instant donné (nul au depart car elles sont soit dans la boite à éclisses, soit insérées dans les cellules)
    eclisseA = 0
    eclisseB = 0
    
    #création des caractéristiques qui définissent une eclisse
    def __init__(self, nom,clefs, type, etat, cellule, representation, deplacement, position, appartenance, voie): 
        super().__init__(nom,clefs, type, etat, cellule, representation, deplacement, position, appartenance) #memes caractéristiques qu'une partie mobile
        self.voie = voie #voie A ou B, permet de savoir a quelle voie appartient l'eclisse

    #fonction permettant de mettre ou de recuperer l'eclissesur la cellule du tableau
    def ajout_suppression_eclisse(self, cellule):
        
        #si aucun pêne n'est sorti, on va pouvoir inserer ou enlever l'eclisse
        if self.verifier_penne_sorti() == []:
            if self.position == 'presente': 
                self.position = 'absente'
                #une eclisse devient disponible (voie A ou B selon le type)
                if self.voie == 'A': 
                    Eclisse.eclisseA += 1
                if self.voie == 'B':
                    Eclisse.eclisseB += 1
                 
                print("suppression", self.nom, self.voie, self.cellule)
                
            #si on veut ajouter l'eclisse, il faut regarder si on en a une de disponible
            elif self.position == 'absente':
                if self.voie == 'A':
                    if Eclisse.eclisseA != 0: # On met l'eclisse voie A si il y en a de dispos
                        Eclisse.position = 'presente'
                        Eclisse.eclisseA -= 1
                if self.voie == 'B':
                    if Eclisse.eclisseB != 0:
                        self.position = 'presente'
                        Eclisse.eclisseB -= 1               
                print("Ajout", self.nom, self.voie, self.cellule)
            return True   
        
        #si on peut pas realiser l'etape, on retourne False
        return False