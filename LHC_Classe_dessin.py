import tkinter as tk

#classe dessin : créé des objets Dessin pour dessiner les cellules électriques
class Dessin:
    
    #Caractéristiques du dessin 
    def __init__(self, canvas):
        self.canvas = canvas #feuille de dessin
        #represente le zoom
        self.facteur_zoom = 1.0
        self.zoomer = 1.0
        #lie les boutons du clavier qui permettront de zoomer : + et -
        self.canvas.bind_all("<KeyPress-+>", self.zoom)
        self.canvas.bind_all("<KeyPress-minus>", self.dezoom)
        #lie les boutons du clavier qui permettront de se déplacer : les flèches
        self.canvas.bind_all("<Left>", self.deplacement_gauche)
        self.canvas.bind_all("<Right>", self.deplacement_droite)
        self.canvas.bind_all("<Up>", self.deplacement_haut)
        self.canvas.bind_all("<Down>", self.deplacement_bas)
        self.canvas.focus_set()  # Assure que le Canvas reçoit le focus pour les événements clavier
        self.taille_texte = {}
        
        
        '''a savoir !
        create_rectangle(x1, y1, x2, y2) créé un rectancle depuis le coin supérieur gauche (x1,y1) vers le coin inférieur droit (x2,y2) 
        create_line(x1, y1, x2, y2) créé une ligne de (x1,y1) jusqu'a (x2,y2)
        create_text(x,y, text=a) ecrit le texte nommé a centré en (x,y) 
        create_oval(x1, y1, x2, y2) créé un ovale contenu dans un rectangle create_rectangle(x1, y1, x2, y2)
        pour dessiner les clefs, toujours la meme methode, on dessine le carre de couleur et on ecirt le nom de la clef a l'interieur
        '''

    #fonction qui créé un rectancle depuis (x1,y1) l'angle haut gauche vers (x2,y2) l'angle bas droit
    def dessiner_rectangle(self, x1, y1, x2, y2):
        self.canvas.create_rectangle(x1, y1, x2, y2) #créé un rectancle depuis (x1,y1) l'angle haut gauche vers (x2,y2) l'angle bas droit
        
    #dessine un fil horizontal partant de (x,y) vers (x+taille, y)
    def dessiner_fil_h(self, x, y, taille):
        self.canvas.create_line(x, y, x + taille, y, fill='black', width=2) #créé une ligne de (x,y) vers (x+taille, y)
        
    #dessine un fil vertical partant de (x,y) vers (x, y+taille)
    def dessiner_fil_v(self, x, y, taille):
        self.canvas.create_line(x, y, x, y + taille, fill='black', width=2)
        
    #dessine un fil horizontal de la longueur de la case et un fil vertical partant du milieu de la case et dirigé vers le bas :  pour les case départ vers une cellule et liant le jeu de barres
    def dessiner_fil_h_dep(self, x, y, taille):
        self.canvas.create_line(x, y + taille/2, x + taille, y + taille/2, fill='black', width=2)
        self.canvas.create_line(x + taille/2, y + taille/2, x  + taille/2 , y + taille, fill='black', width=2)
     
    #dessin pour le depart vers le smalt
    def dessiner_fil_dep_smalt(self, x, y, taille):
        self.canvas.create_line(x + taille/2, y + taille/2, x + taille, y + taille/2, fill='black', width=2)
        self.canvas.create_line(x + taille/2, y, x  + taille/2 , y + taille, fill='black', width=2)
        
    #si on imagine un carre, ca dessine l'angle bas droit
    def dessiner_angle_bas_droit(self, x, y, taille):
        self.canvas.create_line(x, y + taille/2, x + taille/2, y + taille/2, fill='black', width=2)
        self.canvas.create_line(x + taille/2, y, x + taille/2 , y + taille/2, fill='black', width=2)
        
    #si on imagine un carre, ca dessine l'angle bas gauche
    def dessiner_angle_bas_gauche(self, x, y, taille):
        self.canvas.create_line(x + taille/2, y + taille/2, x + taille, y + taille/2, fill='black', width=2)
        self.canvas.create_line(x + taille/2, y, x  + taille/2 , y + taille/2, fill='black', width=2)
        
    #si on imagine un carre, ca dessine l'angle haut gauche
    def dessiner_angle_haut_gauche(self, x, y, taille):
        self.canvas.create_line(x + taille/2, y + taille/2, x + taille, y + taille/2, fill='black', width=2)
        self.canvas.create_line(x + taille/2, y + taille/2, x  + taille/2 , y + taille, fill='black', width=2)
        
    #si on imagine un carre, ca dessine l'angle haut droit
    def dessiner_angle_haut_droit(self, x, y, taille):
        self.canvas.create_line(x , y + taille/2, x + taille/2, y + taille/2, fill='black', width=2)
        self.canvas.create_line(x + taille/2, y + taille/2, x  + taille/2 , y + taille, fill='black', width=2)
        
    #dessine un croisement (un carrefour) : c'est une croix
    def dessiner_croisement(self, x, y, taille):
        self.canvas.create_line(x, y + taille/2, x + taille, y + taille/2, fill='black', width=2)
        self.canvas.create_line(x + taille/2, y, x  + taille/2 , y + taille, fill='black', width=2)
        
    #dessine un disjoncteur, en écrivant "disjoncteur" à gauche du dessin
    def dessiner_disjoncteur(self, x, y, taille):
        self.canvas.create_line(x, y, x , y + 2*taille/5, arrow=tk.FIRST, fill='black', width=2)
        self.canvas.create_line(x , y+2*taille/5, x + taille/5, y + 3*taille/5, fill='black', width=2)
        self.canvas.create_line(x , y + 3*taille/5, x , y + taille, arrow=tk.LAST, fill='black', width=2)
        self.canvas.create_text(x-2*taille/3 , y + taille/2, text='Disjoncteur', fill='black',font=("Arial", taille//5))
    
    #dessine un contacteur, en écrivant "contacteur" à gauche du dessin
    def dessiner_contacteur(self, x, y, taille):
        self.canvas.create_line(x, y, x , y + 2*taille/5, arrow=tk.FIRST, fill='black', width=2)
        self.dessiner_rectangle(x-taille/15, y+taille/6, x+taille/25,  y + taille/3)
        self.canvas.create_line(x , y+2*taille/5, x + taille/5, y + 3*taille/5, fill='black', width=2)
        self.canvas.create_line(x , y + 3*taille/5, x , y + taille, arrow=tk.LAST, fill='black', width=2)
        self.canvas.create_text(x-2*taille/3 , y + taille/2, text='Contacteur', fill='black',font=("Arial", taille//5))
    
    #dessine un fusible, en écrivant "fusible" à gauche du dessin
    def dessiner_fusible(self, x, y, taille):
        self.canvas.create_line(x, y, x , y + taille, arrow=tk.BOTH, fill='black', width=2)
        self.dessiner_rectangle(x-taille/15, y+taille/4, x+taille/25,  y + 3*taille/4)
        self.canvas.create_text(x-2*taille/3 , y + taille/2, tex='Fusible', fill='black',font=("Arial", taille//5))
    
    #dessine une terre, en écrivant "terre" en dessous du dessin
    def dessiner_terre(self, x, y, taille):
        self.canvas.create_line(x, y , x + 5*taille/9, y, fill='black', width=2)
        self.canvas.create_line(x + 5*taille/9, y-taille/3, x + 5*taille/9, y + taille/3, fill='black', width=2)
        self.canvas.create_line(x + 6*taille/9, y - taille/4, x + 6*taille/9 , y + taille/4, fill='black', width=2)
        self.canvas.create_line(x + 7*taille/9, y - taille/6, x + 7*taille/9 , y + taille/6, fill='black', width=2)
        self.canvas.create_text(x + 3*taille/4 , y + taille/2, text='Terre', fill='black',font=("Arial", taille//5))
    
    #dessine un tiroir, en écrivant "tiroir" à gauche du dessin
    def dessiner_tiroir(self, x, y, taille):
        self.canvas.create_line(x, y , x , y + taille, arrow=tk.BOTH, fill='black', width=2)
        self.canvas.create_text(x -taille/2, y + taille/2, text='Tiroir', fill='black',font=("Arial", taille//5))

    #dessine un cercle de centre (x, y) et de rayon "rayon"
    def dessiner_cercle(self, x, y, rayon):
        x1 = x - rayon
        y1 = y - rayon
        x2 = x + rayon
        y2 = y + rayon
        self.canvas.create_oval(x1, y1, x2, y2, outline="black", width=2)
        
    #dessine une source : c'est simplement un cercle avec marqué "source" à l'intérieur
    def dessiner_source(self, x, y, taille):
        self.dessiner_cercle(x + taille/2 , y + taille/2, taille / 2)
        self.canvas.create_text(x + taille/2, y + taille/2, text='Source', fill='black', font=("Arial", taille // 5))
        
    #dessine un transformateur avec ses clefs
    def dessiner_transformateur(self, transformateur, x, y, taille):
        rayon = taille/4
        self.canvas.create_line(x, y, x, y+rayon, fill='black', width=2)
        self.dessiner_cercle( x, y+2*rayon, rayon)
        self.dessiner_cercle(x, y+3*rayon, rayon) 
        i=1
        for clef in transformateur.clefs:
            clef_nom = clef.nom
            #dessine le carré de la clef avec la couleur représentant son etat
            self.dessiner_carre(x+rayon+taille/5,
                                    y+i*taille/(len(transformateur.clefs)+1),
                                    clef,
                                    2*taille/len(transformateur.clefs)
                                    )
            
            #ecrit le nom de la clef dans son carre
            self.canvas.create_text(x+rayon+taille/5,
                                    y+i*taille/(len(transformateur.clefs)+1),
                                    text=clef_nom,
                                    fill='black',
                                    font=("Arial", taille//(2*len(transformateur.clefs))))
            i+=1
        
        self.dessiner_rectangle(x-2*rayon, y, x + 3*rayon, y+5*rayon)
        
    #créé la couleur du carré en fonction de l'état de la clef
    def couleur_carre(self, clef):
        if clef.etat == 'prisonniere':
            return 'red'
        elif clef.etat == 'presente':
            return 'grey'
        else:
            return 'white'
        
    #dessine le carré de couleur représentant la clef et son état, en utilisant la fonction juste au dessus
    def dessiner_carre(self,x, y, clef, taille):
        self.canvas.create_rectangle(x-taille/5, y-taille/5, x+taille/5, y+taille/5, fill=self.couleur_carre(clef))
        
    #dessine le pêne des clefs d'un disjoncteur, en fonction de si il est rentré ou sorti
    def dessiner_penne_disj(self,x,y, clef, taille):
        if clef.penne == 'rentre': #si elle est rentree, c'est un petit rectangle
            self.canvas.create_rectangle(x-taille/5+taille/12, y-taille/5-taille/25, x+taille/5-taille/12, y-taille/5, fill='black')
        if clef.penne == 'sorti': #si elle sortie, le rectangle est plus grand
            self.canvas.create_rectangle(x-taille/5+taille/12, y-taille/5-taille/3, x+taille/5-taille/12, y-taille/5, fill='black')
            
    #dessine le pêne des clefs d'un smalt, en fonction de si il est rentré ou sorti
    def dessiner_penne_smalt(self,x,y, clef, taille):
        if clef.etat == 'prisonniere' or clef.etat == 'presente':
            self.canvas.create_rectangle(x-taille/5-taille/25, y-taille/5+taille/12, x-taille/5, y+taille/5-taille/12, fill='black')
        if clef.etat == 'absente':
            self.canvas.create_rectangle(x-taille/5-taille/3, y-taille/5+taille/12, x-taille/5, y+taille/5-taille/12, fill='black')
            
        
     #dessine un coffret : c'est un carré contenant les clefs 
    
    def dessiner_coffret(self, coffret, x, y, taille):
        self.canvas.create_rectangle(x, y, x + taille, y + taille, width = 2)
        for i in range(len(coffret.clefs)):
            self.dessiner_carre(
                x + taille/5,
                y + taille/5 + i*taille/(len(coffret.clefs)),
                coffret.clefs[i],
                taille
                )
            self.canvas.create_text(
                x + taille/5,
                y + taille/5 + i*taille/(len(coffret.clefs)),
                text=coffret.clefs[i].nom,
                font=("Arial", taille//5)
                )
            
    #dessine la serrure mère 
    def dessiner_serrure_mere(self, x, y, cellule, matrice, serrure, nombre_serrures, position, taille, taille_serrure_mere):
    
        for i in range(len(serrure.clefs)): #positionne les différentes clefs d'un même serrure mère les unes en dessous des autres
            self.dessiner_carre(
                           x + (position+1)*cellule.largeur*taille/(1+nombre_serrures),
                           y +taille_serrure_mere/10 + 9/10*taille_serrure_mere*(i+1)/(1+len(serrure.clefs)), 
                           serrure.chercher_clef_par_nom(serrure.clefs[i].nom),
                           taille_serrure_mere/3
                           )
            self.canvas.create_text(
                               x + (position+1)*cellule.largeur*taille/(1+nombre_serrures),
                               y +taille_serrure_mere/10 + 9/10*taille_serrure_mere*(i+1)/(1+len(serrure.clefs)),
                               text=serrure.clefs[i].nom,
                               font=("Arial", taille_serrure_mere//14)
                               )
        self.canvas.create_text(
                                 x + (position+1)*cellule.largeur*taille/(1+nombre_serrures),
                                 y + taille/5 ,
                                 text=serrure.nom,
                                 fill='black',
                                 font=("Arial", taille//5)
                                 )
          
    #dessine les éléments du disjoncteur : les clefs, les pênes et les blocages
    def dessiner_elements_disjoncteur(self, disjoncteur, i, j, matrice, taille, a,b, x_coin, y_coin):
        #on regarde la representation du verrouillage (matrice) pour la dessiner
        for ligne in range(len(matrice)):
            for colonne in range(len(matrice[0])):
                x = j*taille + x_coin + taille/2
                y = i*taille + y_coin
                if matrice[ligne][colonne] == 2: #si c'est un creux
                    self.dessiner_fil_v(
                                   x + colonne*taille/2 + a,
                                   y + ligne*taille/2 - taille/2 + b,
                                   taille/2
                                   )
                    self.dessiner_fil_h(
                                   x + colonne*taille/2 + a,
                                   y + ligne*taille/2-taille/2 + b,
                                   taille/2
                                   )
                    self.dessiner_fil_v(
                                   x + colonne*taille/2+ taille/2 + a,
                                   y + ligne*taille/2 - taille/2 + b ,
                                   taille/2
                                   )
                if matrice[ligne][colonne] == 1: #si c'est un blocage
                    self.dessiner_fil_h(
                                   x + colonne*taille/2 + a,
                                   y + ligne*taille/2 + b , 
                                   taille/2
                                   )
                if isinstance(matrice[ligne][colonne], str): #si c'est un nom de clef
                    #si c'est un nom de clef, on doit regarder si la manivelle est insérée et décaler la représentation en conséquence
                    if matrice[ligne-1][colonne] == 4:
                        self.dessiner_penne_disj(
                                       x + colonne*taille/2 +taille/4 + a,
                                       y + ligne*taille/2- taille/2 -2*taille/10 + b ,
                                       disjoncteur.chercher_clef_par_nom(matrice[ligne][colonne]),
                                       taille
                                       )
                        self.dessiner_carre(
                                       x + colonne*taille/2 +taille/4 + a,
                                       y + ligne*taille/2- taille/2 -2*taille/10 + b ,
                                       disjoncteur.chercher_clef_par_nom(matrice[ligne][colonne]),
                                       taille
                                       )
                        self.canvas.create_text(x + colonne*taille/2 +taille/4 + a,
                                           y + ligne*taille/2- taille/2 -2*taille/10 + b ,
                                           text=matrice[ligne][colonne],
                                           fill='black',
                                           font=("Arial", taille//5)
                                           )
                    else:
                        self.dessiner_penne_disj(
                                       x + colonne*taille/2 +taille/4 + a,
                                       y + ligne*taille/2 -taille/4 + b ,
                                       disjoncteur.chercher_clef_par_nom(matrice[ligne][colonne]),
                                       taille
                                       )
                        self.dessiner_carre(
                                       x + colonne*taille/2 +taille/4 + a,
                                       y + ligne*taille/2 -taille/4 + b ,
                                       disjoncteur.chercher_clef_par_nom(matrice[ligne][colonne]),
                                       taille
                                       )
                        self.canvas.create_text(x + colonne*taille/2+taille/4 + a,
                                           y + ligne*taille/2-taille/4 + b ,
                                           text=matrice[ligne][colonne],
                                           fill='black',
                                           font=("Arial", taille//5)
                                           )
                if matrice[ligne][colonne]  == 4: #si c'est une représentation de la manivelle
                    self.canvas.create_line(x + colonne*taille/2 + a,
                                       y + ligne*taille/2-9*taille/20 + b ,
                                       x + colonne*taille/2+ taille/2 + a,
                                       y + ligne*taille/2-9*taille/20 + b ,
                                       fill='red',
                                       width=2
                                       )
                
    #dessine les éléments du smalt : clefs, pênes, blocages : idem que pour le disjoncteur
    def dessiner_elements_smalt(self, smalt, i, j, matrice, taille, a, b, x_coin, y_coin):
        x_smalt = 2
        y_smalt = 1
        for ligne in range(len(matrice)):
            for colonne in range(len(matrice[0])):
                x = j*taille + x_coin
                y = i*taille + y_coin + taille/2
                if matrice[ligne][colonne] == 2:
                    self.dessiner_fil_h(
                                   x + (colonne-y_smalt)*taille/2 + a,
                                   y + (ligne-x_smalt-1)*taille/2 + b , 
                                   taille/2
                                   )
                    self.dessiner_fil_v(
                                   x + (colonne-y_smalt)*taille/2 + a,
                                   y + (ligne-x_smalt-1)*taille/2 + b , 
                                   taille/2
                                   )
                    self.dessiner_fil_h(
                                   x + (colonne-y_smalt)*taille/2 + a,
                                   y + (ligne-x_smalt)*taille/2 + b , 
                                   taille/2
                                   )
                if matrice[ligne][colonne] == 1:
                    self.dessiner_fil_v(
                                   x + (colonne-y_smalt+1)*taille/2 + a,
                                   y + (ligne-x_smalt-1)*taille/2 + b , 
                                   taille/2
                                   )
                if isinstance(matrice[ligne][colonne], str):
                    self.dessiner_penne_smalt(
                                   x + (colonne-y_smalt)*taille/2+taille/4 + a,
                                   y + (ligne-x_smalt)*taille/2-taille/4 + b ,
                                   smalt.chercher_clef_par_nom(matrice[ligne][colonne]),
                                   taille
                                   )
                    self.dessiner_carre(
                                   x + (colonne-y_smalt)*taille/2+taille/4 + a,
                                   y + (ligne-x_smalt)*taille/2-taille/4 + b ,
                                   smalt.chercher_clef_par_nom(matrice[ligne][colonne]),
                                   taille
                                   )
                    self.canvas.create_text(x + (colonne-y_smalt)*taille/2+taille/4 + a,
                                       y + (ligne-x_smalt)*taille/2-taille/4 + b ,
                                       text=matrice[ligne][colonne], 
                                       fill='black',
                                       font=("Arial", taille//5)
                                       )

    #effectue un zoom de 10% 
    def zoom(self, event=None):
        self.appliquer_zoom(1.1)  # Zoom avant de 10%
        
    #effectue un dézoom de 10%
    def dezoom(self, event=None):
        self.appliquer_zoom(0.9)  # Zoom arrière de 10%

     #fonction qui appliquera un zoom ou dezoom choisi
    def appliquer_zoom(self, zoom):
        self.facteur_zoom *= zoom

        #Applique le zoom à tous les éléments
        self.canvas.scale("all", 0, 0, zoom, zoom)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        #met à jour la taille du texte en proportion du zoom
        self.zoom_taille_texte(zoom)

    #permet de zoomer le texte de la même manière que le reste
    def zoom_taille_texte(self, zoom):
        for item_id in self.canvas.find_all():
            if self.canvas.type(item_id) == "text": #regarde si l'élément dans la feuille de dessin est du texte 
                taille_actuelle = self.canvas.itemcget(item_id, "font") #récupère la police du texte
                
                if isinstance(taille_actuelle, str):
                    
                    font_parts = taille_actuelle.split()
                    font_family = font_parts[0]
                    font_size = int(font_parts[1])
                    if item_id not in self.taille_texte:
                        self.taille_texte[item_id] = font_size

                    real_font_size = self.taille_texte[item_id]
                    
                    #on calcule la nouvelle taille avec la valeur de zoom
                    nouvelle_taille = real_font_size * zoom
                    nouvelle_taille = max(nouvelle_taille, 2)  # Taille minimale
                    
                    self.taille_texte[item_id] = nouvelle_taille
                    
                    #on applique la nouvelle taille (arrondie) à l'affichage
                    new_font = f"{font_family} {round(nouvelle_taille)}"
                    self.canvas.itemconfig(item_id, font=new_font)

#ces 4 fonctions permettent de se déplacer de gauche à droite sur le dessin via les flèches du clavier
    def deplacement_gauche(self, event):
        #Déplace le contenu du canvas vers la gauche
        self.canvas.xview_scroll(-1, "units")
     
    def deplacement_droite(self, event):
        #Déplace le contenu du canvas vers la droite
        self.canvas.xview_scroll(1, "units")
     
    def deplacement_haut(self, event):
        #Déplace le contenu du canvas vers le haut
        self.canvas.yview_scroll(-1, "units")
     
    def deplacement_bas(self, event):
        #Déplace le contenu du canvas vers le bas
        self.canvas.yview_scroll(1, "units")
