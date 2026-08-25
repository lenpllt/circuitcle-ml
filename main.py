import tkinter as tk
from tkinter import PhotoImage
from Classe_executionLHC import ExecutionLHC
from Classe_executionLHT import ExecutionLHT
import Classe_modifs_excel_LHC
import Classe_modifs_excel_LHT

def run_lhc_table():
    app = ExecutionLHC(root)
    app.execution(root)

def run_modifs_lhc():
    Classe_modifs_excel_LHC.ModifsExcel(root)  # Appel à la fonction main() du module Classe_modifs_excel_LHC

def run_lht_table():
    app = ExecutionLHT(root)
    app.execution(root)

def run_modifs_lht():
    Classe_modifs_excel_LHT.ModifsExcel(root)  # Appel à la fonction main() du module Classe_modifs_excel_LHT

# Création de la fenêtre principale
root = tk.Tk()
root.geometry("300x200")
root.title("CIRCUIT CLE")
icon = PhotoImage(file='Logo_sans_fond.png')
root.iconphoto(False, icon)
root.iconbitmap('Logo_avec_fond.ico')

# Création des boutons avec des libellés personnalisés
buttons_info = [
    ("Tableau LHC", run_lhc_table),
    ("Modification du tableau LHC", run_modifs_lhc),
    ("Tableau LHT", run_lht_table),
    ("Modification du tableau LHT", run_modifs_lht)
]

for text, command in buttons_info:
    btn = tk.Button(root, text=text, command=command)
    btn.pack(pady=10)

# Lancement de la boucle principale Tkinter
root.mainloop()
