-- =============================================================================
-- CircuitClé V3.5 — Documentation schéma SQLite
-- EDF / DIPDE — M2 Data Science — Léna Pillet — 2026
-- =============================================================================
--
-- La base de données est SQLite IN-MEMORY : sqlite3.connect(':memory:')
-- Elle est reconstruite à chaque session depuis les fichiers Excel (dataLHC/ / dataLHT/).
-- Il n'existe pas de fichier .db persistant.
--
-- Source : LHC_Classe_initialisation_cellules.py (méthode create_sqlite_db_from_excel)
--          LHT_Classe_initialisation_cellules.py
--
-- Chaque feuille Excel du palier sélectionné (ex. 900_preVD_LHC.xlsx) devient
-- une table SQLite. Les tables ci-dessous correspondent aux feuilles standard.
-- =============================================================================


-- Table : clef
-- Clefs électriques associées à chaque cellule HTA
CREATE TABLE IF NOT EXISTS clef (
    cellule         TEXT,    -- Identifiant de la cellule (ex. LHC005JA)
    numero          TEXT,    -- Numéro de la clef
    etat            TEXT,    -- État de la clef (libre / engagée / prisonnière)
    type_clef       TEXT,    -- Type de clef (mécanique, SMALT, etc.)
    equipement      TEXT     -- Équipement associé
);
CREATE INDEX IF NOT EXISTS idx_clef_cellule ON clef(cellule);


-- Table : partie_mobile
-- Parties mobiles (débrochables) par cellule
CREATE TABLE IF NOT EXISTS partie_mobile (
    cellule         TEXT,    -- Identifiant de la cellule
    position        TEXT,    -- Position (embrochée / débrochée / test)
    equipement      TEXT,    -- Identifiant de l'équipement
    tension         TEXT     -- Niveau de tension
);
CREATE INDEX IF NOT EXISTS idx_partie_mobile_cellule ON partie_mobile(cellule);


-- Table : smalt
-- Système de Mise à la Terre (SMALT) par cellule
CREATE TABLE IF NOT EXISTS smalt (
    cellule         TEXT,    -- Identifiant de la cellule
    etat            TEXT,    -- État du SMALT (posé / non posé)
    emplacement     TEXT,    -- Emplacement dans le tableau
    numero          TEXT     -- Numéro du SMALT
);
CREATE INDEX IF NOT EXISTS idx_smalt_cellule ON smalt(cellule);


-- Table : cellule
-- Données générales des cellules HTA du tableau
CREATE TABLE IF NOT EXISTS cellule (
    nom             TEXT,    -- Identifiant de la cellule
    type_cellule    TEXT,    -- Type (arrivée, départ, couplage, etc.)
    tension         TEXT,    -- Tension nominale (kV)
    tableau         TEXT     -- Identifiant du tableau parent
);


-- Table : transformateur
-- Transformateurs associés aux cellules
CREATE TABLE IF NOT EXISTS transformateur (
    cellule         TEXT,    -- Cellule associée
    puissance       TEXT,    -- Puissance (kVA)
    rapport         TEXT,    -- Rapport de transformation
    etat            TEXT     -- État (en service / hors tension)
);


-- Table : coffret
-- Coffrets de commande et de protection
CREATE TABLE IF NOT EXISTS coffret (
    cellule         TEXT,    -- Cellule associée
    type_coffret    TEXT,    -- Type de coffret
    etat            TEXT     -- État (fermé / ouvert / condamné)
);


-- Table : panneau
-- Panneaux de signalisation et de consignation
CREATE TABLE IF NOT EXISTS panneau (
    cellule         TEXT,    -- Cellule associée
    type_panneau    TEXT,    -- Type de panneau (danger, interdiction, etc.)
    pose            TEXT     -- Posé (oui / non)
);


-- =============================================================================
-- Requêtes de référence utilisées dans le benchmark (benchmark_sql.py)
-- =============================================================================

-- Q1 : Clefs d'une cellule donnée
-- SELECT * FROM clef WHERE cellule = 'LHC005JA';

-- Q2 : Parties mobiles d'une cellule donnée
-- SELECT * FROM partie_mobile WHERE cellule = 'LHC005JA';

-- Q3 : SMALT d'une cellule donnée
-- SELECT * FROM smalt WHERE cellule = 'LHC005JA';

-- Résultats du benchmark (100 répétitions, voir benchmark_resultats.json) :
--   Gain moyen après indexation : > 50 % sur les trois requêtes
-- =============================================================================
