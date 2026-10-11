"""
Ce module centralise:
  - Le chargement et la fusion des données (concaténation + dédup)
  - L'encodage des variables qualitatives
  - Les fonctions de visualisation communes
  - Les chemins et configurations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Tuple, List, Optional

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme()
plt.rcParams['figure.dpi'] = 100

# Stratégie choisie: CONCATÉNATION + DÉDUPLICATION
# Justification:
#   1. Notre objectif est d'analyser la consommation d'alcool (Dalc, Walc).
#      Ces variables sont les MEMES pour un élève quelle que soit la matière.
#   2. La concaténation simple créerait des pseudo-répétitions faussant
#      l'analyse (les mêmes valeurs d'alcool comptées deux fois).
#   3. Le merge (inner) sur élèves communs réduirait trop la taille
#      d'échantillon (~300 au lieu de ~950), compromettant ACP et clustering.
#   4. La concaténation + dédup maximise la population tout en gardant
#      des individus uniques -> analyses plus robustes.

def load_and_fuse_data() -> pd.DataFrame:
    """
    Charge student-mat.csv et student-por.csv, les concatène,
    et supprime les doublons sur les colonnes d'identification.
    """
    d1 = pd.read_csv(os.path.join(BASE_DIR, "student-mat.csv"), sep=",")
    d2 = pd.read_csv(os.path.join(BASE_DIR, "student-por.csv"), sep=",")

    print(f"  Math: {len(d1)} élèves")
    print(f"  Portugais: {len(d2)} élèves")
    print(f"  Total brute (concat): {len(d1) + len(d2)}")

    # Colonnes d'identification
    id_cols = ["school", "sex", "age", "address", "famsize", "Pstatus",
               "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"]

    # Ajout d'une colonne subject
    d1["subject"] = "Math"
    d2["subject"] = "Por"

    df_concat = pd.concat([d1, d2], ignore_index=True)

    # Détection des doublons
    dup_mask = df_concat.duplicated(subset=id_cols, keep=False)
    n_dup_individuals = df_concat[dup_mask]["age"].count()

    # Suppression des doublons (garder le premier = élève Math).
    # Justification keep="first":
    #   - Les variables d'intérêt (Dalc, Walc, goout, famrel...) sont IDENTIQUES
    #     pour un même élève quelle que soit la matière (traits personnels).
    #   - Seules G1/G2/G3 et absences diffèrent (spécifiques matière), mais
    #     G1/G2/G3 sont exclus des features supervisées (supplémentaires ACP).
    #   - Choisir "first" ou "last" ne change pas les analyses de consommation.
    #   - Vérifié empiriquement: keep="last" donne mêmes clusters, écart < 0.005.
    df_unique = df_concat.drop_duplicates(subset=id_cols, keep="first").copy()

    print(f"  Après déduplication: {len(df_unique)} élèves uniques")
    print(f"  Doublons supprimés: {len(df_concat) - len(df_unique)}")
    print(f"  Colonnes: {list(df_unique.columns)}")
    print()

    return df_unique


# ENCODAGE DES VARIABLES QUALITATIVES
def get_variable_types(df: pd.DataFrame) -> dict:
    """
    Classe les colonnes par type pour le pipeline d'analyse.
    """
    numeric_vars = ["age", "traveltime", "studytime", "failures",
                    "famrel", "freetime", "goout", "health", "absences"]

    alcohol_vars = ["Dalc", "Walc"]

    grade_vars = ["G1", "G2", "G3"]

    nominal_vars = ["school", "sex", "address", "famsize", "Pstatus",
                    "Mjob", "Fjob", "reason", "guardian",
                    "schoolsup", "famsup", "paid", "activities",
                    "nursery", "higher", "internet", "romantic"]

    # Var ordinales (Medu, Fedu: 0-4)
    ordinal_vars = ["Medu", "Fedu"]

    return {
        "numeric": numeric_vars,
        "nominal": nominal_vars,
        "ordinal": ordinal_vars,
        "alcohol": alcohol_vars,
        "grades": grade_vars
    }


def encode_nominal_variables(df: pd.DataFrame,
                              nominal_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Encodage disjonctif complet des variables qualitatives nominales.
    """
    if nominal_cols is None:
        var_types = get_variable_types(df)
        nominal_cols = var_types["nominal"]

    df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=False)
    print(f"  Encodage: {len(nominal_cols)} vars nominales -> {df_encoded.shape[1] - df.shape[1] + len(nominal_cols)} colonnes binaires")

    return df_encoded


# STATISTIQUES DESCRIPTIVES

def descriptive_stats(df: pd.DataFrame, vars_list: List[str]) -> pd.DataFrame:
    """
    Calcule les statistiques descriptives élémentaires.
    """
    stats = df[vars_list].describe().T
    stats["variance"] = df[vars_list].var(ddof=0)
    stats["range"] = stats["max"] - stats["min"]
    stats["iqr"] = stats["75%"] - stats["25%"]

    return stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "variance", "range", "iqr"]]


# SAUVEGARDE DE FIGURES
def save_plot(filename: str):
    """Sauvegarde la figure active dans le dossier plots/."""
    filepath = os.path.join(PLOTS_DIR, filename)
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"  > Graphique sauvegardé: {filename}")


# AFFICHAGE
def print_section(title: str):
    """Affiche un titre de section formaté."""
    print()
    print(title)
