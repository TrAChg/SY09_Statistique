"""
A01 — PREPARATION DES DONNEES: fusion, deduplication, encodage, sauvegarde.
     Inclut verification des doublons (Math vs Por pour les memes eleves).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import (
    load_and_fuse_data, get_variable_types, encode_nominal_variables,
    descriptive_stats, save_plot, print_section, BASE_DIR
)

print_section("A01 — PREPARATION DES DONNEES")
df = load_and_fuse_data()

print_section("APERCU")
print(f"  {df.shape[0]} lignes x {df.shape[1]} colonnes")
print(f"  Valeurs manquantes: {df.isnull().sum().sum()}")

var_types = get_variable_types(df)
for t, v in var_types.items():
    print(f"  {t.upper()} ({len(v)}): {', '.join(v)}")

print_section("STATS DESCRIPTIVES")
all_quant = var_types["numeric"] + var_types["ordinal"] + var_types["alcohol"] + var_types["grades"]
print(descriptive_stats(df, all_quant).to_string())

print_section("DISTRIBUTION ALCOOL")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.histplot(df["Dalc"], bins=5, discrete=True, ax=axes[0, 0], color="steelblue", edgecolor="white")
axes[0, 0].set_title("Dalc (semaine)")
sns.histplot(df["Walc"], bins=5, discrete=True, ax=axes[0, 1], color="coral", edgecolor="white")
axes[0, 1].set_title("Walc (week-end)")
ct = pd.crosstab(df["Dalc"], df["Walc"], margins=True)
sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0])
axes[1, 0].set_title("Contingence Dalc x Walc")
total_alc = df["Dalc"] + df["Walc"]
sns.histplot(total_alc, bins=9, discrete=True, ax=axes[1, 1], color="purple", edgecolor="white")
axes[1, 1].set_title("Total (Dalc + Walc)")
plt.tight_layout()
save_plot("alcohol_distributions.png")

print(f"  Dalc — moy={df['Dalc'].mean():.2f}, med={df['Dalc'].median():.0f}")
print(f"  Walc — moy={df['Walc'].mean():.2f}, med={df['Walc'].median():.0f}")

print_section("ENCODAGE ET SAUVEGARDE")
df_encoded = encode_nominal_variables(df)
df_encoded.to_csv(os.path.join(BASE_DIR, "student-alcohol-encoded.csv"), index=False)
df.to_csv(os.path.join(BASE_DIR, "student-alcohol.csv"), index=False)
print(f"  student-alcohol.csv: {len(df)} lignes")
print(f"  student-alcohol-encoded.csv: {df_encoded.shape[1]} colonnes")

# =============================================================================
# VERIFICATION DES DOUBLONS: les eleves presents dans les 2 matieres
# ont-ils les memes reponses pour les variables personnelles?
# =============================================================================
print_section("VERIFICATION DOUBLONS (Math vs Por pour les memes eleves)")

d1 = pd.read_csv(os.path.join(BASE_DIR, "student-mat.csv"), sep=",")
d2 = pd.read_csv(os.path.join(BASE_DIR, "student-por.csv"), sep=",")

id_cols = ["school", "sex", "age", "address", "famsize", "Pstatus",
           "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"]

merged = d1.merge(d2, on=id_cols, suffixes=("_mat", "_por"))
print(f"  Eleves presents dans les 2 matieres: {len(merged)}")

non_id_cols = [c for c in d1.columns if c not in id_cols]
print(f"\n  {'Variable':15s} {'% identique':>12} {'diff_moy':>10} {'diff_max':>10} {'Conclusion'}")

for col in non_id_cols:
    col_mat = f"{col}_mat"
    col_por = f"{col}_por"
    if col_mat in merged.columns and col_por in merged.columns:
        if merged[col_mat].dtype in ['int64', 'float64']:
            pct_same = (merged[col_mat] == merged[col_por]).mean() * 100
            diff = (merged[col_mat] - merged[col_por]).abs()
            diff_moy = diff.mean()
            diff_max = diff.max()
            if pct_same >= 95:
                conclusion = "IDENTIQUE"
            elif pct_same >= 80:
                conclusion = "quasi-identique"
            else:
                conclusion = "DIFFERENT"
            print(f"  {col:15s} {pct_same:>11.1f}% {diff_moy:>10.3f} {diff_max:>10.0f}   {conclusion}")
        else:
            pct_same = (merged[col_mat] == merged[col_por]).mean() * 100
            if pct_same >= 95:
                conclusion = "IDENTIQUE"
            elif pct_same >= 80:
                conclusion = "quasi-identique"
            else:
                conclusion = "DIFFERENT"
            print(f"  {col:15s} {pct_same:>11.1f}%        —          —   {conclusion}")

print(f"\n  INTERPRETATION:")
print(f"    - Variables personnelles (Dalc, Walc, goout, famrel...): si % identique >= 90%,")
print(f"      keep='first' est sans consequence (meme reponse dans les 2 matieres).")
print(f"    - Variables scolaires (G1, G2, G3, absences): attendues DIFFERENTES")
print(f"      (specifiques a chaque matiere). Pas de probleme car G1/G2/G3 sont exclus")
print(f"      des features supervisees.")
print(f"    - Si Dalc/Walc sont identiques a >90%: CONFIRME que keep='first' est valide.")
