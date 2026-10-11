"""
B01 — ANALYSE EXPLORATOIRE: Dalc vs Walc + corrélations + profils qualitatifs
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import get_variable_types, save_plot, print_section, BASE_DIR

print_section("B01 — ANALYSE EXPLORATOIRE")
df = pd.read_csv(os.path.join(BASE_DIR, "student-alcohol.csv"))
var_types = get_variable_types(df)
print(f"  {len(df)} élèves, {df.shape[1]} colonnes")

all_quant = var_types["numeric"] + var_types["ordinal"] + var_types["alcohol"] + var_types["grades"]

# UNIVARIÉE — toutes les variables quantitatives
print_section("UNIVARIÉE")
n_vars = len(all_quant)
ncols = 4
nrows = (n_vars + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows))
for i, ax in enumerate(axes.flat):
    if i < n_vars:
        sns.boxplot(y=df[all_quant[i]], ax=ax, color="steelblue", width=0.3)
        ax.set_title(all_quant[i])
    else:
        ax.axis("off")
plt.tight_layout()
save_plot("univariate_boxplots.png")

print(f"  Corr Dalc-Walc: {df[['Dalc','Walc']].corr().iloc[0,1]:.3f}")
print(f"  Dalc — moy={df['Dalc'].mean():.2f}, med={df['Dalc'].median():.0f}")
print(f"  Walc — moy={df['Walc'].mean():.2f}, med={df['Walc'].median():.0f}")

# MATRICE DE CORRÉLATION
print_section("CORRÉLATION")
corr = df[all_quant].corr()
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="coolwarm", vmin=-1, vmax=1, center=0,
            annot=True, fmt=".2f", square=True, linewidths=0.5)
plt.title("Matrice de corrélation")
plt.tight_layout()
save_plot("correlation_matrix.png")

for target in ["Dalc", "Walc"]:
    top = corr[target].drop(["Dalc", "Walc"]).sort_values(key=abs, ascending=False).head(5)
    print(f"\n  Top corrélations avec {target}:")
    for v, r in top.items():
        print(f"    {v:15s}: {r:+.3f}")

# BIVARIÉE — contexte vs Dalc/Walc
print_section("BIVARIÉE")
context_vars = [v for v in var_types["numeric"] if v not in ["absences"]] + var_types["ordinal"]
ncols_biv = 3
nrows_biv = (len(context_vars) + ncols_biv - 1) // ncols_biv
fig, axes = plt.subplots(nrows_biv, ncols_biv, figsize=(15, 4 * nrows_biv))
for i, ax in enumerate(axes.flat):
    if i < len(context_vars):
        v = context_vars[i]
        means = df.groupby(v)[["Dalc", "Walc"]].mean()
        ax.plot(means.index, means["Dalc"], "o-", color="steelblue", label="Dalc")
        ax.plot(means.index, means["Walc"], "s-", color="coral", label="Walc")
        ax.set_title(v)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        ax.axis("off")
plt.tight_layout()
save_plot("bivariate_means.png")

# QUALITATIVES VS ALCOOL (χ² + Cramér's V, sans profils — ceux-ci seront définis en B05)
print_section("QUALITATIVES VS ALCOOL")
qual_vars = var_types["nominal"]

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.values.sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

for target in ["Dalc", "Walc"]:
    print(f"  Tests χ² + Cramér's V (variable × {target}):")
    rows = []
    for v in qual_vars:
        ct = pd.crosstab(df[v], df[target])
        chi2, p, _, _ = chi2_contingency(ct)
        v_cramer = cramers_v(ct)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        rows.append((v, chi2, p, v_cramer, sig))
    # tri par intensité (Cramér's V) décroissante
    for v, chi2, p, vc, sig in sorted(rows, key=lambda x: x[3], reverse=True):
        print(f"    {v:15s}: χ²={chi2:6.2f} p={p:.4f} V={vc:.3f} {sig}")
    print()

print_section("B01 TERMINÉ")
