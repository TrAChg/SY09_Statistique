"""
B02 — REPRÉSENTATION EUCLIDIENNE: centrage-réduction, distances, inertie
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import get_variable_types, save_plot, print_section, BASE_DIR

print_section("B02 — REPRÉSENTATION EUCLIDIENNE")

df_encoded = pd.read_csv(os.path.join(BASE_DIR, "student-alcohol-encoded.csv"))
df_raw = pd.read_csv(os.path.join(BASE_DIR, "student-alcohol.csv"))
var_types = get_variable_types(df_raw)

# Variables quantitatives = NUMERIC (9) + ORDINAL (2: Medu, Fedu).
quantitative_vars = var_types["numeric"] + var_types["ordinal"]
print(f"  Variables actives ({len(quantitative_vars)}): {', '.join(quantitative_vars)}")

X = df_encoded[quantitative_vars].values
n, p = X.shape
means = np.mean(X, axis=0)
stds = np.std(X, axis=0, ddof=0)
X_scaled = (X - means) / stds

print(f"  {n} individus × {p} variables centrées-réduites")

pd.DataFrame(X_scaled, columns=quantitative_vars).to_csv(
    os.path.join(BASE_DIR, "student-alcohol-scaled.csv"), index=False)

print_section("INERTIE TOTALE")
g = np.mean(X_scaled, axis=0)
dist_to_center = np.sum((X_scaled - g)**2, axis=1)
inertia = np.mean(dist_to_center)
print(f"  I_total = Tr(V) = {inertia:.4f}")

print_section("DISTANCES ENTRE INDIVIDUS")
D = squareform(pdist(X_scaled, metric='euclidean'))
print(f"  Min={D[D>0].min():.3f}, Max={D.max():.3f}, Moy={D[D>0].mean():.3f}")

print_section("VISUALISATION")
idx_go, idx_ft = quantitative_vars.index('goout'), quantitative_vars.index('freetime')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sc = axes[0].scatter(X_scaled[:, idx_go], X_scaled[:, idx_ft],
                     c=df_raw['Dalc'].values, cmap='YlOrRd', alpha=0.6,
                     edgecolors='black', linewidth=0.3, s=30)
axes[0].set_xlabel('goout'); axes[0].set_ylabel('freetime')
axes[0].set_title('Nuage (goout × freetime, coloré par Dalc)')
plt.colorbar(sc, ax=axes[0], label='Dalc')

axes[1].hist(dist_to_center, bins=30, color='steelblue', edgecolor='white')
axes[1].axvline(inertia, color='red', ls='--', label=f'I={inertia:.2f}')
axes[1].set_xlabel('Distance² au centre'); axes[1].set_title('Distribution des distances')
axes[1].legend()
plt.tight_layout()
save_plot("02_data_cloud.png")

print_section("DISTANCES ENTRE VARIABLES")
corr = df_encoded[quantitative_vars].corr()
var_dist = 1 - np.abs(corr)
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(var_dist, dtype=bool))
sns.heatmap(var_dist, mask=mask, cmap='viridis', annot=True, fmt='.2f', square=True)
plt.title('Distance entre variables (1-|r|)')
plt.tight_layout()
save_plot("02_variable_distances.png")

pairs = var_dist.where(np.triu(np.ones(var_dist.shape), k=1).astype(bool)).unstack().dropna().sort_values()
print("  Paires les plus proches:")
for (v1, v2), d in pairs.head(5).items():
    print(f"    {v1} ↔ {v2}: {d:.3f}")
