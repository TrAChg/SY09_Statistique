"""
B03 — ACP: axes principaux, cercle des corrélations, Dalc/Walc en supplémentaire
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import save_plot, print_section, BASE_DIR

print_section("B03 — ACP")

df_encoded = pd.read_csv(os.path.join(BASE_DIR, "student-alcohol-encoded.csv"))
df_raw = pd.read_csv(os.path.join(BASE_DIR, "student-alcohol.csv"))

active_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
               'famrel', 'freetime', 'goout', 'health', 'absences']
suppl_cols = ['Dalc', 'Walc', 'G1', 'G2', 'G3']
print(f"  Actives: {len(active_cols)} | Supplémentaires: {suppl_cols}")

X_scaled = StandardScaler().fit_transform(df_encoded[active_cols].values)
X_suppl = df_encoded[suppl_cols].values

pca = PCA()
X_proj = pca.fit_transform(X_scaled)
eigenvalues = pca.explained_variance_
explained = pca.explained_variance_ratio_
cumul = np.cumsum(explained)

print(f"  Kaiser (λ>1): {(eigenvalues > 1).sum()} axes")
print(f"  80% variance: {np.argmax(cumul >= 0.8) + 1} axes")
for k in range(5):
    print(f"    PC{k+1}: λ={eigenvalues[k]:.3f} ({explained[k]*100:.1f}%, cum={cumul[k]*100:.1f}%)")

# Nombre d'axes à interpréter = critère de Kaiser (objectif)
n_axes = int((eigenvalues > 1).sum())
print(f"\n  → On interprète {n_axes} axes (critère de Kaiser: λ>1)")
print(f"    Justification: un axe avec λ>1 résume plus d'info qu'une variable seule")

# Scree plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(range(1, len(eigenvalues)+1), eigenvalues, color='steelblue')
axes[0].axhline(1, color='red', ls='--', label='Kaiser')
axes[0].set_xlabel('Axe'); axes[0].set_ylabel('λ'); axes[0].legend()
axes[1].plot(range(1, len(cumul)+1), cumul*100, 'o-', color='coral')
axes[1].axhline(80, color='gray', ls='--', label='80%')
axes[1].set_xlabel('Axe'); axes[1].set_ylabel('Cumul %'); axes[1].legend()
plt.tight_layout()
save_plot("B03_scree_plot.png")

# Cercle des corrélations
active_corr = np.corrcoef(X_scaled.T, X_proj.T)[:len(active_cols), len(active_cols):]
suppl_corr = np.corrcoef(X_suppl.T, X_proj.T)[:len(suppl_cols), len(suppl_cols):]

# Affichage texte des loadings (pour analyse sans image) — sur n_axes (Kaiser)
print_section(f"CERCLE DES CORRÉLATIONS — Loadings (PC1..PC{n_axes})")
header = "  " + f"{'variable':<12}" + "".join(f"{'PC'+str(k+1):>7}" for k in range(n_axes)) + f"{'cos²(PC1-2)':>13}"
print("  Variables ACTIVES:")
print(header)
for i, v in enumerate(active_cols):
    cos2 = active_corr[i, 0]**2 + active_corr[i, 1]**2
    line = f"  {v:<12}" + "".join(f"{active_corr[i,k]:>7.3f}" for k in range(n_axes)) + f"{cos2:>13.3f}"
    print(line)

print("\n  Variables SUPPLÉMENTAIRES (Dalc/Walc + notes):")
print(header)
for i, v in enumerate(suppl_cols):
    cos2 = suppl_corr[i, 0]**2 + suppl_corr[i, 1]**2
    line = f"  {v:<12}" + "".join(f"{suppl_corr[i,k]:>7.3f}" for k in range(n_axes)) + f"{cos2:>13.3f}"
    print(line)

# Interprétation des axes: seuil = sqrt(1/p) (contribution uniforme, justifié)
seuil = np.sqrt(1 / len(active_cols))
print_section(f"INTERPRÉTATION DES AXES (|loading| > {seuil:.3f} = √(1/p), p={len(active_cols)})")
print(f"  Justification du seuil: si les {len(active_cols)} variables contribuaient")
print(f"  également à un axe, chacune aurait |loading|=√(1/p)={seuil:.3f}.")
print(f"  Une variable dépasse ce seuil = contribution supérieure à la moyenne.\n")
for pc in range(n_axes):
    dominant = [(active_cols[i], active_corr[i, pc]) for i in range(len(active_cols))
                if abs(active_corr[i, pc]) > seuil]
    dominant.sort(key=lambda x: abs(x[1]), reverse=True)
    desc = ", ".join(f"{v}({c:+.2f})" for v, c in dominant)
    print(f"  PC{pc+1} ({explained[pc]*100:.1f}%): {desc if desc else 'aucune variable dominante'}")


fig, ax = plt.subplots(figsize=(10, 10))
circle = plt.Circle((0, 0), 1, fill=False, color='gray', ls='--')
ax.add_patch(circle)
for i, v in enumerate(active_cols):
    ax.arrow(0, 0, active_corr[i, 0], active_corr[i, 1],
             head_width=0.03, fc='steelblue', ec='steelblue', alpha=0.7)
    ax.text(active_corr[i, 0]*1.12, active_corr[i, 1]*1.12, v,
            color='steelblue', fontsize=9, ha='center')
for i, v in enumerate(suppl_cols):
    ax.arrow(0, 0, suppl_corr[i, 0], suppl_corr[i, 1],
             head_width=0.04, fc='red', ec='red', lw=2.5)
    ax.text(suppl_corr[i, 0]*1.15, suppl_corr[i, 1]*1.15, v,
            color='red', fontsize=11, fontweight='bold', ha='center')
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}%)')
ax.set_title('Cercle des corrélations (bleu=actives, rouge=supplémentaires)')
ax.set_aspect('equal')
plt.tight_layout()
save_plot("B03_correlation_circle.png")

# Projection individus
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, col, label in zip(axes, ['Dalc', 'Walc'], ['Dalc', 'Walc']):
    sc = ax.scatter(X_proj[:, 0], X_proj[:, 1], c=df_raw[col], cmap='YlOrRd',
                    alpha=0.6, s=20, edgecolors='black', linewidth=0.2)
    ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}%)')
    ax.set_title(f'Individus — coloré par {label}')
    plt.colorbar(sc, ax=ax, label=label)
plt.tight_layout()
save_plot("B03_individuals_projection.png")

# Sauvegarde
pd.DataFrame(X_proj[:, :10], columns=[f'PC{i+1}' for i in range(10)]).to_csv(
    os.path.join(BASE_DIR, "student-alcohol-pca.csv"), index=False)
print(f"  Sauvé: student-alcohol-pca.csv (10 axes)")

# Lien entre coordonnées ACP et alcool (corrélation des PC avec Dalc/Walc)
print_section("LIEN AXES ACP ↔ ALCOOL")
print(f"  {'axe':<6} {'corr Dalc':>10} {'corr Walc':>10}")
for pc in range(5):
    r_dalc = np.corrcoef(X_proj[:, pc], df_raw['Dalc'])[0, 1]
    r_walc = np.corrcoef(X_proj[:, pc], df_raw['Walc'])[0, 1]
    print(f"  PC{pc+1:<4} {r_dalc:>10.3f} {r_walc:>10.3f}")
print("  → Si toutes les corrélations sont faibles (<0.3), l'alcool n'est PAS")
print("    porté par les axes de contexte → confirme qu'il est un signal secondaire")

